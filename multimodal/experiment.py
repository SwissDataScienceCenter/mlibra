"""
experiment.py
Main experiment logic and classes for the multimodal MALDI + gene expression experiment.
"""
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import wandb
from config import MultiModalConfig
from l3di.lgpmultimodal import LGPMultiModal

def density_scatter(ax, x, y, x_min, x_max, y_min, y_max, bins=50, **kwargs):
    """
    Creates a scatter plot on the provided axis using a density-based color
    from a 2D histogram with square bins.
    """
    data_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_range = data_range / 2
    x_lim = (x_min, x_max)
    y_lim = (y_min, y_max)

    # Create bin edges
    xedges = np.linspace(x_lim[0], x_lim[1], bins + 1)
    yedges = np.linspace(y_lim[0], y_lim[1], bins + 1)

    # Determine the bin index for each point
    x_bin = np.digitize(x, xedges) - 1
    y_bin = np.digitize(y, yedges) - 1

    # Build a 2D histogram for the points
    counts = np.zeros((bins, bins))
    for xb, yb in zip(x_bin, y_bin):
        if xb == bins:
            xb = bins - 1
        if yb == bins:
            yb = bins - 1
        if 0 <= xb < bins and 0 <= yb < bins:
            counts[yb, xb] += 1

    # Get density for each point from its bin
    density = np.array(
        [
            counts[yb if yb < bins else bins - 1, xb if xb < bins else bins - 1]
            for xb, yb in zip(x_bin, y_bin)
        ]
    )

    # Create the scatter plot with density coloring using the inferno colormap
    sc = ax.scatter(x, y, c=density, cmap="inferno", **kwargs)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    return sc

class MultiModalExperiment:
    def __init__(self, config: MultiModalConfig, multimodal_model: LGPMultiModal, coord_mean: torch.Tensor, coord_std: torch.Tensor):
        self.config = config
        self.coord_mean = coord_mean
        self.coord_std = coord_std

        self.train_filter = config.section_filter
        self.test_filter = config.test_filter
        self.multimodal_model = multimodal_model

        # Data storage
        self.coordinates_train = None
        self.coordinates_test = None
        
        # Modality 1: MALDI data
        self.maldi_train_data_original = None
        self.maldi_test_data_original = None
        self.maldi_train_data = None
        self.maldi_test_data = None
        self.maldi_col_means = None
        self.maldi_col_stds = None
        
        # Modality 2: Gene expression data
        self.gene_train_data_original = None
        self.gene_test_data_original = None
        self.gene_train_data = None
        self.gene_test_data = None
        self.gene_col_means = None
        self.gene_col_stds = None
        
        # Other data
        self.pixel_coordinates_train = None
        self.pixel_coordinates_test = None
        self.current_epoch = 0

    def load_train_sections(self):
        self.train_sections = pd.read_parquet(self.config.maldi_file,
                                                  columns=["Section"],
                                                  filters=self.train_filter)

    def load_train_pixel_coordinates(self):
        logging.info("Loading training pixel coordinates")
        coordinates_names = ["x", "y"]
        self.pixel_coordinates_train = pd.read_parquet(self.config.maldi_file,
                                                    columns=coordinates_names,
                                                    filters=self.train_filter).values

    def load_train_samples(self):
        logging.info("Loading training samples")
        self.train_samples = pd.read_parquet(self.config.maldi_file,
                                             columns=["Sample"],
                                             filters=self.train_filter)

    def load_maldi_train_data(self):
        """Load and preprocess MALDI training data (modality 1)"""
        logging.info("Loading MALDI training data")
        self.maldi_train_data = pd.read_parquet(self.config.maldi_file,
                                          columns=[str(i) for i in self.config.selected_lipids_names],
                                          filters=self.train_filter).values
        self.maldi_train_data = torch.tensor(self.maldi_train_data, dtype=torch.float32)
        assert self.maldi_train_data.shape[0] > 0, "No MALDI training data found with the given filter"
        
        # Handle NaNs and negative values
        n_nans = torch.isnan(self.maldi_train_data).sum()
        assert not torch.isnan(self.maldi_train_data).any(), f"MALDI training data contains NaNs: {n_nans} NaNs found"
        
        n_negatives = (self.maldi_train_data < 0).sum()
        logging.info(f"MALDI training data contains {n_negatives} negative values")
        logging.info("Imputing negative values to zero in MALDI data")
        self.maldi_train_data[self.maldi_train_data < 0] = 0

        # Log transformation if specified
        if self.config.log_transform:
            logging.info("Applying log transformation to MALDI training data")
            self.maldi_train_data = torch.log(self.maldi_train_data + 1e-10)
        else:
            logging.info("Skipping log transformation for MALDI data")
        
        logging.info(f"MALDI training data shape: {self.maldi_train_data.shape}")

        # Normalization
        logging.info("Normalizing MALDI training data")
        if (self.config.exp_path / "maldi_means.pth").exists() and (self.config.exp_path / "maldi_stds.pth").exists():
            logging.info("Loading MALDI column mean and std from files")
            maldi_col_means = torch.load(self.config.exp_path / "maldi_means.pth")
            maldi_col_stds = torch.load(self.config.exp_path / "maldi_stds.pth")
        else:
            logging.info("Computing MALDI channels means and stds")
            maldi_col_means = self.maldi_train_data.mean(dim=0)
            maldi_col_stds = self.maldi_train_data.std(dim=0)
            torch.save(maldi_col_means, self.config.exp_path / "maldi_means.pth")
            torch.save(maldi_col_stds, self.config.exp_path / "maldi_stds.pth")
        
        assert not torch.isnan(maldi_col_means).any(), "There are nans in MALDI col_means"
        assert not torch.isnan(maldi_col_stds).any(), "There are nans in MALDI col_stds"
        self.maldi_col_means = maldi_col_means
        self.maldi_col_stds = maldi_col_stds
        self.maldi_train_data = (self.maldi_train_data - maldi_col_means) / maldi_col_stds
        
        n_nans = torch.isnan(self.maldi_train_data).sum()
        assert not torch.isnan(self.maldi_train_data).any(), f"MALDI training data contains NaNs after normalization: {n_nans} NaNs found"
        logging.info("MALDI training data normalization successful")

    def load_gene_train_data(self):
        """Load and preprocess gene expression training data (modality 2)"""
        logging.info("Loading gene expression training data")
        self.gene_train_data = pd.read_parquet(self.config.gene_file,
                                         columns=[str(i) for i in self.config.selected_genes_names],
                                         filters=self.train_filter).values
        self.gene_train_data = torch.tensor(self.gene_train_data, dtype=torch.float32)
        assert self.gene_train_data.shape[0] > 0, "No gene expression training data found with the given filter"
        
        # Handle NaNs and negative values
        n_nans = torch.isnan(self.gene_train_data).sum()
        assert not torch.isnan(self.gene_train_data).any(), f"Gene expression training data contains NaNs: {n_nans} NaNs found"
        
        n_negatives = (self.gene_train_data < 0).sum()
        logging.info(f"Gene expression training data contains {n_negatives} negative values")
        logging.info("Imputing negative values to zero in gene expression data")
        self.gene_train_data[self.gene_train_data < 0] = 0

        # Log transformation if specified
        if self.config.log_transform:
            logging.info("Applying log transformation to gene expression training data")
            self.gene_train_data = torch.log(self.gene_train_data + 1e-10)
        else:
            logging.info("Skipping log transformation for gene expression data")
        
        logging.info(f"Gene expression training data shape: {self.gene_train_data.shape}")

        # Normalization
        logging.info("Normalizing gene expression training data")
        if (self.config.exp_path / "gene_means.pth").exists() and (self.config.exp_path / "gene_stds.pth").exists():
            logging.info("Loading gene expression column mean and std from files")
            gene_col_means = torch.load(self.config.exp_path / "gene_means.pth")
            gene_col_stds = torch.load(self.config.exp_path / "gene_stds.pth")
        else:
            logging.info("Computing gene expression channels means and stds")
            gene_col_means = self.gene_train_data.mean(dim=0)
            gene_col_stds = self.gene_train_data.std(dim=0)
            torch.save(gene_col_means, self.config.exp_path / "gene_means.pth")
            torch.save(gene_col_stds, self.config.exp_path / "gene_stds.pth")
        
        assert not torch.isnan(gene_col_means).any(), "There are nans in gene expression col_means"
        assert not torch.isnan(gene_col_stds).any(), "There are nans in gene expression col_stds"
        self.gene_col_means = gene_col_means
        self.gene_col_stds = gene_col_stds
        self.gene_train_data = (self.gene_train_data - gene_col_means) / gene_col_stds
        
        n_nans = torch.isnan(self.gene_train_data).sum()
        assert not torch.isnan(self.gene_train_data).any(), f"Gene expression training data contains NaNs after normalization: {n_nans} NaNs found"
        logging.info("Gene expression training data normalization successful")

    def load_train_data(self):
        """Load both MALDI and gene expression training data and coordinates"""
        self.load_maldi_train_data()
        self.load_gene_train_data()
        
        # Load coordinates
        logging.info("Loading coordinates")
        coordinates_names = ["xccf", "yccf", "zccf"]
        self.coordinates_train = pd.read_parquet(self.config.maldi_file,
                                                 columns=coordinates_names,
                                                 filters=self.train_filter).values
        self.coordinates_train = torch.tensor(self.coordinates_train, dtype=torch.float32)
        logging.info(f"Coordinates shape: {self.coordinates_train.shape}")
        
        # Normalize coordinates
        logging.info("Normalizing coordinates")
        self.coordinates_train = (self.coordinates_train - self.coord_mean) / self.coord_std
        n_nans = torch.isnan(self.coordinates_train).sum()
        assert not torch.isnan(self.coordinates_train).any(), f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
        logging.info("Coordinates normalization successful")

        # Create dataset with both modalities
        self.dataset_train = torch.utils.data.TensorDataset(
            self.maldi_train_data, 
            self.gene_train_data, 
            self.coordinates_train
        )

    def load_maldi_test_data(self):
        """Load and preprocess MALDI test data"""
        logging.info("Loading MALDI test data")
        self.maldi_test_data = pd.read_parquet(self.config.maldi_file,
                                         columns=[str(i) for i in self.config.selected_lipids_names],
                                         filters=self.test_filter).values
        self.maldi_test_data = torch.tensor(self.maldi_test_data, dtype=torch.float32)
        assert self.maldi_test_data.shape[0] > 0, "No MALDI test data found with the given filter"
        
        # Handle NaNs and negative values
        n_nans = torch.isnan(self.maldi_test_data).sum()
        assert not torch.isnan(self.maldi_test_data).any(), f"MALDI test data contains NaNs: {n_nans} NaNs found"
        
        n_negatives = (self.maldi_test_data < 0).sum()
        logging.info(f"MALDI test data contains {n_negatives} negative values")
        self.maldi_test_data[self.maldi_test_data < 0] = 0

        # Log transformation if specified
        if self.config.log_transform:
            logging.info("Applying log transformation to MALDI test data")
            self.maldi_test_data = torch.log(self.maldi_test_data + 1e-10)
        
        # Normalization using training statistics
        logging.info("Normalizing MALDI test data using training statistics")
        self.maldi_test_data = (self.maldi_test_data - self.maldi_col_means) / self.maldi_col_stds

    def load_gene_test_data(self):
        """Load and preprocess gene expression test data"""
        logging.info("Loading gene expression test data")
        self.gene_test_data = pd.read_parquet(self.config.gene_file,
                                        columns=[str(i) for i in self.config.selected_genes_names],
                                        filters=self.test_filter).values
        self.gene_test_data = torch.tensor(self.gene_test_data, dtype=torch.float32)
        assert self.gene_test_data.shape[0] > 0, "No gene expression test data found with the given filter"
        
        # Handle NaNs and negative values
        n_nans = torch.isnan(self.gene_test_data).sum()
        assert not torch.isnan(self.gene_test_data).any(), f"Gene expression test data contains NaNs: {n_nans} NaNs found"
        
        n_negatives = (self.gene_test_data < 0).sum()
        logging.info(f"Gene expression test data contains {n_negatives} negative values")
        self.gene_test_data[self.gene_test_data < 0] = 0

        # Log transformation if specified
        if self.config.log_transform:
            logging.info("Applying log transformation to gene expression test data")
            self.gene_test_data = torch.log(self.gene_test_data + 1e-10)
        
        # Normalization using training statistics
        logging.info("Normalizing gene expression test data using training statistics")
        self.gene_test_data = (self.gene_test_data - self.gene_col_means) / self.gene_col_stds

    def load_test_data(self):
        """Load both MALDI and gene expression test data"""
        self.load_maldi_test_data()
        self.load_gene_test_data()

    def load_coord_train_data(self):
        if self.coordinates_train is None:
            logging.info("Loading coordinates for training data")
            coordinates_names = ["xccf", "yccf", "zccf"]
            self.coordinates_train = pd.read_parquet(self.config.maldi_file,
                                                     columns=coordinates_names,
                                                     filters=self.train_filter).values
            self.coordinates_train = torch.tensor(self.coordinates_train, dtype=torch.float32)
            logging.info(f"Coordinates shape: {self.coordinates_train.shape}")
            logging.info("Normalizing coordinates")
            self.coordinates_train = (self.coordinates_train - self.coord_mean) / self.coord_std
            n_nans = torch.isnan(self.coordinates_train).sum()
            assert not torch.isnan(self.coordinates_train).any(), f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
            logging.info("Coordinates normalization successful")

    def load_coord_test_data(self):
        if self.coordinates_test is None:
            logging.info("Loading coordinates for test data")
            coordinates_names = ["xccf", "yccf", "zccf"]
            self.coordinates_test = pd.read_parquet(self.config.maldi_file,
                                                    columns=coordinates_names,
                                                    filters=self.test_filter).values
            self.coordinates_test = torch.tensor(self.coordinates_test, dtype=torch.float32)
            logging.info(f"Coordinates shape: {self.coordinates_test.shape}")
            logging.info("Normalizing coordinates")
            self.coordinates_test = (self.coordinates_test - self.coord_mean) / self.coord_std
            n_nans = torch.isnan(self.coordinates_test).sum()
            assert not torch.isnan(self.coordinates_test).any(), f"Coordinates contain NaNs: {n_nans} NaNs found in coordinates"
            logging.info("Coordinates normalization successful")

    def load_checkpoint(self):
        if len(list(self.config.checkpoint_path.glob("*.pth"))) > 0:
            last_checkpoint = max(self.config.checkpoint_path.glob("*.pth"), key=lambda x: x.stat().st_mtime)
            self.multimodal_model.load_state_dict(torch.load(last_checkpoint, map_location=torch.device(self.config.device)))
            self.current_epoch = len(list(self.config.checkpoint_path.glob("*.pth")))
            logging.info(f"Loaded checkpoint {last_checkpoint}")

    @property
    def maldi_true_values_train(self):
        """Returns the true MALDI values for the training set as a numpy array."""
        if self.maldi_train_data_original is None:
            self.maldi_train_data_original = np.load(self.config.exp_path / "train" / "maldi_true_values.npy")
        return self.maldi_train_data_original

    @property
    def gene_true_values_train(self):
        """Returns the true gene expression values for the training set as a numpy array."""
        if self.gene_train_data_original is None:
            self.gene_train_data_original = np.load(self.config.exp_path / "train" / "gene_true_values.npy")
        return self.gene_train_data_original

    @property
    def maldi_predictions_train(self):
        """Returns the MALDI predictions for the training set as a numpy array."""
        train_path = self.config.exp_path / "train"
        predictions_file = train_path / "maldi_predictions.npy"
        if not predictions_file.exists():
            logging.error("MALDI predictions for the training set do not exist. Please run the experiment first.")
            return None
        return np.load(predictions_file)

    @property
    def gene_predictions_train(self):
        """Returns the gene expression predictions for the training set as a numpy array."""
        train_path = self.config.exp_path / "train"
        predictions_file = train_path / "gene_predictions.npy"
        if not predictions_file.exists():
            logging.error("Gene expression predictions for the training set do not exist. Please run the experiment first.")
            return None
        return np.load(predictions_file)

    def train_fit(self):
        """Train the multimodal model"""
        optimizer = torch.optim.AdamW(self.multimodal_model.parameters(), lr=self.config.learning_rate, weight_decay=1e-3)
        logging.info("Ready to roll")

        dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

        logging.info("Model file not found, starting training from scratch")
        if self.current_epoch < self.config.epochs:
            logging.info(f"Starting training from epoch {self.current_epoch}")
            self.multimodal_model.train_model(
                self.config.exp_path,
                dataloader_train,
                optimizer,
                epochs=self.config.epochs,
                current_epoch=self.current_epoch,
                print_every=1000,
                alpha=self.config.alpha
            )

    def predict_original_scale(self, train_predictions_maldi, train_predictions_gene, test_predictions_maldi, test_predictions_gene):
        """Convert predictions back to original scale"""
        if self.maldi_train_data is None:
            self.load_maldi_train_data()
        if self.gene_train_data is None:
            self.load_gene_train_data()
        if self.maldi_test_data is None:
            self.load_maldi_test_data()
        if self.gene_test_data is None:
            self.load_gene_test_data()

        # Convert MALDI predictions to original scale
        logging.info("Converting MALDI predictions to original scale")
        train_predictions_maldi = train_predictions_maldi * self.maldi_col_stds + self.maldi_col_means
        test_predictions_maldi = test_predictions_maldi * self.maldi_col_stds + self.maldi_col_means
        
        if self.config.log_transform:
            train_predictions_maldi = torch.exp(train_predictions_maldi) - 1e-10
            test_predictions_maldi = torch.exp(test_predictions_maldi) - 1e-10

        # Convert gene expression predictions to original scale
        logging.info("Converting gene expression predictions to original scale")
        train_predictions_gene = train_predictions_gene * self.gene_col_stds + self.gene_col_means
        test_predictions_gene = test_predictions_gene * self.gene_col_stds + self.gene_col_means
        
        if self.config.log_transform:
            train_predictions_gene = torch.exp(train_predictions_gene) - 1e-10
            test_predictions_gene = torch.exp(test_predictions_gene) - 1e-10

        # Convert true values to original scale
        maldi_train_data = self.maldi_train_data * self.maldi_col_stds + self.maldi_col_means
        maldi_test_data = self.maldi_test_data * self.maldi_col_stds + self.maldi_col_means
        gene_train_data = self.gene_train_data * self.gene_col_stds + self.gene_col_means
        gene_test_data = self.gene_test_data * self.gene_col_stds + self.gene_col_means
        
        if self.config.log_transform:
            maldi_train_data = torch.exp(maldi_train_data) - 1e-10
            maldi_test_data = torch.exp(maldi_test_data) - 1e-10
            gene_train_data = torch.exp(gene_train_data) - 1e-10
            gene_test_data = torch.exp(gene_test_data) - 1e-10

        # Save data and predictions in the original scale
        self.maldi_train_data_original = maldi_train_data.numpy()
        self.maldi_test_data_original = maldi_test_data.numpy()
        self.gene_train_data_original = gene_train_data.numpy()
        self.gene_test_data_original = gene_test_data.numpy()
        
        # Save predictions
        np.save(self.config.exp_path / "train" / "maldi_predictions.npy", train_predictions_maldi.numpy())
        np.save(self.config.exp_path / "test" / "maldi_predictions.npy", test_predictions_maldi.numpy())
        np.save(self.config.exp_path / "train" / "gene_predictions.npy", train_predictions_gene.numpy())
        np.save(self.config.exp_path / "test" / "gene_predictions.npy", test_predictions_gene.numpy())
        
        # Save true values
        np.save(self.config.exp_path / "train" / "maldi_true_values.npy", maldi_train_data.numpy())
        np.save(self.config.exp_path / "test" / "maldi_true_values.npy", maldi_test_data.numpy())
        np.save(self.config.exp_path / "train" / "gene_true_values.npy", gene_train_data.numpy())
        np.save(self.config.exp_path / "test" / "gene_true_values.npy", gene_test_data.numpy())

    def plot_multimodal_scatter(self, lipid, gene, dataset="train"):
        """Generate scatter plot comparing predictions vs true values for a lipid and gene"""
        if dataset == "train":
            maldi_true = self.maldi_true_values_train
            maldi_pred = self.maldi_predictions_train
            gene_true = self.gene_true_values_train
            gene_pred = self.gene_predictions_train
        else:
            # Implement test data loading if needed
            pass

        if lipid not in self.config.selected_lipids_names:
            print(f"Lipid {lipid} not found in the selected lipids.")
            return
        if gene not in self.config.selected_genes_names:
            print(f"Gene {gene} not found in the selected genes.")
            return

        lipid_index = self.config.selected_lipids_names.index(lipid)
        gene_index = self.config.selected_genes_names.index(gene)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # MALDI scatter plot
        maldi_true_values = maldi_true[:, lipid_index]
        maldi_pred_values = maldi_pred[:, lipid_index]
        maldi_correlation = np.corrcoef(maldi_true_values, maldi_pred_values)[0, 1]
        
        axes[0].scatter(maldi_true_values, maldi_pred_values, alpha=0.5, s=1)
        axes[0].plot([maldi_true_values.min(), maldi_true_values.max()], 
                     [maldi_true_values.min(), maldi_true_values.max()], 'k--', lw=2)
        axes[0].set_xlabel(f"{lipid} (true)")
        axes[0].set_ylabel(f"{lipid} (predicted)")
        axes[0].set_title(f"MALDI: {lipid}\nCorrelation: {maldi_correlation:.3f}")

        # Gene expression scatter plot
        gene_true_values = gene_true[:, gene_index]
        gene_pred_values = gene_pred[:, gene_index]
        gene_correlation = np.corrcoef(gene_true_values, gene_pred_values)[0, 1]
        
        axes[1].scatter(gene_true_values, gene_pred_values, alpha=0.5, s=1)
        axes[1].plot([gene_true_values.min(), gene_true_values.max()], 
                     [gene_true_values.min(), gene_true_values.max()], 'k--', lw=2)
        axes[1].set_xlabel(f"{gene} (true)")
        axes[1].set_ylabel(f"{gene} (predicted)")
        axes[1].set_title(f"Gene Expression: {gene}\nCorrelation: {gene_correlation:.3f}")

        plt.tight_layout()
        plt.show()

    def run(self):
        """Run the multimodal experiment."""
        logging.info("Starting multimodal training loop")
        
        if (self.config.exp_path / "model.pth").exists():
            logging.info("Loading model from file")
            self.multimodal_model.load_state_dict(torch.load(self.config.exp_path / "model.pth", map_location=self.config.device))
            logging.info("Model loaded successfully")
        else:
            self.load_train_data()
            self.current_epoch = 0
            self.load_checkpoint()
            wandb.init(name=self.config.exp_name,
                       project="l3di_multimodal",
                       config=self.config.to_dict())
            self.train_fit()
            wandb.finish()
            logging.info("Training completed, saving model")

        # Predict on training set
        logging.info("Predicting on the training set")
        train_path = self.config.exp_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        train_predictions_file = train_path / "predictions.pth"
        
        if not train_predictions_file.exists():
            self.load_coord_train_data()
            logging.info("Predicting on the training set")
            self.multimodal_model.eval()
            train_pred_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.coordinates_train),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            logging.info("Predicting on the training set using the model")
            maldi_predictions_list = []
            gene_predictions_list = []
            
            for batch in tqdm(train_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                maldi_pred, gene_pred, gp_posterior_1, gp_posterior_2, attention_weights = self.multimodal_model.predict(coordinates_batch)
                maldi_predictions_list.append(maldi_pred.detach().cpu())
                gene_predictions_list.append(gene_pred.detach().cpu())
            
            train_maldi_predictions = torch.cat(maldi_predictions_list, dim=0)
            train_gene_predictions = torch.cat(gene_predictions_list, dim=0)
            
            torch.save({
                'maldi': train_maldi_predictions,
                'gene': train_gene_predictions
            }, train_predictions_file)
        else:
            logging.info("Train predictions already exist, loading from file")
            train_predictions = torch.load(train_predictions_file)
            train_maldi_predictions = train_predictions['maldi']
            train_gene_predictions = train_predictions['gene']

        # Predict on test set
        logging.info("Predicting on the test set")
        test_path = self.config.exp_path / "test"
        test_path.mkdir(parents=True, exist_ok=True)
        test_predictions_file = test_path / "predictions.pth"
        
        if not test_predictions_file.exists():
            self.load_coord_test_data()
            logging.info("Predicting on the test set")
            self.multimodal_model.eval()
            test_pred_dataloader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(self.coordinates_test),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            maldi_predictions_list = []
            gene_predictions_list = []
            
            for batch in tqdm(test_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                maldi_pred, gene_pred, gp_posterior_1, gp_posterior_2, attention_weights = self.multimodal_model.predict(coordinates_batch)
                maldi_predictions_list.append(maldi_pred.detach().cpu())
                gene_predictions_list.append(gene_pred.detach().cpu())
            
            test_maldi_predictions = torch.cat(maldi_predictions_list, dim=0)
            test_gene_predictions = torch.cat(gene_predictions_list, dim=0)
            
            torch.save({
                'maldi': test_maldi_predictions,
                'gene': test_gene_predictions
            }, test_predictions_file)
        else:
            logging.info("Test predictions already exist, loading from file")
            test_predictions = torch.load(test_predictions_file)
            test_maldi_predictions = test_predictions['maldi']
            test_gene_predictions = test_predictions['gene']

        # Convert predictions to original scale
        train_maldi_predictions_file = train_path / "maldi_predictions.npy"
        test_maldi_predictions_file = test_path / "maldi_predictions.npy"
        
        if not train_maldi_predictions_file.exists() or not test_maldi_predictions_file.exists():
            logging.info("Converting predictions to original scale")
            self.predict_original_scale(
                train_maldi_predictions, train_gene_predictions,
                test_maldi_predictions, test_gene_predictions
            )
