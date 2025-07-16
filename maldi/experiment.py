"""
experiment.py
Main experiment logic and classes for the MALDI experiment.
"""
import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import wandb


class MaldiExperiment:
    def __init__(self, config, lgp_model, coord_mean, coord_std):
        self.config = config
        self.coord_mean = coord_mean
        self.coord_std = coord_std

        self.train_filter = config.section_filter
        self.test_filter = config.test_filter
        self.lgp_model = lgp_model

        self.coordinates_train = None
        self.coordinates_test = None

    def load_train_data(self):
        self.train_data = pd.read_parquet(self.config.maldi_file,
                                          columns=[str(i) for i in self.config.selected_lipids_names],
                                          filters=self.train_filter).values
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        assert self.train_data.shape[0] > 0, "No training data found with the given filter"
        n_nans = torch.isnan(self.train_data).sum()
        assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs: {n_nans} NaNs found"
        n_zeros = (self.train_data == 0).sum()
        logging.info(f"Training data contains {n_zeros} zeros")
        n_negatives = (self.train_data < 0).sum()
        logging.info(f"Training data contains {n_negatives} negative values")
        logging.info("imputing negative values to zero")
        self.train_data[self.train_data < 0] = 0

        log_transform = self.config.log_transform
        if log_transform:
            logging.info("Applying log transformation to training data")
            self.train_data = np.log(self.train_data + 1e-10)
            n_nans = torch.isnan(self.train_data).sum()
            assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs after log transformation: {n_nans} NaNs found after log transformation"
        else:
            logging.info("Skipping log transformation")
        logging.info(f"Training data shape: {self.train_data.shape}")

        logging.info("normalizing training data")
        if (self.config.exp_path / "lipid_means.pth").exists() and (self.config.exp_path / "lipid_std.pth").exists():
            logging.info("Loading column mean and std from files")
            col_means = torch.load(self.config.exp_path / "lipid_means.pth")
            col_std = torch.load(self.config.exp_path / "lipid_std.pth")
        else:
            logging.info("computing channels means and stds")
            col_means = self.train_data.mean(dim=0)
            col_stds = self.train_data.std(dim=0)
            torch.save(col_means, self.config.exp_path / "lipid_means.pth")
            torch.save(col_stds, self.config.exp_path / "lipid_stds.pth")
        assert not torch.isnan(col_means).any(), "there are nans in col_means"
        assert not torch.isnan(col_stds).any(), "there are nans in col_stds"
        self.col_means = col_means
        self.col_stds = col_stds
        self.train_data = (self.train_data - col_means) / col_stds
        n_nans = torch.isnan(self.train_data).sum()
        assert not torch.isnan(self.train_data).any(), f"Training data contains NaNs after normalization: {n_nans} NaNs found after normalization"
        logging.info("Training data normalization successful")

        logging.info("Loading coordinates")
        coordinates_names =["xccf","yccf","zccf"]
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

        self.dataset_train = torch.utils.data.TensorDataset(self.train_data, self.coordinates_train)

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
            self.lgp_model.load_state_dict(torch.load(last_checkpoint, map_location=torch.device(self.config.device)))
            self.current_epoch = len(list(self.config.checkpoint_path.glob("*.pth")))
            logging.info(f"loaded checkpoint {last_checkpoint}")




    def train_fit(self):
            optimizer = torch.optim.Adam(self.lgp_model.parameters(), lr=self.config.learning_rate)
            logging.info("ready to roll")


            dataloader_train= torch.utils.data.DataLoader(
                self.dataset_train,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0
            )

            logging.info("Model file not found, starting training from scratch")
            if self.current_epoch < self.config.epochs:
                logging.info(f"Starting training from epoch {self.current_epoch}")
                self.lgp_model.train_model(self.config.exp_path,
                                      dataloader_train,
                                      optimizer,
                                      epochs=self.config.epochs,
                                      current_epoch=self.current_epoch,
                                      print_every=1000)


    def run(self):
        """Run the experiment."""
        # Fit he model using the training data,
        logging.info("Starting training loop")
        if(self.config.exp_path / "model.pth").exists():
            logging.info("Loading model from file")
            self.lgp_model.load_state_dict(torch.load(self.config.exp_path / "model.pth", map_location=self.config.device))
            logging.info("Model loaded successfully")
        else:
            self.load_train_data()
            self.current_epoch = 0
            self.load_checkpoint()
            wandb.init(name=self.config.exp_name,
                       project="l3di_maldi",
                       config=self.config.to_dict()
                       )
            self.train_fit()
            wandb.finish()
            logging.info("Training completed, saving model")

        # Predict in the train set
        logging.info("Predicting in the train set")
        train_path = self.config.exp_path / "train"
        train_path.mkdir(parents=True, exist_ok=True)
        train_predictions_file = train_path / "predictions.pth"
        if not train_predictions_file.exists():
            self.load_coord_train_data()
            logging.info("Predicting in the train set")
            self.lgp_model.eval()
            train_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_train),
                                                                batch_size=self.config.batch_size,
                                                                shuffle=False,
                                                                num_workers=0)
            logging.info("Predicting in the train set using the model")
            predictions_list = []
            posterior_list = []
            for batch in tqdm(train_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                predictions, gp_posterior = self.lgp_model.predict(coordinates_batch)
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(gp_posterior.mean.detach().cpu())
            train_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            torch.save(train_predictions, train_predictions_file)
            torch.save(posterior, train_predictions_file.with_name("posterior.pth"))
        else:
            logging.info("Train predictions already exist, loading from file")
            train_predictions = torch.load(train_predictions_file)

        test_path = self.config.exp_path / "test"
        test_path.mkdir(parents=True, exist_ok=True)
        test_predictions_file = test_path / "predictions.pth"
        if not test_predictions_file.exists():
            self.load_coord_test_data()
            logging.info("Predicting in the test set")
            self.lgp_model.eval()
            self.load_coord_test_data()
            test_pred_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.coordinates_test),
                                                               batch_size=self.config.batch_size,
                                                               shuffle=False,
                                                               num_workers=0)
            predictions_list = []
            posterior_list = []
            for batch in tqdm(test_pred_dataloader):
                coordinates_batch = batch[0].to(self.config.device)
                predictions,posterior = self.lgp_model.predict(coordinates_batch)
                predictions_list.append(predictions.detach().cpu())
                posterior_list.append(posterior.mean.detach().cpu())
            test_predictions = torch.cat(predictions_list, dim=0)
            posterior = torch.cat(posterior_list, dim=0)
            torch.save(test_predictions, test_predictions_file)
            torch.save(posterior, test_predictions_file.with_name("posterior.pth"))
        else:
            logging.info("Test predictions already exist, loading from file")
            test_predictions = torch.load(test_predictions_file)
