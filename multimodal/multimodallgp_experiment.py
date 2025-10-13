"""This script sets up and runs a multimodal MALDI + gene expression experiment using the l3di library."""
from l3di.lgp import IndependentMultitaskGPModel
from l3di.lgpmultimodal import LGPMultiModal
from experiment import MultiModalExperiment
from config import MultiModalConfig
from argparse import ArgumentParser
from utils import get_inducing_points
import logging

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run multimodal MALDI + gene expression experiment with l3di.")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--gene-file", dest="gene_file", type=str, required=True, help="Path to the gene expression file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--available-genes-file", dest="available_genes_file", type=str, required=True, help="File with available genes.")
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices-dataset-file", dest="slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--num-inducing", dest="num_inducing", type=int, default=100, help="Number of inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent-dim", dest="latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on (e.g., 'cpu', 'cuda').")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model.")
    parser.add_argument("--log-transform", dest="log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.5, help="Parameter for the GP model.")
    parser.add_argument("--n-pixels", dest="n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2000, help="Batch size for training")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for balancing modality losses (0.5 = equal weight)")
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=8, help="Number of attention heads for cross-attention")
    parser.add_argument("--load-args", dest="load_args", action='store_true', help="Load arguments from a file instead of command line.")
    
    # Add other arguments as needed
    return vars(parser.parse_args())

def setup_experiment(args):
    config = MultiModalConfig.from_args(args)
    logging.info("Configuration created successfully")
    logging.info("Getting inducing points")
    inducing_points, coord_mean, coord_std = get_inducing_points(
        config.exp_path, config.dataset_path, config.num_inducing
    )
    logging.info("Inducing points obtained successfully")

    logging.info("Creating GP models for both modalities")
    voxel_size = 0.025
    n_pixel = args["n_pixels"]
    minimal_length_scale = args["n_pixels"] * voxel_size / (coord_std.sum()/3)
    logging.info(f"minimal length scale in um: {n_pixel * voxel_size}")
    
    # GP model for MALDI data (modality 1)
    gp_model_maldi = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        minimal_length_scale=minimal_length_scale,
        input_dim=3
    )
    
    # GP model for gene expression data (modality 2)
    gp_model_gene = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        minimal_length_scale=minimal_length_scale,
        input_dim=3
    )
    
    logging.info("GP models created successfully")

    logging.info("Creating LGPMultiModal instance")
    multimodal_model = LGPMultiModal(
        p1=len(config.selected_lipids_names),  # number of lipids
        p2=len(config.selected_genes_names),   # number of genes
        d=config.latent_dim,
        n_neurons=[100, 100],
        dropout=[0.1, 0.1],
        activation='relu',
        device=config.device,
        gp_model_1=gp_model_maldi,
        gp_model_2=gp_model_gene,
        num_heads=config.num_heads
    )
    return MultiModalExperiment(config, multimodal_model, coord_mean, coord_std)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting multimodal MALDI + gene expression experiment")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")
    experiment = setup_experiment(args)
    experiment.run()
