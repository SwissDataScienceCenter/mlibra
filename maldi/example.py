"""
example.py
Example script to run the MALDI experiment using l3di.
"""

from l3di.lgp import LGP, IndependentMultitaskGPModel
from experiment import MaldiExperiment
from config import MaldiConfig
from argparse import ArgumentParser
from utils import get_inducing_points

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run MALDI experiment with l3di.")
    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available_lipids_file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices_dataset_file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--num_inducing", type=int, default=100, help="Number of inducing points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--latent_dim", type=int, default=10, help="Dimensionality of the latent space.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on (e.g., 'cpu', 'cuda').")
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for the GP model.")
    parser.add_argument("--log_transform", action='store_true', help="Apply log transformation to the data.")
    parser.add_argument("--nu", type=float, default=1.5, help="Parameter for the GP model.")
    parser.add_argument("--n_pixels", type=int, default=10, help="Number of pixels to consider in the experiment.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")

    # Add other arguments as needed
    return vars(parser.parse_args())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting MALDI experiment")
    args = parse_args()
    logging.info(f"Parsed arguments: {args}")
    # Create configuration from parsed arguments
    config = MaldiConfig.from_args(args)
    logging.info("Configuration created successfully")
    logging.info("Getting inducing points")
    inducing_points, coord_mean, coord_std = get_inducing_points(
        config.exp_path, config.dataset_path, config.num_inducing
    )
    logging.info("Inducing points obtained successfully")

    loggin.info("Creating GP model")
    gp_model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        n_pixels=config.n_pixels,
        input_dim=3
    )
    logging.info("GP model created successfully")

    logging.info("Creating LGP instance")
    lgp_model = LGP(
        gp_model=gp_model,
        p= len(config.selected_lipids_names),
        d= config.latent_dim,
        n_inducing=[100, 100],
        dropout=[0.1, 0.1],
        activation='relu',
        device=config.device
    )
    experiment = MaldiExperiment(config,lgp_model,coord_mean, coord_std)
    experiment.run()
