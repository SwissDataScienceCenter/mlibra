import torch
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maldi.config import MaldiConfig
from maldi.experiment import MaldiExperiment
from maldi.utils import get_inducing_points
from l3di.lgpmoe import LGPMOE
from l3di.lgp import IndependentMultitaskGPModel, Custom3DKernel


def create_mask(available_channels, selected_channels:np.ndarray, selected_channels_names:np.ndarray, latent_dim):
    if len(available_channels) == 0:
        raise ValueError("No available channels")
    if len(selected_channels) == len(available_channels):
        mask = torch.zeros((len(selected_channels), latent_dim))
        # we select those elements of selected_channels_names that start with PC, PE, PG, PA, PS, PI and set the mask to 1
        pc_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PC")]
        pe_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PE")]
        pg_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PG")]
        pa_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PA")]
        ps_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PS")]
        pi_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("PI")]
        glycerophospholipids= pc_list + pe_list + pg_list + pa_list + ps_list + pi_list
        # we set the mask to 1 for those elements
        mask[glycerophospholipids, 0] = 1
        # now the Lysophospholipids	LPC, LPE, LPA, LPS, LPG
        lpc_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("LPC")]
        lpe_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("LPE")]
        lpa_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("LPA")]
        lps_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("LPS")]
        lpg_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("LPG")]
        lysophospholipids = lpc_list + lpe_list + lpa_list + lps_list + lpg_list
        # we set the mask to 1 for those elements
        mask[lysophospholipids, 1] = 1
        # now the sphingolipids	SM, Cer, HexCer, Hex2Cer
        sm_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("SM")]
        cer_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("Cer")]
        hexc_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("HexCer")]
        hexc2_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("Hex2Cer")]
        glyco_sphingolipids = hexc_list + hexc2_list
        simple_sphingolipids = sm_list + cer_list
        # we set the mask to 1 for those elements
        mask[glyco_sphingolipids, 2] = 1
        mask[simple_sphingolipids, 3] = 1
        # now Neutral Storage Lipids	TG, DG
        tg_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("TG")]
        dg_list = [index for index, name in enumerate(selected_channels_names) if name.startswith("DG")]
        neutral_storage_lipids = tg_list + dg_list
        # we set the mask to 1 for those elements
        mask[neutral_storage_lipids, 4] = 1
        # the rest (if any) are set to 1
        rest_list = [index for index in range(len(selected_channels_names)) if index not in glycerophospholipids + lysophospholipids + glyco_sphingolipids + simple_sphingolipids + neutral_storage_lipids]
        if len(rest_list) > 0:
            if latent_dim > 5:
                mask[rest_list, 5:] = 1
        return mask
    else:
        return None




def parse_args():
    parser = argparse.ArgumentParser(description='LGPMOE experiment')

    parser.add_argument("--mode", type=str, required=True, help="Experiment mode (e.g., 'train', 'test').")
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", dest="maldi_file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--exp-name", dest="exp_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", dest="available_lipids_file", type=str, required=True, help="File with available lipids.")
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
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts in the mixture of experts model.")


    args=parser.parse_args()
    # we return dict of args
    return vars(args)

def main():
    args = parse_args()
    args['exp_name'] = "lgpmoe_" + args['exp_name']
    # Set random seed

    # Load config
    config = MaldiConfig.from_args(args)

    available_lipids= np.load(config.available_lipids_file, allow_pickle=True)

    expert_mask = create_mask(
        config.available_lipids,
        config.selected_channels,
        config.selected_lipids_names,
        args["num_experts"]
    )
    # Setup device
    print(f"Using device: {config.device}")
    
    # Create experiment directory
    # Load inducing points for GP
    inducing_points, coord_mean, coord_std = get_inducing_points(config.exp_path, config.dataset_path, num_inducing=config.num_inducing)

    logging.info("Creating GP model")
    voxel_size = 0.025
    n_pixel = args["n_pixels"]
    minimal_length_scale = args["n_pixels"] * voxel_size / (coord_std.sum()/3)
    logging.info(f"minimal length scale in um: {n_pixel * voxel_size}")

    # Initialize GP model
    gp_model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel,
        nu=config.nu,
        minimal_length_scale=minimal_length_scale,
        input_dim=3
    )
    
    # Initialize LGPMOE model
    lgpmoe_model = LGPMOE(
        p= len(config.selected_lipids_names),
        d= config.latent_dim,
        n_neurons=[255, 255, 255],
        dropout=0.1,
        activation='relu',
        device=config.device,
        gp_model=gp_model,
        expert_matrix=expert_mask
    )

    experiment = MaldiExperiment(config, lgpmoe_model, coord_mean, coord_std)
    experiment.run()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting LGPMOE experiment")
    main()
    logging.info("Experiment completed successfully")
