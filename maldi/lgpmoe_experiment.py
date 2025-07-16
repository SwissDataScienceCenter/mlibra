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


def parse_args():
    parser = argparse.ArgumentParser(description='LGPMOE experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in the mixture')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    config = MaldiConfig.from_args(args)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_path = os.path.join(config.experiment_path, 'lgpmoe')
    os.makedirs(exp_path, exist_ok=True)
    
    # Load inducing points for GP
    inducing_points = get_inducing_points(exp_path, config.dataset_path, num_inducing=config.num_inducing)
    inducing_points = torch.tensor(inducing_points, dtype=torch.float32).to(device)
    
    # Initialize GP model
    gp_model = IndependentMultitaskGPModel(
        inducing_points=inducing_points,
        num_tasks=config.latent_dim,
        kernel_type=config.kernel_type,
        nu=config.nu,
        minimal_length_scale=config.minimal_length_scale,
    ).to(device)
    
    # Initialize LGPMOE model
    lgpmoe_model = LGPMOE(
        p=config.input_dim,
        d=config.latent_dim,
        n_neurons=config.decoder_neurons,
        dropout=config.dropout,
        activation=config.activation,
        device=device,
        gp_model=gp_model,
        num_experts=args.num_experts
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam([
        {'params': lgpmoe_model.parameters(), 'lr': args.lr},
    ])
    
    # Setup experiment
    experiment = MaldiExperiment(
        config=config,
        lgp_model=lgpmoe_model,
        coord_mean=0,
        coord_std=1,
    )
    
    # Load checkpoint if specified
    current_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        lgpmoe_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {current_epoch}")
    
    # Load training data
    train_data, train_coords = experiment.load_train_data()
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_coords, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train model
    print("Starting training...")
    lgpmoe_model.train_model(
        exp_path=exp_path,
        dataloader=train_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        current_epoch=current_epoch,
        print_every=10
    )
    
    # Load test data for evaluation
    test_data, test_coords = experiment.load_coord_test_data()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_coords = torch.tensor(test_coords, dtype=torch.float32).to(device)
    
    # Evaluate model
    print("Evaluating model...")
    lgpmoe_model.eval()
    with torch.no_grad():
        reconstructed, latent, expert_weights = lgpmoe_model.predict(test_coords)
        
        # Calculate mean squared error
        mse = torch.mean((reconstructed - test_data) ** 2).item()
        print(f"Test MSE: {mse:.4f}")
        
        # Save expert weights for analysis
        np.save(os.path.join(exp_path, 'expert_weights.npy'), expert_weights.cpu().numpy())
        
        # Visualize expert weights for first few test samples
        plt.figure(figsize=(12, 8))
        for i in range(min(5, expert_weights.shape[0])):
            plt.subplot(5, 1, i+1)
            plt.bar(range(args.num_experts), expert_weights[i].cpu().numpy())
            plt.title(f"Sample {i+1} Expert Weights")
            plt.xlabel("Expert")
            plt.ylabel("Weight")
        plt.tight_layout()
        plt.savefig(os.path.join(exp_path, 'expert_weights.png'))
    
    print("Experiment completed successfully.")


if __name__ == '__main__':
    main()
