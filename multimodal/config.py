"""
Configuration module for multimodal MALDI + gene expression experiments.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

@dataclass
class MultiModalConfig:
    mode: str
    dataset_path: str
    maldi_file: str
    gene_file: str
    exp_name: str
    available_lipids_file: str
    available_genes_file: str
    output_dir: str
    slices_dataset_file: str
    num_inducing: int
    seed: int
    epochs: int
    latent_dim: int
    device: str
    kernel: str
    log_transform: bool
    nu: float
    n_pixels: int
    learning_rate: float
    batch_size: int
    alpha: float  # Weight for balancing modality losses
    num_heads: int  # Number of attention heads
    load_args: bool
    
    # Derived attributes
    exp_path: Path = None
    checkpoint_path: Path = None
    section_filter: List = None
    test_filter: List = None
    selected_lipids_names: List = None
    selected_genes_names: List = None

    @staticmethod
    def from_args(args):
        """Create config from parsed arguments"""
        config = MultiModalConfig(**args)
        
        # Set up paths
        config.exp_path = Path(config.output_dir) / config.exp_name
        config.exp_path.mkdir(parents=True, exist_ok=True)
        config.checkpoint_path = config.exp_path / "checkpoints"
        config.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Read available lipids and genes
        available_lipids = pd.read_csv(config.available_lipids_file)
        available_genes = pd.read_csv(config.available_genes_file)
        
        # For now, select all available lipids and genes
        # You can modify this to select specific subsets
        config.selected_lipids_names = available_lipids.iloc[:, 0].tolist()
        config.selected_genes_names = available_genes.iloc[:, 0].tolist()
        
        # Set up filters for train/test split
        # This is a simple example - you may want to customize this
        config.section_filter = [("Section", "in", [1, 2, 3, 4, 5])]  # Training sections
        config.test_filter = [("Section", "in", [6, 7])]  # Test sections
        
        return config
    
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            'mode': self.mode,
            'exp_name': self.exp_name,
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'num_heads': self.num_heads,
            'kernel': self.kernel,
            'log_transform': self.log_transform,
            'nu': self.nu,
            'n_pixels': self.n_pixels,
            'num_lipids': len(self.selected_lipids_names) if self.selected_lipids_names else 0,
            'num_genes': len(self.selected_genes_names) if self.selected_genes_names else 0
        }

def extract_filters(left_out_slice):
    """Extract filters for data splitting"""
    if left_out_slice == "all":
        section_filter = None
        test_filter = None
    else:
        # Parse left_out_slice to determine train/test split
        # This is a placeholder implementation
        section_filter = [("Section", "!=", left_out_slice)]
        test_filter = [("Section", "==", left_out_slice)]
    
    return section_filter, test_filter

def read_channels(selected_channels, available_channels):
    """Read and validate selected channels"""
    if selected_channels == "all":
        return available_channels.tolist()
    else:
        # Parse selected_channels and validate against available_channels
        # This is a placeholder implementation
        return selected_channels
