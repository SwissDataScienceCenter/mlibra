import gpytorch
import numpy as np
import torch
import torch.nn.functional as F
from gpytorch.distributions import (MultitaskMultivariateNormal,
                                    MultivariateNormal)
from gpytorch.kernels import (MaternKernel, MultitaskKernel, RBFKernel,
                              ScaleKernel, ProductKernel, PeriodicKernel)
from gpytorch.means import ConstantMean, MultitaskMean, LinearMean, ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  IndependentMultitaskVariationalStrategy,
                                  MultitaskVariationalStrategy,
                                  NaturalVariationalDistribution,
                                  VariationalStrategy)
from gpytorch.likelihoods.multitask_gaussian_likelihood import MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan
from torch import nn
from tqdm import tqdm
import wandb
import math


class CrossAttention(nn.Module):
    """
    Cross-attention module that allows two modalities to attend to each other.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """
        Args:
            query: (batch_size, seq_len_q, dim)
            key: (batch_size, seq_len_k, dim) 
            value: (batch_size, seq_len_v, dim)
        """
        batch_size = query.size(0)
        
        # Project to Q, K, V
        Q = self.query_proj(query)  # (batch_size, seq_len_q, dim)
        K = self.key_proj(key)      # (batch_size, seq_len_k, dim)
        V = self.value_proj(value)  # (batch_size, seq_len_v, dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.dim)  # (batch_size, seq_len_q, dim)
        
        # Final projection
        output = self.out_proj(attended_values)
        
        return output, attention_weights


class MultiModalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between two modalities.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiModalCrossAttention, self).__init__()
        self.cross_attn_1_to_2 = CrossAttention(dim, num_heads, dropout)
        self.cross_attn_2_to_1 = CrossAttention(dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1_mod1 = nn.LayerNorm(dim)
        self.norm1_mod2 = nn.LayerNorm(dim)
        self.norm2_mod1 = nn.LayerNorm(dim)
        self.norm2_mod2 = nn.LayerNorm(dim)
        
        # Feed-forward networks
        self.ffn_mod1 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_mod2 = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, latent_mod1, latent_mod2):
        """
        Args:
            latent_mod1: (batch_size, latent_dim) - latent representation from modality 1
            latent_mod2: (batch_size, latent_dim) - latent representation from modality 2
        """
        # Add sequence dimension for attention
        latent_mod1 = latent_mod1.unsqueeze(1)  # (batch_size, 1, latent_dim)
        latent_mod2 = latent_mod2.unsqueeze(1)  # (batch_size, 1, latent_dim)
        
        # Cross-attention: modality 1 attends to modality 2
        attended_mod1, attn_weights_1_to_2 = self.cross_attn_1_to_2(latent_mod1, latent_mod2, latent_mod2)
        attended_mod1 = self.norm1_mod1(latent_mod1 + attended_mod1)
        
        # Cross-attention: modality 2 attends to modality 1  
        attended_mod2, attn_weights_2_to_1 = self.cross_attn_2_to_1(latent_mod2, latent_mod1, latent_mod1)
        attended_mod2 = self.norm1_mod2(latent_mod2 + attended_mod2)
        
        # Feed-forward networks
        ffn_out_mod1 = self.ffn_mod1(attended_mod1)
        attended_mod1 = self.norm2_mod1(attended_mod1 + ffn_out_mod1)
        
        ffn_out_mod2 = self.ffn_mod2(attended_mod2)
        attended_mod2 = self.norm2_mod2(attended_mod2 + ffn_out_mod2)
        
        # Remove sequence dimension
        attended_mod1 = attended_mod1.squeeze(1)  # (batch_size, latent_dim)
        attended_mod2 = attended_mod2.squeeze(1)  # (batch_size, latent_dim)
        
        return attended_mod1, attended_mod2, attn_weights_1_to_2, attn_weights_2_to_1


class LGPMultiModal(nn.Module):
    """
    Multimodal Latent Gaussian Process model with cross-attention between modalities.
    
    This model uses two separate GP models for two different modalities, applies
    cross-attention between their latent representations, and has separate decoder
    heads for each modality.
    """
    def __init__(self, p1, p2, d, n_neurons, dropout, activation, device, gp_model_1, gp_model_2, num_heads=8):
        """
        Args:
            p1: number of channels for modality 1
            p2: number of channels for modality 2
            d: latent dimension (should be same for both modalities)
            n_neurons: list of hidden layer sizes for decoders
            dropout: list of dropout rates for decoders
            activation: activation function ('relu' or 'tanh')
            device: computation device
            gp_model_1: GP model for modality 1
            gp_model_2: GP model for modality 2
            num_heads: number of attention heads for cross-attention
        """
        super(LGPMultiModal, self).__init__()
        self.mode = "lgp_multimodal"
        self.p1 = p1  # number of channels for modality 1
        self.p2 = p2  # number of channels for modality 2
        self.d = d    # latent dimension
        
        # Noise parameters for each modality
        self.log_var_n_1 = nn.Parameter(torch.zeros(p1))
        self.log_var_n_2 = nn.Parameter(torch.zeros(p2))
        
        # GP models for each modality
        self.gp_model_1 = gp_model_1
        self.gp_model_2 = gp_model_2
        
        # Cross-attention module
        self.cross_attention = MultiModalCrossAttention(d, num_heads, dropout[0] if len(dropout) > 0 else 0.1)
        
        # Separate decoder networks for each modality
        self.decoder_layers_1 = self.build_decoder(n_neurons[::-1], dropout[::-1], activation, d)
        self.output_layer_1 = nn.Linear(n_neurons[0], p1)
        
        self.decoder_layers_2 = self.build_decoder(n_neurons[::-1], dropout[::-1], activation, d)
        self.output_layer_2 = nn.Linear(n_neurons[0], p2)
        
        self.float_type = torch.float32
        self.device = device
        self.to(device)
        
    def build_decoder(self, n_neurons, dropout, activation, input_dim):
        layers = []
        for i in range(len(n_neurons)):
            layers.append(nn.Linear(input_dim, n_neurons[i]))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            if dropout[i] > 0:
                layers.append(nn.Dropout(dropout[i]))
            input_dim = n_neurons[i]
        return nn.Sequential(*layers)
    
    def decode_modality_1(self, z):
        """Decode latent representation to modality 1"""
        h = self.decoder_layers_1(z)
        x_mu = self.output_layer_1(h)
        return x_mu
    
    def decode_modality_2(self, z):
        """Decode latent representation to modality 2"""
        h = self.decoder_layers_2(z)
        x_mu = self.output_layer_2(h)
        return x_mu
    
    def forward(self, coords):
        """
        Forward pass through the multimodal model.
        
        Args:
            coords: coordinate tensor
            
        Returns:
            x_reconstructed_1: reconstructed data for modality 1
            x_reconstructed_2: reconstructed data for modality 2
            gp_posterior_1: GP posterior for modality 1
            gp_posterior_2: GP posterior for modality 2
            attention_weights: cross-attention weights
        """
        # Get latent representations from both GP models
        gp_posterior_1 = self.gp_model_1(coords)
        gp_posterior_2 = self.gp_model_2(coords)
        
        latent_1 = gp_posterior_1.mean
        latent_2 = gp_posterior_2.mean
        
        # Apply cross-attention between modalities
        attended_latent_1, attended_latent_2, attn_1_to_2, attn_2_to_1 = self.cross_attention(latent_1, latent_2)
        
        # Decode each modality using attended latent representations
        x_reconstructed_1 = self.decode_modality_1(attended_latent_1)
        x_reconstructed_2 = self.decode_modality_2(attended_latent_2)
        
        attention_weights = {
            'mod1_to_mod2': attn_1_to_2,
            'mod2_to_mod1': attn_2_to_1
        }
        
        return x_reconstructed_1, x_reconstructed_2, gp_posterior_1, gp_posterior_2, attention_weights
    
    def encode(self, coords):
        """
        Encode the coordinates using both GP models.
        """
        gp_posterior_1 = self.gp_model_1(coords)
        gp_posterior_2 = self.gp_model_2(coords)
        return gp_posterior_1.mean, gp_posterior_1.variance, gp_posterior_2.mean, gp_posterior_2.variance
    
    def predict(self, coords):
        """
        Prediction with both modalities.
        """
        with torch.no_grad():
            x_reconstructed_1, x_reconstructed_2, gp_posterior_1, gp_posterior_2, attention_weights = self.forward(coords)
            return x_reconstructed_1, x_reconstructed_2, gp_posterior_1, gp_posterior_2, attention_weights
    
    def loss_function(self, x1, x2, x_reconstructed_1, x_reconstructed_2, beta=1.0, alpha=1.0):
        """
        Compute the loss function for both modalities.
        
        Args:
            x1: true data for modality 1
            x2: true data for modality 2
            x_reconstructed_1: reconstructed data for modality 1
            x_reconstructed_2: reconstructed data for modality 2
            beta: weight for KL divergence
            alpha: weight for balancing between modality losses
        """
        # Reconstruction losses for both modalities
        recon_loss_1 = self.nll_loss(x1, x_reconstructed_1, self.log_var_n_1)
        recon_loss_2 = self.nll_loss(x2, x_reconstructed_2, self.log_var_n_2)
        
        # Combined reconstruction loss
        recon_loss = alpha * recon_loss_1 + (1 - alpha) * recon_loss_2
        
        # KL divergences from both GP models
        kl_gp_1 = self.gp_model_1.variational_strategy.kl_divergence().sum()
        kl_gp_2 = self.gp_model_2.variational_strategy.kl_divergence().sum()
        kl_loss = kl_gp_1 + kl_gp_2
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss, recon_loss_1, recon_loss_2
    
    def nll_loss(self, x, x_reconstructed, log_var_x):
        """Negative log-likelihood loss"""
        nll_loss = 0.5 * torch.sum((x - x_reconstructed).pow(2) / torch.exp(log_var_x) + log_var_x)
        return nll_loss
    
    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000, alpha=0.5):
        """
        Train the multimodal model.
        
        Args:
            exp_path: experiment path for saving checkpoints
            dataloader: training data loader (should return x1, x2, coords)
            optimizer: optimizer
            epochs: number of epochs
            current_epoch: starting epoch
            print_every: frequency of printing progress
            alpha: weight for balancing modality losses
        """
        self.to(self.device)
        self.train()
        
        for epoch in range(current_epoch, epochs):
            mean_loss = 0
            reconstr_loss = 0
            reconstr_loss_1 = 0
            reconstr_loss_2 = 0
            kl_loss = 0
            mse_loss_1 = 0
            mse_loss_2 = 0
            
            for i, data in enumerate(tqdm(dataloader)):
                x1, x2, coord = data
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                coord = coord.to(self.device)
                
                optimizer.zero_grad()
                x_reconstructed_1, x_reconstructed_2, gp_posterior_1, gp_posterior_2, attention_weights = self(coord)
                
                loss, recon_loss, kl_div, recon_loss_1, recon_loss_2 = self.loss_function(
                    x1, x2, x_reconstructed_1, x_reconstructed_2, beta=1.0, alpha=alpha)
                
                loss.backward()
                optimizer.step()
                
                mean_loss += loss.item()
                reconstr_loss += recon_loss.item()
                reconstr_loss_1 += recon_loss_1.item()
                reconstr_loss_2 += recon_loss_2.item()
                kl_loss += kl_div.item()
                
                mse_loss_batch_1 = F.mse_loss(x_reconstructed_1.detach(), x1.detach()).item()
                mse_loss_batch_2 = F.mse_loss(x_reconstructed_2.detach(), x2.detach()).item()
                mse_loss_1 += mse_loss_batch_1
                mse_loss_2 += mse_loss_batch_2
                
                if i % 10 == 0:
                    wandb.log({
                        "loss_batch": loss.item(),
                        "mse_loss_batch_mod1": mse_loss_batch_1,
                        "mse_loss_batch_mod2": mse_loss_batch_2
                    })
            
            # Save checkpoint
            torch.save(self.state_dict(), exp_path / f"checkpoints/model_{epoch}.pth")
            
            # Log epoch metrics
            num_batches = len(dataloader)
            wandb.log({
                "loss": mean_loss / num_batches,
                "reconstruction_loss": reconstr_loss / num_batches,
                "reconstruction_loss_mod1": reconstr_loss_1 / num_batches,
                "reconstruction_loss_mod2": reconstr_loss_2 / num_batches,
                "kl_loss": kl_loss / num_batches,
                "mse_loss_mod1": mse_loss_1 / num_batches,
                "mse_loss_mod2": mse_loss_2 / num_batches
            })
            
            print(f"Epoch {epoch} loss: {mean_loss / num_batches}")
            print(f"Epoch {epoch} reconstruction loss mod1: {reconstr_loss_1 / num_batches}")
            print(f"Epoch {epoch} reconstruction loss mod2: {reconstr_loss_2 / num_batches}")
            print(f"Epoch {epoch} mse loss mod1: {mse_loss_1 / num_batches}")
            print(f"Epoch {epoch} mse loss mod2: {mse_loss_2 / num_batches}")
        
        torch.save(self.state_dict(), exp_path / "model.pth")
