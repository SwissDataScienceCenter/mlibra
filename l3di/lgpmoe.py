import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.models import ApproximateGP
import numpy as np
import os
import time
from tqdm import tqdm

class LGPMOE(nn.Module):
    """
    Latent Gaussian Process model with a soft mixture of experts as the decoder.
    Similar to LGP but uses a mixture of experts architecture for the decoder network.
    """
    def __init__(self, p, d, n_neurons, dropout, activation, device, gp_model, num_experts=4):
        """
        Initialize the LGPMOE model.
        
        Args:
            p: dimensionality of the input data
            d: dimensionality of the latent space
            n_neurons: list of neurons per layer for the decoder
            dropout: dropout rate
            activation: activation function
            device: device to use (cpu or cuda)
            gp_model: the GP model to use
            num_experts: number of expert networks in the mixture
        """
        super(LGPMOE, self).__init__()
        
        self.p = p  # dimensionality of the input data
        self.d = d  # dimensionality of the latent space
        self.device = device
        self.gp_model = gp_model
        self.num_experts = num_experts
        
        # Build decoder components for mixture of experts
        self.gate_network = self.build_gate_network(d, num_experts, dropout, activation)
        self.expert_networks = nn.ModuleList([
            self.build_decoder(n_neurons, dropout, activation, d) for _ in range(num_experts)
        ])
        
        # Learnable log variance for the output
        self.log_var = nn.Parameter(torch.zeros(1, p))

    def build_gate_network(self, input_dim, num_experts, dropout, activation):
        """
        Build the gating network that determines the weighting of each expert.
        
        Args:
            input_dim: dimensionality of the input
            num_experts: number of experts
            dropout: dropout rate
            activation: activation function
            
        Returns:
            gate_network: nn.Sequential
        """
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2)
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
            
        gate_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(64, num_experts),
            # No softmax here as we'll apply it later
        )
        
        return gate_network
        
    def build_decoder(self, n_neurons, dropout, activation, input_dim):
        """
        Build the decoder network.
        
        Args:
            n_neurons: list of neurons per layer
            dropout: dropout rate
            activation: activation function
            input_dim: dimensionality of the input
            
        Returns:
            decoder: nn.Sequential
        """
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2)
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, n_neurons[0]))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(n_neurons) - 1):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(n_neurons[-1], self.p))
        
        return nn.Sequential(*layers)
    
    def decode(self, z):
        """
        Decode the latent representation using the mixture of experts.
        
        Args:
            z: latent representation
            
        Returns:
            x_reconstructed: reconstructed input
        """
        # Get expert outputs
        expert_outputs = [expert(z) for expert in self.expert_networks]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, num_experts, p]
        
        # Get gating weights
        gate_logits = self.gate_network(z)  # Shape: [batch_size, num_experts]
        gate_weights = F.softmax(gate_logits, dim=1).unsqueeze(2)  # Shape: [batch_size, num_experts, 1]
        
        # Combine expert outputs with gating weights
        x_reconstructed = torch.sum(expert_outputs * gate_weights, dim=1)  # Shape: [batch_size, p]
        
        return x_reconstructed, gate_weights.squeeze(2)
    
    def forward(self, coords):
        """
        Forward pass through the model.
        
        Args:
            coords: coordinates
            
        Returns:
            x_reconstructed: reconstructed input
            z_mean: mean of the latent representation
        """
        # Get latent space from GP
        z_dist = self.gp_model(coords)
        z_mean = z_dist.mean
        
        # Decode
        x_reconstructed, gate_weights = self.decode(z_mean)
        
        return x_reconstructed, z_mean, gate_weights
    
    def predict(self, coords):
        """
        Make predictions for the given coordinates.
        
        Args:
            coords: coordinates
            
        Returns:
            x_reconstructed: reconstructed input
            z_mean: mean of the latent representation
            gate_weights: weights assigned to each expert
        """
        with torch.no_grad():
            self.gp_model.eval()
            
            # Get latent space from GP
            z_dist = self.gp_model(coords)
            z_mean = z_dist.mean
            
            # Decode
            x_reconstructed, gate_weights = self.decode(z_mean)
            
        return x_reconstructed, z_mean, gate_weights
    
    def loss_function(self, x, x_reconstructed, beta=1.0):
        """
        Compute the loss function.
        
        Args:
            x: input data
            x_reconstructed: reconstructed input
            beta: weighting for the KL divergence term
            
        Returns:
            loss: total loss
            nll: negative log likelihood
        """
        # Reconstruction loss (NLL)
        nll = self.nll_loss(x, x_reconstructed, self.log_var)
        
        # Total loss is just NLL for LGP (no KL divergence term)
        loss = nll
        
        return loss, nll
    
    def nll_loss(self, x, x_reconstructed, log_var_x):
        """
        Compute the negative log likelihood.
        
        Args:
            x: input data
            x_reconstructed: reconstructed input
            log_var_x: log variance of the output
            
        Returns:
            nll: negative log likelihood
        """
        var_x = torch.exp(log_var_x)
        nll = 0.5 * torch.sum(torch.pow(x - x_reconstructed, 2) / var_x + log_var_x + np.log(2 * np.pi), dim=1)
        return torch.mean(nll)
    
    def train_model(self, exp_path, dataloader, optimizer, epochs, current_epoch, print_every=1000):
        """
        Train the model.
        
        Args:
            exp_path: path to save the model
            dataloader: dataloader for training data
            optimizer: optimizer
            epochs: number of epochs
            current_epoch: current epoch
            print_every: print loss every n steps
            
        Returns:
            losses: list of losses
        """
        losses = []
        start_time = time.time()
        
        for epoch in range(current_epoch, epochs):
            self.train()
            self.gp_model.train()
            
            running_loss = 0.0
            running_nll = 0.0
            
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            
            for i, (batch_features, batch_coords) in pbar:
                batch_features = batch_features.to(self.device)
                batch_coords = batch_coords.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                x_reconstructed, _, _ = self(batch_coords)
                
                # Compute loss
                loss, nll = self.loss_function(batch_features, x_reconstructed)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Print statistics
                running_loss += loss.item()
                running_nll += nll.item()
                
                if i % print_every == (print_every - 1):
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / print_every:.4f}, "
                          f"nll: {running_nll / print_every:.4f}")
                    running_loss = 0.0
                    running_nll = 0.0
                
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(exp_path, f"checkpoint_epoch_{epoch + 1}.pt"))
            
            current_time = time.time()
            print(f"Epoch {epoch + 1} completed in {(current_time - start_time) / 60:.2f} minutes")
            start_time = current_time
            
        return losses
