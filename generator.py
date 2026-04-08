import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List

class TimeStepEmbedding(nn.Module):
    """
    Sinusoidal embedding for diffusion time steps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        cond: [B, cond_dim]
        """
        h = F.relu(self.norm1(self.conv1(x)))
        # Add condition (time + site latent)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = F.relu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

class EpsilonNet(nn.Module):
    """
    Backbone for the Diffusion Model (1D ResNet).
    Predicts noise epsilon given x_t, t, and conditions.
    """
    def __init__(self, nwp_dim: int, site_latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Total condition dimension: NWP at each step + TimeStep embedding + Site Latent
        self.time_mlp = nn.Sequential(
            TimeStepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Initial projection of power sequence
        self.init_conv = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        
        # Stacks of residual blocks
        # Condition for blocks = NWP (per-step) + TimeStep + Site Latent
        self.block1 = ResidualBlock1D(hidden_dim, hidden_dim, nwp_dim + hidden_dim + site_latent_dim)
        self.block2 = ResidualBlock1D(hidden_dim, hidden_dim, nwp_dim + hidden_dim + site_latent_dim)
        
        self.final_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, nwp: torch.Tensor, site_latent: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, seq_len, 1] - Noisy power
        t: [B] - Diffusion step
        nwp: [B, seq_len, nwp_dim]
        site_latent: [B, site_latent_dim]
        """
        x = x_t.transpose(1, 2) # [B, 1, L]
        t_embed = self.time_mlp(t) # [B, hidden_dim]
        
        # Broad cast time and site latent to match seq_len if we were doing per-step cond
        # But here we use ResidualBlock1D's cond_proj which handles the global cond
        # We'll concatenate global site_latent and global t_embed
        global_cond = torch.cat([t_embed, site_latent], dim=-1) # [B, hidden_dim + site_latent_dim]
        
        # For NWP, we might want to feed it into the blocks as a sequence. 
        # Simplified: Concatenate mean NWP to global cond, or use as feature map.
        # Let's concatenate mean NWP for global context in this block structure.
        nwp_mean = nwp.mean(dim=1) # [B, nwp_dim]
        full_cond = torch.cat([global_cond, nwp_mean], dim=-1) # [B, ...]
        
        h = self.init_conv(x)
        h = self.block1(h, full_cond)
        h = self.block2(h, full_cond)
        
        epsilon_theta = self.final_conv(h)
        return epsilon_theta.transpose(1, 2) # [B, L, 1]

class DiffusionGenerator(nn.Module):
    """
    Physics-Informed Diffusion Model for PV Power Generation.
    """
    def __init__(
        self, 
        nwp_dim: int = 7, 
        site_latent_dim: int = 64, 
        timesteps: int = 100,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.timesteps = timesteps
        self.eps_net = EpsilonNet(nwp_dim, site_latent_dim, hidden_dim)
        
        # Diffusion schedules (Linear schedule)
        beta = torch.linspace(1e-4, 0.02, timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_bar', alpha_bar)

    def compute_loss(
        self, 
        real_power: torch.Tensor, 
        nwp: torch.Tensor, 
        site_latent: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Modified DDPM loss. Returns both MSE and x0_pred for Physics Constraints.
        """
        B, L, _ = real_power.shape
        t = torch.randint(0, self.timesteps, (B,), device=real_power.device).long()
        
        noise = torch.randn_like(real_power)
        
        # q(x_t | x_0)
        a_bar = self.alpha_bar[t].view(B, 1, 1)
        x_t = torch.sqrt(a_bar) * real_power + torch.sqrt(1 - a_bar) * noise
        
        # Predict noise: epsilon_theta
        eps_theta = self.eps_net(x_t, t, nwp, site_latent)
        
        # Standard Diffusion Loss (MSE on noise)
        l_mse = F.mse_loss(eps_theta, noise)
        
        # --- Physics-Informed Bridge ---
        # Derive x0_pred from x_t and eps_theta to apply CPT Loss
        # x0 = (x_t - sqrt(1 - a_bar) * eps_theta) / sqrt(a_bar)
        x0_pred = (x_t - torch.sqrt(1 - a_bar) * eps_theta) / torch.sqrt(a_bar)
        
        return {
            'l_mse': l_mse,
            'p_generated': x0_pred # Reconstructed power for CPT Loss
        }

    @torch.no_grad()
    def sample(
        self, 
        nwp: torch.Tensor, 
        site_latent: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse Diffusion Process (Inference).
        """
        B, L, _ = nwp.shape
        device = nwp.device
        
        # Start from pure noise
        x = torch.randn(B, L, 1, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            eps_theta = self.eps_net(x, t, nwp, site_latent)
            
            # Reverse step
            a = self.alpha[i]
            a_bar = self.alpha_bar[i]
            beta = self.beta[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            # x_{t-1} = 1/sqrt(a_t) * (x_t - (1-a_t)/sqrt(1-a_bar_t) * eps_theta) + sigma_t * z
            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - a_bar)) * eps_theta) + torch.sqrt(beta) * noise
            
        return x # Generated Power [B, L, 1]

if __name__ == "__main__":
    # Test Diffusion Generator
    batch_size = 4
    seq_len = 192
    nwp_dim = 7
    site_latent_dim = 64
    
    model = DiffusionGenerator(nwp_dim=nwp_dim, site_latent_dim=site_latent_dim)
    
    nwp = torch.randn(batch_size, seq_len, nwp_dim)
    site_latent = torch.randn(batch_size, site_latent_dim)
    real_power = torch.randn(batch_size, seq_len, 1)
    
    # Training loss
    loss = model.compute_loss(real_power, nwp, site_latent)
    print(f"Diffusion Training Loss: {loss.item()}")
    
    # Inference sampling
    generated = model.sample(nwp, site_latent)
    print(f"Generated Power Shape: {generated.shape}") # [4, 192, 1]
