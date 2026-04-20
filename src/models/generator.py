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

class FiLMBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Conditions (NWP + Site Latent) scale and shift the internal features.
    """
    def __init__(self, cond_dim: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L], cond: [B, cond_dim]
        style = self.mlp(cond).unsqueeze(-1) # [B, 2*C, 1]
        gamma, beta = style.chunk(2, dim=1)
        return x * (1 + gamma) + beta

class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Fusion Module.
    Backbone features (Query) attend to dynamic NWP features and static Site Latent (Key/Value).
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model] - Backbone features (Query)
        context: [B, L_ctx, d_model] - Combined NWP and Site Fingerprint (Key/Value)
        """
        # x is [B, L, C], MultiheadAttention expects [B, L, C] if batch_first=True
        attn_out, _ = self.attn(query=x, key=context, value=context)
        return self.norm(x + attn_out)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Keep FiLM for time-step conditioning only
        self.time_film = FiLMBlock(time_cond_dim, out_channels)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        t_emb: [B, time_cond_dim]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Apply time-step modulation
        h = self.time_film(h, t_emb)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.shortcut(x)

class EpsilonNet(nn.Module):
    """
    Enhanced Backbone for Diffusion Model.
    - Channel Concat: (Noisy Power, Clear-sky GHI)
    - Cross-Attention: Backbone attends to (NWP Encoder Output, Site Fingerprint)
    """
    def __init__(self, nwp_dim: int, site_latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            TimeStepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. NWP Sequence Encoder
        self.nwp_encoder = NWPEncoder(nwp_dim, hidden_dim)
        
        # 3. Site Latent Projection (to match hidden_dim for sequence injection)
        self.site_proj = nn.Linear(site_latent_dim, hidden_dim)
        
        # 4. Initial projection: Input is 2 channels [Noisy Power, GHI Clearsky]
        self.init_conv = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        
        # 5. Fusion & Attention Layers
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, hidden_dim, hidden_dim) # Cond on t_emb only
            for _ in range(4)
        ])
        
        self.cross_attn = CrossAttentionBlock(hidden_dim)
        
        self.final_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        nwp: torch.Tensor, 
        site_latent: torch.Tensor,
        ghi_clearsky: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t: [B, L, 1]
        ghi_clearsky: [B, L, 1]
        """
        # --- 1. Channel Concatenation (Physical Anchor) ---
        # Concat noisy power and GHI anchor in the channel dimension
        x_input = torch.cat([x_t, ghi_clearsky], dim=-1) # [B, L, 2]
        h = self.init_conv(x_input.transpose(1, 2)) # [B, hidden_dim, L]
        
        # --- 2. Encode Context (NWP + Site Fingerprint) ---
        t_embed = self.time_mlp(t) # [B, hidden_dim]
        nwp_feat = self.nwp_encoder(nwp.transpose(1, 2)) # [B, hidden_dim, L]
        site_feat = self.site_proj(site_latent).unsqueeze(1) # [B, 1, hidden_dim]
        
        # Context for Cross-Attention: Combine dynamic NWP and static Site Fingerprint
        # context: [B, L+1, hidden_dim]
        context = torch.cat([nwp_feat.transpose(1, 2), site_feat], dim=1)
        
        # --- 3. Backbone Processing ---
        for block in self.blocks:
            h = block(h, t_embed)
            
        # --- 4. Cross-Attention Fusion ---
        # Query: h (backbone), Key/Value: context (NWP + Site)
        h = h.transpose(1, 2) # [B, L, hidden_dim]
        h = self.cross_attn(h, context) # [B, L, hidden_dim]
        h = h.transpose(1, 2) # [B, hidden_dim, L]
            
        epsilon_theta = self.final_conv(h)
        return epsilon_theta.transpose(1, 2) # [B, L, 1]

class DiffusionGenerator(nn.Module):
    """
    Physics-Informed Diffusion Model (Refactored with Cross-Attention).
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
        site_latent: torch.Tensor,
        ghi_clearsky: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = real_power.shape
        t = torch.randint(0, self.timesteps, (B,), device=real_power.device).long()
        noise = torch.randn_like(real_power)
        
        a_bar = self.alpha_bar[t].view(B, 1, 1)
        x_t = torch.sqrt(a_bar) * real_power + torch.sqrt(1 - a_bar) * noise
        
        # Pass ghi_clearsky for channel concat
        eps_theta = self.eps_net(x_t, t, nwp, site_latent, ghi_clearsky)
        
        l_mse = F.mse_loss(eps_theta, noise)
        x0_pred = (x_t - torch.sqrt(1 - a_bar) * eps_theta) / torch.sqrt(a_bar)
        
        return {
            'l_mse': l_mse,
            'p_generated': x0_pred
        }

    @torch.no_grad()
    def sample(
        self, 
        nwp: torch.Tensor, 
        site_latent: torch.Tensor,
        ghi_clearsky: torch.Tensor,
        noise_scale: float = 1.0,
        custom_steps: Optional[int] = None
    ) -> torch.Tensor:
        B, L, _ = nwp.shape
        device = nwp.device
        x = torch.randn(B, L, 1, device=device)
        
        step_indices = np.arange(self.timesteps)
            
        for i in reversed(step_indices):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Pass ghi_clearsky for channel concat
            eps_theta = self.eps_net(x, t, nwp, site_latent, ghi_clearsky)
            
            a = self.alpha[i]
            a_bar = self.alpha_bar[i]
            beta = self.beta[i]
            
            noise = torch.randn_like(x) * noise_scale if i > 0 else 0
            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - a_bar)) * eps_theta) + torch.sqrt(beta) * noise
            
            x = torch.clamp(x, min=0.0)
            x = torch.min(x, ghi_clearsky.to(device))
            
        return x

if __name__ == "__main__":
    # Test Diffusion Generator
    batch_size = 4
    seq_len = 96 # 1 day
    nwp_dim = 7
    site_latent_dim = 64
    
    model = DiffusionGenerator(nwp_dim=nwp_dim, site_latent_dim=site_latent_dim)
    
    nwp = torch.randn(batch_size, seq_len, nwp_dim)
    site_latent = torch.randn(batch_size, site_latent_dim)
    real_power = torch.randn(batch_size, seq_len, 1)
    
    # Training loss
    loss_res = model.compute_loss(real_power, nwp, site_latent)
    print(f"Diffusion Training Loss (MSE): {loss_res['l_mse'].item()}")
    
    # Inference sampling
    generated = model.sample(nwp, site_latent)
    print(f"Generated Power Shape: {generated.shape}") # [4, 96, 1]
