from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    # Dimensions
    # nwp_dim = 7: [GHI, Temp, ClearSky_GHI, Hour_Sin, Hour_Cos, Month_Sin, Month_Cos]
    nwp_dim: int = 7
    site_latent_dim: int = 64
    # memory_input_dim = 8: NWP (7) + Observed_Power (1)
    memory_input_dim: int = 8 
    
    # Sequences
    history_len: int = 672 # 7 days
    future_len: int = 192  # 2 days
    total_len: int = 864   # 672 + 192
    
    # Diffusion
    timesteps: int = 100
    diffusion_hidden_dim: int = 128
    
    # Transformer
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    
    # Physics (CPT Loss)
    alpha_physics: float = 10.0
    beta_smoothness: float = 1.0

config = ModelConfig()
