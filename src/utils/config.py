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
    history_len: int = 0   # No longer used for generation
    future_len: int = 96   # 1 day @ 15min
    total_len: int = 96    # Matches future_len
    
    # Diffusion
    timesteps: int = 100
    diffusion_hidden_dim: int = 128
    
    # Transformer
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    
    # Physics (CPT Loss)
    alpha_physics: float = 10.0
    gamma_fluctuation: float = 0.5  # Match ruggedness/fluctuation
    delta_monotonicity: float = 0.1 # Trend consistency (CLT-inspired)

config = ModelConfig()
