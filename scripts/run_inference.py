import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import config
from src.utils.memory import SiteMemoryBank
from src.models.generator import DiffusionGenerator
from scripts.pipeline import PVGenerationPipeline

def load_trained_model(checkpoint_path: Path, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize components
    memory_bank = SiteMemoryBank(
        num_clusters=32, 
        latent_dim=config.site_latent_dim, 
        input_dim=config.memory_input_dim
    ).to(device)
    
    generator = DiffusionGenerator(
        nwp_dim=config.nwp_dim, 
        site_latent_dim=config.site_latent_dim, 
        timesteps=config.timesteps,
        hidden_dim=config.diffusion_hidden_dim
    ).to(device)
    
    # Load state dicts
    memory_bank.load_state_dict(checkpoint['memory_bank_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Successfully loaded model from {checkpoint_path} (Epoch {checkpoint['epoch']+1})")
    return memory_bank, generator

def run_inference(
    checkpoint_path: str, 
    num_sites: int = 1, 
    days_to_generate: int = 365,
    noise_scale: float = 1.0,
    custom_steps: Optional[int] = None
):
    # 1. Setup device and model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    memory_bank, generator = load_trained_model(Path(checkpoint_path), device)
    pipeline = PVGenerationPipeline(memory_bank, generator, device=device)
    
    # ... (Mock data preparation same as before)
    batch_size = num_sites
    future_len = 96 * days_to_generate
    context_len = 96 * 30
    
    print(f"Preparing input for {num_sites} sites, generating {days_to_generate} days...")
    print(f"Sampling configuration: noise_scale={noise_scale}, steps={custom_steps if custom_steps else config.timesteps}")
    
    # [B, Context_Len, 8] (NWP + Power)
    mock_context_data = torch.randn(batch_size, context_len, 8).to(device)
    # [B, Future_Len, 7]
    mock_nwp_full = torch.randn(batch_size, future_len, 7).to(device)
    # [B, Future_Len, 1]
    mock_ghi_cs_full = torch.abs(torch.randn(batch_size, future_len, 1)).to(device) * 1000
    
    # 3. Step 1: Extract Identity (Fingerprint)
    print("Step 1: Extracting site fingerprints...")
    site_latents = pipeline.extract_fingerprint(mock_context_data)
    
    # 4. Step 2: Generate Power
    print("Step 2: Generating synthetic power data (Reverse Diffusion)...")
    generated_power = pipeline.generate_full_year(
        site_latents, 
        mock_nwp_full, 
        mock_ghi_cs_full, 
        chunk_size=96,
        noise_scale=noise_scale,
        custom_steps=custom_steps
    )
    
    # ... (Rest of output results same as before)
    res_np = generated_power.cpu().numpy()
    print(f"Inference complete! Shape: {res_np.shape}")
    return res_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script for PV Diffusion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--num_sites", type=int, default=1)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Controls ruggedness (Cloud intensity). >1.0 for more, <1.0 for less.")
    parser.add_argument("--steps", type=int, default=None, help="Custom sampling steps.")
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint {args.checkpoint} not found.")
    else:
        run_inference(args.checkpoint, args.num_sites, args.days, args.noise_scale, args.steps)
