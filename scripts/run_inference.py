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
    days_to_generate: int = 365
):
    # 1. Setup device and model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    memory_bank, generator = load_trained_model(Path(checkpoint_path), device)
    pipeline = PVGenerationPipeline(memory_bank, generator, device=device)
    
    # 2. Prepare Mock Input Data for Inference
    # In practice, you would load real NWP data and 1-month context data here.
    batch_size = num_sites
    future_len = 96 * days_to_generate
    context_len = 96 * 30 # 1 month context for site identity extraction
    
    print(f"Preparing input for {num_sites} sites, generating {days_to_generate} days...")
    
    # [B, Context_Len, 8] (NWP + Power)
    mock_context_data = torch.randn(batch_size, context_len, 8).to(device)
    # [B, Future_Len, 7] (GHI, Temp, ClearSky_GHI, Hour_Sin, Hour_Cos, Month_Sin, Month_Cos)
    mock_nwp_full = torch.randn(batch_size, future_len, 7).to(device)
    # [B, Future_Len, 1]
    mock_ghi_cs_full = torch.abs(torch.randn(batch_size, future_len, 1)).to(device) * 1000
    
    # 3. Step 1: Extract Identity (Fingerprint)
    print("Step 1: Extracting site fingerprints...")
    site_latents = pipeline.extract_fingerprint(mock_context_data)
    
    # 4. Step 2: Generate Power
    print("Step 2: Generating synthetic power data (Reverse Diffusion)...")
    # We use chunked generation for stability and memory efficiency
    generated_power = pipeline.generate_full_year(
        site_latents, 
        mock_nwp_full, 
        mock_ghi_cs_full, 
        chunk_size=96 # Generate day by day
    )
    
    # 5. Output results
    res_np = generated_power.cpu().numpy()
    print(f"Inference complete! Shape: {res_np.shape}")
    print(f"Sample generated power (first 10 steps of site 0):")
    print(res_np[0, :10, 0])
    
    return res_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script for PV Diffusion")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--num_sites", type=int, default=1)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint {args.checkpoint} not found.")
    else:
        run_inference(args.checkpoint, args.num_sites, args.days)
