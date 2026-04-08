import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from memory import SiteMemoryBank
from generator import DiffusionGenerator

class PVGenerationPipeline:
    """
    Pipeline for Few-Shot PV Data Generation.
    Integrates Site Latent extraction and Diffusion-based generation.
    """
    def __init__(
        self, 
        memory_bank: SiteMemoryBank, 
        generator: DiffusionGenerator,
        device: str = "cpu"
    ):
        self.memory_bank = memory_bank.to(device)
        self.generator = generator.to(device)
        self.device = device

    def extract_fingerprint(self, context_data: torch.Tensor) -> torch.Tensor:
        """
        Extract the site latent vector (fingerprint) using a small window of real data.
        
        Args:
            context_data: [batch_size, context_seq_len, input_dim] 
                          (input_dim typically includes NWP + Real Power)
        """
        self.memory_bank.eval()
        with torch.no_grad():
            site_latent = self.memory_bank(context_data=context_data.to(self.device))
        return site_latent

    def generate_synthetic_data(
        self, 
        site_latent: torch.Tensor, 
        nwp_seq: torch.Tensor, 
        ghi_clearsky: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates synthetic power for a specific period.
        
        Args:
            site_latent: [batch_size, latent_dim]
            nwp_seq: [batch_size, seq_len, nwp_dim]
            ghi_clearsky: [batch_size, seq_len, 1] (Physical anchor)
        """
        self.generator.eval()
        with torch.no_grad():
            # Sampling from Diffusion Model with step-by-step physical constraints
            p_gen = self.generator.sample(
                nwp=nwp_seq.to(self.device), 
                site_latent=site_latent.to(self.device),
                ghi_clearsky=ghi_clearsky.to(self.device)
            )
            
            # --- Physical Bounding (Hard Constraint Safety Layer) ---
            # 1. Non-negativity
            p_gen = torch.clamp(p_gen, min=0.0)
            # 2. Clear-sky Upper Bound
            p_gen = torch.min(p_gen, ghi_clearsky.to(self.device))
            
        return p_gen

    def generate_full_year(
        self, 
        site_latent: torch.Tensor, 
        full_year_nwp: torch.Tensor,
        full_year_ghi_cs: torch.Tensor,
        chunk_size: int = 192
    ) -> torch.Tensor:
        """
        Generates 12 months of synthetic data by processing in temporal chunks.
        
        Args:
            site_latent: [batch_size, latent_dim]
            full_year_nwp: [batch_size, total_steps, nwp_dim]
            full_year_ghi_cs: [batch_size, total_steps, 1]
        """
        total_steps = full_year_nwp.shape[1]
        all_generated = []
        
        # Site latent is shared across all chunks for this site
        for i in range(0, total_steps, chunk_size):
            end_idx = min(i + chunk_size, total_steps)
            
            nwp_chunk = full_year_nwp[:, i:end_idx, :]
            ghi_chunk = full_year_ghi_cs[:, i:end_idx, :]
            
            # Handle potential padding if last chunk is too small for model
            curr_chunk_len = nwp_chunk.shape[1]
            if curr_chunk_len < chunk_size:
                # Padding logic if required by architecture (our 1D ResNet might be flexible)
                # For this demo, we assume the ResNet can handle variable lengths or we pad.
                pass 
                
            p_chunk = self.generate_synthetic_data(site_latent, nwp_chunk, ghi_chunk)
            all_generated.append(p_chunk)
            
        return torch.cat(all_generated, dim=1)

if __name__ == "__main__":
    # Mock initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    memory = SiteMemoryBank(num_clusters=32, latent_dim=64, input_dim=8)
    gen = DiffusionGenerator(nwp_dim=7, site_latent_dim=64, timesteps=10) # Small timesteps for test
    
    pipeline = PVGenerationPipeline(memory, gen, device=device)
    
    # Mock data
    batch_size = 2
    context_len = 2880 # 1 month
    full_year_len = 192 * 5 # Small mock "year"
    
    context_data = torch.randn(batch_size, context_len, 8)
    full_year_nwp = torch.randn(batch_size, full_year_len, 7)
    full_year_ghi_cs = torch.abs(torch.randn(batch_size, full_year_len, 1)) * 1000
    
    # 1. Adaptation: Extract Fingerprint
    print("Extracting site fingerprint...")
    latent = pipeline.extract_fingerprint(context_data)
    print(f"Latent shape: {latent.shape}")
    
    # 2. Generation: Full Year
    print("Generating full year synthetic data...")
    synthetic_power = pipeline.generate_full_year(latent, full_year_nwp, full_year_ghi_cs)
    print(f"Synthetic data shape: {synthetic_power.shape}")
    
    # Final check on physical bounds
    max_violation = (synthetic_power - full_year_ghi_cs.to(device)).max()
    print(f"Max physical violation (should be <=0): {max_violation.item()}")
