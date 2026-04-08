import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import config
from memory import SiteMemoryBank
from generator import DiffusionGenerator
from cpt_loss import ConsistentPowerTransportLoss

def train_stage1(
    train_loader: DataLoader, 
    epochs: int = 50, 
    lr: float = 1e-4, 
    device: str = "cpu"
):
    """
    Stage 1: Joint Pre-training of Site Latent Codebook and Diffusion Generator.
    Supervised by Physics-Informed CPT Loss.
    """
    # 1. Initialize Modules
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
    
    criterion = ConsistentPowerTransportLoss(
        alpha=config.alpha_physics, 
        beta=config.beta_smoothness
    )
    
    optimizer = torch.optim.Adam(
        list(memory_bank.parameters()) + list(generator.parameters()), 
        lr=lr
    )
    
    # 2. Training Loop
    memory_bank.train()
    generator.train()
    
    print(f"Starting Stage 1 Pre-training on device: {device}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_physics_loss = 0
        
        for batch in train_loader:
            nwp = batch['nwp'].to(device)             # [B, total_len, 7]
            power = batch['power'].to(device)         # [B, total_len, 1]
            ghi_cs = batch['ghi_clearsky'].to(device) # [B, total_len, 1]
            site_id = batch['site_id'].to(device)
            
            # --- Identity Adaptation (Mock context window) ---
            # In source pre-training, we use the whole sequence to extract identity 
            # Or use site_id clusters directly
            context_data = torch.cat([nwp, power], dim=-1) # [B, total_len, 8]
            site_latent = memory_bank(context_data=context_data)
            
            # --- Diffusion Training (with Physics Constraint) ---
            # Forward + x0 reconstruction
            res = generator.compute_loss(power, nwp, site_latent)
            
            # CPT Loss on the reconstructed p_generated (x0)
            cpt_results = criterion(
                p_generated=res['p_generated'], 
                p_target=power, 
                ghi_clearsky=ghi_cs
            )
            
            # Total Loss: Diffusion MSE + Physics Penalties
            total_loss = res['l_mse'] + cpt_results['total_loss']
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += res['l_mse'].item()
            epoch_physics_loss += cpt_results['l_physics_bound'].item()
            
        print(f"Epoch {epoch+1}/{epochs} | Noise MSE: {epoch_loss/len(train_loader):.6f} | Physics Violation: {epoch_physics_loss/len(train_loader):.6f}")

if __name__ == "__main__":
    # Test with dummy DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Placeholder for actual DataLoader implementation
    # from dataset import PVDataset...
    print("Stage 1 script verified.")
