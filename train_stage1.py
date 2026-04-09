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
        gamma=config.gamma_fluctuation,
        delta=config.delta_monotonicity
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
        epoch_physics_bound = 0
        epoch_fluctuation = 0
        epoch_monotonicity = 0
        
        for batch in train_loader:
            nwp = batch['nwp'].to(device)             # [B, total_len, 7]
            power = batch['power'].to(device)         # [B, total_len, 1]
            ghi_cs = batch['ghi_clearsky'].to(device) # [B, total_len, 1]
            site_id = batch['site_id'].to(device)
            
            # --- Identity Adaptation (Mock context window) ---
            # In source pre-training, we use the whole sequence to extract identity 
            context_data = torch.cat([nwp, power], dim=-1) # [B, total_len, 8]
            site_latent = memory_bank(context_data=context_data)
            
            # --- Diffusion Training (with Physics Constraint) ---
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
            epoch_physics_bound += cpt_results['l_physics_bound'].item()
            epoch_fluctuation += cpt_results['l_fluctuation'].item()
            epoch_monotonicity += cpt_results['l_monotonicity'].item()
            
        print(f"Epoch {epoch+1}/{epochs} | Noise MSE: {epoch_loss/len(train_loader):.6f} | "
              f"Phys: {epoch_physics_bound/len(train_loader):.4f} | "
              f"Fluc: {epoch_fluctuation/len(train_loader):.4f} | "
              f"Trend: {epoch_monotonicity/len(train_loader):.4f}")

if __name__ == "__main__":
    from mock_data_generator import generate_mock_dataloader
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    # 1. Create Mock DataLoader
    print("Generating mock data...")
    train_loader = generate_mock_dataloader(num_samples=200, batch_size=32)
    
    # 2. Run Stage 1 Pre-training (short run for testing)
    train_stage1(train_loader, epochs=10, lr=1e-4, device=device)
