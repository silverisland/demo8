import os
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from src.utils.config import config
from src.utils.memory import SiteMemoryBank
from src.models.generator import DiffusionGenerator
from src.models.cpt_loss import ConsistentPowerTransportLoss

def setup_logger(save_dir: Path):
    """Set up logging to console and file."""
    logger = logging.getLogger("Stage1")
    logger.setLevel(logging.INFO)
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    log_file = save_dir / "train.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def save_checkpoint(epoch, memory_bank, generator, optimizer, loss, save_path, is_best=False):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'memory_bank_state_dict': memory_bank.state_dict(),
        'generator_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(state, save_path)
    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(state, best_path)

def train_stage1(
    train_loader: DataLoader, 
    epochs: int, 
    lr: float, 
    device: str,
    save_dir: Path
):
    """
    Stage 1: Joint Pre-training of Site Latent Codebook and Diffusion Generator.
    Supervised by Physics-Informed CPT Loss.
    """
    logger = setup_logger(save_dir)
    logger.info(f"Starting Stage 1 Pre-training on device: {device}")
    
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
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_physics_bound = 0
        epoch_fluctuation = 0
        epoch_monotonicity = 0
        
        for batch in train_loader:
            nwp = batch['nwp'].to(device)
            power = batch['power'].to(device)
            ghi_cs = batch['ghi_clearsky'].to(device)
            site_id = batch['site_id'].to(device)
            
            # --- Identity Adaptation ---
            context_data = torch.cat([nwp, power], dim=-1)
            site_latent = memory_bank(context_data=context_data)
            
            # --- Diffusion Training (with Physics Constraint) ---
            res = generator.compute_loss(power, nwp, site_latent)
            
            # CPT Loss on the reconstructed p_generated (x0)
            cpt_results = criterion(
                p_generated=res['p_generated'], 
                p_target=power, 
                ghi_clearsky=ghi_cs
            )
            
            total_loss = res['l_mse'] + cpt_results['total_loss']
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += res['l_mse'].item()
            epoch_physics_bound += cpt_results['l_physics_bound'].item()
            epoch_fluctuation += cpt_results['l_fluctuation'].item()
            epoch_monotonicity += cpt_results['l_monotonicity'].item()
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.6f} | "
            f"Phys: {epoch_physics_bound/len(train_loader):.4f} | "
            f"Fluc: {epoch_fluctuation/len(train_loader):.4f} | "
            f"Trend: {epoch_monotonicity/len(train_loader):.4f}"
        )
        
        # Save checkpoints
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        save_checkpoint(
            epoch=epoch,
            memory_bank=memory_bank,
            generator=generator,
            optimizer=optimizer,
            loss=avg_loss,
            save_path=save_dir / "latest_model.pth",
            is_best=is_best
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Pre-training for PV Diffusion")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--exp_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    from src.data.mock_data_generator import generate_mock_dataloader
    
    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    # Experiment directory
    save_dir = Path("outputs") / f"stage1_{args.exp_name}"
    
    print(f"Generating {args.num_samples} mock samples...")
    train_loader = generate_mock_dataloader(num_samples=args.num_samples, batch_size=args.batch_size)
    
    train_stage1(
        train_loader=train_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device,
        save_dir=save_dir
    )
