import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistentPowerTransportLoss(nn.Module):
    """
    Consistent Power Transport (CPT) Constraint Loss.
    Ensures generated PV power is physically plausible.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha # Physics bound penalty weight
        self.beta = beta   # Smoothness penalty weight
        
    def forward(
        self, 
        p_generated: torch.Tensor, 
        p_target: torch.Tensor, 
        ghi_clearsky: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            p_generated: Generated PV power [batch_size, seq_len, 1]
            p_target: Ground truth PV power [batch_size, seq_len, 1]
            ghi_clearsky: Physical upper bound (clear-sky GHI) [batch_size, seq_len, 1]
        """
        # 1. L_recon: Standard Reconstruction Loss (MSE)
        l_recon = F.mse_loss(p_generated, p_target)
        
        # 2. L_physics_bound: ReLU(P_gen - GHI_clearsky)
        # Penalizes power exceeding the physical upper bound
        # We assume power and ghi are on similar scales or normalized
        violation = p_generated - ghi_clearsky
        l_physics_bound = torch.mean(F.relu(violation)**2)
        
        # 3. L_smoothness: Temporal difference penalty
        # Prevents unnatural high-frequency jitter (PV power changes smoothly unless NWP changes)
        # Note: True PV can be volatile due to clouds, but the generator's baseline should be stable.
        diff_gen = p_generated[:, 1:] - p_generated[:, :-1]
        l_smoothness = torch.mean(diff_gen**2)
        
        # Total Loss
        total_loss = l_recon + self.alpha * l_physics_bound + self.beta * l_smoothness
        
        return {
            'total_loss': total_loss,
            'l_recon': l_recon,
            'l_physics_bound': l_physics_bound,
            'l_smoothness': l_smoothness
        }

if __name__ == "__main__":
    # Test Loss Function
    batch_size = 4
    seq_len = 192 # 2 days
    
    criterion = ConsistentPowerTransportLoss(alpha=10.0, beta=1.0)
    
    p_gen = torch.randn(batch_size, seq_len, 1) + 500 # Slightly positive
    p_gt = torch.randn(batch_size, seq_len, 1) + 500
    # Create clear-sky GHI (bell shape dummy)
    ghi = torch.zeros(batch_size, seq_len, 1)
    ghi[:, 48:144, :] = 1000 # Daytime dummy
    
    losses = criterion(p_gen, p_gt, ghi)
    
    print(f"Total Loss: {losses['total_loss'].item()}")
    print(f"Physics Bound Loss: {losses['l_physics_bound'].item()}")
    print(f"Smoothness Loss: {losses['l_smoothness'].item()}")
