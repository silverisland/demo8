import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistentPowerTransportLoss(nn.Module):
    """
    Consistent Power Transport (CPT) Constraint Loss - Anti-Smoothing Version.
    Inspired by CLT (Consistent Light Transport) and Physics-Informed ML.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 0.5, delta: float = 0.1):
        super().__init__()
        self.alpha = alpha   # Physics ceiling bound penalty weight
        self.gamma = gamma   # Fluctuation (ruggedness) matching weight
        self.delta = delta   # Monotonicity trend weight (CLT-inspired)
        
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
        
        # 2. L_physics_bound: Physical Ceiling (ReLU(P_gen - GHI_clearsky))
        violation = p_generated - ghi_clearsky
        l_physics_bound = torch.mean(F.relu(violation)**2)
        
        # 3. L_fluctuation: Ruggedness Matching (Anti-Smoothing)
        # We want the 'variation' of generated power to match target power
        diff_gen = p_generated[:, 1:] - p_generated[:, :-1]
        diff_target = p_target[:, 1:] - p_target[:, :-1]
        
        # Match the Mean Absolute Error of the first-order difference
        # This encourages the model to produce similar 'jumpiness' as the real data
        l_fluctuation = F.mse_loss(torch.abs(diff_gen), torch.abs(diff_target))
        
        # 4. L_monotonicity: Global Trend Consistency (CLT-inspired)
        # Penalize cases where GHI increases significantly but P decreases significantly.
        diff_ghi = ghi_clearsky[:, 1:] - ghi_clearsky[:, :-1]
        trend_violation = torch.relu(-(diff_gen * diff_ghi)) 
        l_monotonicity = torch.mean(trend_violation)
        
        # Total Loss
        total_loss = l_recon + \
                     self.alpha * l_physics_bound + \
                     self.gamma * l_fluctuation + \
                     self.delta * l_monotonicity
        
        return {
            'total_loss': total_loss,
            'l_recon': l_recon,
            'l_physics_bound': l_physics_bound,
            'l_fluctuation': l_fluctuation,
            'l_monotonicity': l_monotonicity
        }

if __name__ == "__main__":
    # Test Loss Function
    batch_size = 4
    seq_len = 192 # 2 days
    
    criterion = ConsistentPowerTransportLoss(alpha=10.0, gamma=1.0, delta=1.0)
    
    p_gen = torch.randn(batch_size, seq_len, 1) + 500
    p_gt = torch.randn(batch_size, seq_len, 1) + 500
    ghi = torch.zeros(batch_size, seq_len, 1)
    ghi[:, 48:144, :] = 1000
    
    losses = criterion(p_gen, p_gt, ghi)
    
    print(f"Total Loss: {losses['total_loss'].item()}")
    print(f"Physics Bound Loss: {losses['l_physics_bound'].item()}")
    print(f"Fluctuation Loss: {losses['l_fluctuation'].item()}")
    print(f"Monotonicity Loss: {losses['l_monotonicity'].item()}")
