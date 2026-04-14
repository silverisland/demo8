import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple

class GatedResidualNetwork(nn.Module):
    """
    GRN from TFT: Provides non-linear processing and gating.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.lin1(x))
        h = self.lin2(h)
        # Gating: GLU-like behavior
        g = torch.sigmoid(self.gate(h))
        # Project skip
        skip = self.skip(x)
        return self.norm(skip + g * h)

class VariableSelectionNetwork(nn.Module):
    """
    VSN from TFT: Learns the importance of each input variable.
    """
    def __init__(self, num_vars: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.d_model = d_model
        
        # Encoders for each variable
        self.var_encoders = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout) for _ in range(num_vars)
        ])
        
        # Weights for each variable
        self.weight_net = GatedResidualNetwork(num_vars * d_model, d_model, num_vars, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, num_vars]
        B, L, _ = x.shape
        
        # Encode each variable individually
        var_outputs = []
        for i in range(self.num_vars):
            var_outputs.append(self.var_encoders[i](x[:, :, i:i+1]))
        
        # Stacked: [B, L, num_vars, d_model]
        stacked = torch.stack(var_outputs, dim=2)
        
        # Compute selection weights
        flat = stacked.view(B, L, -1)
        weights = F.softmax(self.weight_net(flat), dim=-1) # [B, L, num_vars]
        
        # Weighted sum
        out = torch.sum(weights.unsqueeze(-1) * stacked, dim=2) # [B, L, d_model]
        return out

class TemporalFusionForecaster(nn.Module):
    """
    Improved Forecaster inspired by Temporal Fusion Transformer (TFT).
    Uses Variable Selection and Gated Residual Networks.
    """
    def __init__(
        self, 
        nwp_dim: int = 7, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Variable Selection for NWP
        self.nwp_vsn = VariableSelectionNetwork(nwp_dim, d_model, dropout)
        
        # 2. Encoder for Power
        self.power_encoder = GatedResidualNetwork(1, d_model, d_model, dropout)
        
        # 3. Position Encoding
        self.pos_emb = nn.Parameter(torch.randn(1000, d_model)) 
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Final Output Head
        self.output_head = nn.Sequential(
            GatedResidualNetwork(d_model, d_model, d_model, dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, past_power: torch.Tensor, nwp_seq: torch.Tensor) -> torch.Tensor:
        """
        past_power: [B, history_len, 1]
        nwp_seq: [B, total_len, nwp_dim]
        """
        B, total_len, _ = nwp_seq.shape
        history_len = past_power.shape[1]
        
        # NWP Variable Selection
        nwp_feat = self.nwp_vsn(nwp_seq) # [B, total_len, d_model]
        
        # Power Encoding
        past_power_feat = self.power_encoder(past_power) # [B, history_len, d_model]
        
        # Combine: Future power is unknown, we only have NWP for future
        # Use zeros for future power features
        future_power_feat = torch.zeros(B, total_len - history_len, self.d_model, device=past_power.device)
        full_power_feat = torch.cat([past_power_feat, future_power_feat], dim=1)
        
        # Fusion
        x = nwp_feat + full_power_feat + self.pos_emb[:total_len, :]
        
        # Transformer (with causal mask)
        mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).bool().to(x.device)
        feat = self.transformer(x, mask=mask)
        
        # Prediction for future
        future_feat = feat[:, history_len:, :]
        preds = self.output_head(future_feat)
        
        return preds

def train_with_augmentation(
    model: nn.Module,
    real_loader: DataLoader,
    synthetic_loader: Optional[DataLoader],
    epochs: int = 10,
    lr: float = 1e-3,
    synthetic_ratio: float = 0.5
):
    """
    Training loop that mixes Real and Synthetic data.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    device = next(model.parameters()).device
    model.train()

    # Initialize synthetic iterator if loader is provided
    synthetic_iter = None
    if synthetic_loader is not None and synthetic_ratio > 0:
        synthetic_iter = iter(synthetic_loader)
    
    for epoch in range(epochs):
        total_loss = 0
        # Iterate through real data
        for i, real_batch in enumerate(real_loader):
            # 1. Real Data Step
            nwp = real_batch['nwp'].to(device)
            power = real_batch['power'].to(device)
            history_len = 672 # As per dataset.py
            
            past_p = power[:, :history_len, :]
            target_p = power[:, history_len:, :]
            
            pred_p = model(past_p, nwp)
            loss_real = criterion(pred_p, target_p)
            
            # 2. Synthetic Data Step (if provided)
            loss_syn = torch.tensor(0.0).to(device)
            if synthetic_loader is not None and synthetic_ratio > 0:
                try:
                    syn_batch = next(synthetic_iter)
                except StopIteration:
                    # Refresh synthetic iterator when exhausted
                    synthetic_iter = iter(synthetic_loader)
                    try:
                        syn_batch = next(synthetic_iter)
                    except StopIteration:
                        # synthetic_loader is empty, skip synthetic data
                        syn_batch = None

                if syn_batch is not None:
                    nwp_s = syn_batch['nwp'].to(device)
                    power_s = syn_batch['power'].to(device)

                    pred_s = model(power_s[:, :history_len, :], nwp_s)
                    loss_syn = criterion(pred_s, power_s[:, history_len:, :])
            
            # Weighted Loss
            loss = (1 - synthetic_ratio) * loss_real + synthetic_ratio * loss_syn
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(real_loader):.6f}")

if __name__ == "__main__":
    # Test Forecaster
    B = 8
    H = 672
    F = 192
    total = H + F
    nwp_dim = 7
    
    model = TemporalFusionForecaster(nwp_dim=nwp_dim)
    
    past_p = torch.randn(B, H, 1)
    nwp = torch.randn(B, total, nwp_dim)
    
    out = model(past_p, nwp)
    print(f"Prediction Shape: {out.shape}") # [8, 192, 1]
