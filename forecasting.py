import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, List, Optional, Tuple

class TimeSeriesTransformer(nn.Module):
    """
    Lightweight Transformer-based Forecaster.
    Predicts future power given past power and NWP (History + Future).
    """
    def __init__(
        self, 
        nwp_dim: int = 7, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        # Input projections
        self.power_proj = nn.Linear(1, d_model)
        self.nwp_proj = nn.Linear(nwp_dim, d_model)
        
        # Positional Encoding (using learned embeddings for simplicity)
        self.pos_emb = nn.Parameter(torch.randn(1000, d_model)) 
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, past_power: torch.Tensor, nwp_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            past_power: [B, history_len, 1]
            nwp_seq: [B, total_len, nwp_dim]
        Returns:
            future_power_pred: [B, future_len, 1]
        """
        B, total_len, _ = nwp_seq.shape
        history_len = past_power.shape[1]
        future_len = total_len - history_len
        
        # 1. Project and combine inputs
        # Combine NWP features for the whole sequence
        nwp_feat = self.nwp_proj(nwp_seq) # [B, total_len, d_model]
        
        # Project past power and pad with zeros for future positions to keep same seq length
        # Or more effectively, we only use history for encoder and attend to future NWP
        # Let's use a simpler approach: encode full NWP + past power
        past_power_feat = self.power_proj(past_power) # [B, history_len, d_model]
        
        # Pad power features with 0s for the future part
        future_power_feat = torch.zeros(B, future_len, d_model, device=past_power.device)
        full_power_feat = torch.cat([past_power_feat, future_power_feat], dim=1) # [B, total_len, d_model]
        
        # Final combined embedding
        x = nwp_feat + full_power_feat + self.pos_emb[:total_len, :]
        
        # 2. Transformer Encoding
        # We use a causal mask so future target power doesn't leak (though it's 0 here)
        # But more importantly, we want to predict future_len positions
        mask = torch.triu(torch.ones(total_len, total_len), diagonal=1).bool().to(x.device)
        
        feat = self.transformer_encoder(x, mask=mask)
        
        # 3. Project to Power
        # We only care about the predictions for the future window
        future_feat = feat[:, history_len:, :]
        preds = self.output_layer(future_feat)
        
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
            if synthetic_loader is not None:
                try:
                    syn_batch = next(iter(synthetic_loader))
                except StopIteration:
                    # Refresh synthetic iterator if it's shorter/longer
                    syn_batch = next(iter(synthetic_loader))
                
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
    
    model = TimeSeriesTransformer(nwp_dim=nwp_dim)
    
    past_p = torch.randn(B, H, 1)
    nwp = torch.randn(B, total, nwp_dim)
    
    out = model(past_p, nwp)
    print(f"Prediction Shape: {out.shape}") # [8, 192, 1]
