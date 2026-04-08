import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class SiteMemoryBank(nn.Module):
    """
    Site Latent Codebook (Prototype Memory).
    Decouples transient weather from steady-state site physical efficiency.
    """
    def __init__(
        self, 
        num_clusters: int = 32, 
        latent_dim: int = 64, 
        input_dim: int = 8, # NWP (7) + Power (1)
        hidden_dim: int = 128
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        
        # The Codebook: learnable embeddings for site clusters
        self.codebook = nn.Parameter(torch.randn(num_clusters, latent_dim))
        
        # Lightweight Encoder to generate Query from context data
        # Using GRU + MaxPool to get a global site representation
        self.encoder = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Map encoder hidden state to Query dimension (latent_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Multi-head attention or simple scaled dot-product attention
        self.scale = latent_dim ** -0.5

    def extract_fingerprint(self, context_data: torch.Tensor) -> torch.Tensor:
        """
        Extracts the site latent vector from a context window (e.g., 1 month of data).
        Args:
            context_data: [batch_size, seq_len, input_dim]
        Returns:
            site_latent: [batch_size, latent_dim]
        """
        # 1. Encode context sequence
        # We don't need the whole sequence output, just the final/aggregated state
        _, h_n = self.encoder(context_data) # h_n: [2, batch_size, hidden_dim]
        
        # Concatenate bidirectional hidden states
        query = torch.cat([h_n[0], h_n[1]], dim=-1) # [batch_size, hidden_dim * 2]
        query = self.query_proj(query) # [batch_size, latent_dim]
        
        # 2. Attention over Codebook
        # Query: [batch_size, latent_dim]
        # Codebook (Keys/Values): [num_clusters, latent_dim]
        
        # Compute attention weights: [batch_size, num_clusters]
        attn_scores = torch.matmul(query, self.codebook.t()) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of codebook entries: [batch_size, latent_dim]
        site_latent = torch.matmul(attn_weights, self.codebook)
        
        return site_latent

    def forward(self, context_data: Optional[torch.Tensor] = None, site_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for pre-training or few-shot inference.
        """
        if context_data is not None:
            return self.extract_fingerprint(context_data)
        elif site_id is not None:
            # Direct lookup if we know the site cluster (useful for pre-training experiments)
            # In actual few-shot, we always use context_data
            return self.codebook[site_id % self.num_clusters]
        else:
            raise ValueError("Either context_data or site_id must be provided")

if __name__ == "__main__":
    # Test Codebook
    batch_size = 4
    seq_len = 2880 # 1 month @ 15min
    input_dim = 8
    
    model = SiteMemoryBank()
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    latent = model(context_data=dummy_input)
    print(f"Site Latent Vector Shape: {latent.shape}") # Expected: [4, 64]
