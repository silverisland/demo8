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

    def get_query(self, context_data: torch.Tensor) -> torch.Tensor:
        """
        Step 1: Encode context sequence into a raw query vector.
        Args:
            context_data: [batch_size, seq_len, input_dim]
        Returns:
            query: [batch_size, latent_dim]
        """
        _, h_n = self.encoder(context_data) # h_n: [2, batch_size, hidden_dim]
        query = torch.cat([h_n[0], h_n[1]], dim=-1) # [batch_size, hidden_dim * 2]
        return self.query_proj(query)

    def match_codebook(self, query: torch.Tensor) -> torch.Tensor:
        """
        Step 2: Map query vector to codebook space via attention.
        Args:
            query: [N, latent_dim] (N can be batch size or 1)
        Returns:
            site_latent: [N, latent_dim]
        """
        attn_scores = torch.matmul(query, self.codebook.t()) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        site_latent = torch.matmul(attn_weights, self.codebook)
        return site_latent

    def extract_fingerprint(self, context_data: torch.Tensor, aggregate: bool = False) -> torch.Tensor:
        """
        Extracts the site latent vector. 
        If aggregate is True, it means all samples in context_data belong to the SAME site (Inference mode).
        """
        query = self.get_query(context_data)
        
        if aggregate:
            # Mean aggregation across the temporal/batch samples for a single site
            query = query.mean(dim=0, keepdim=True)
            
        return self.match_codebook(query)

    def forward(self, context_data: Optional[torch.Tensor] = None, site_id: Optional[torch.Tensor] = None, aggregate: bool = False) -> torch.Tensor:
        """
        Forward pass for pre-training or few-shot inference.
        """
        if context_data is not None:
            return self.extract_fingerprint(context_data, aggregate=aggregate)
        elif site_id is not None:
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
