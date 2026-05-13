"""
Resampler4: VisionHead-like module to convert SmolVLM2 output to fixed 4 queries.
Matches the interface that DM_Adapter was pretrained with.

Input: [B, L, 1024] - SmolVLM2 hidden states (variable length)
Output: [B, 4, 1024] - Fixed 4 query tokens (match VisionHead output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Resampler4(nn.Module):
    """
    Resampler to convert SmolVLM2 output [B, L, 1024] → [B, 4, 1024]
    Tương tự VisionHead trong OmniVideo gốc (4 queries cố định)
    
    Uses cross-attention with 4 learnable queries to aggregate information
    from variable-length SmolVLM2 sequence into fixed 4 tokens.
    """
    
    def __init__(self, d_model=1024, num_queries=4, n_heads=16, dropout=0.0):
        """
        Args:
            d_model: Hidden dimension (1024 for SmolVLM2)
            num_queries: Number of output queries (4 to match VisionHead)
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        
        # Learnable queries (similar to VisionHead's 4 queries)
        # Initialize with small random values
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        
        # Layer norms for query and key/value
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        
        # Multi-head cross-attention
        self.attn = nn.MultiheadAttention(
            d_model, 
            n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # MLP for post-processing
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize queries with small random values (already done in __init__)
        # Initialize MLP layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, attn_mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor with shape [B, L, 1024]
               where L is variable sequence length (50-200)
            attn_mask: Optional attention mask [B, L] where 1=keep, 0=pad
        
        Returns:
            Output tensor with shape [B, 4, 1024]
        """
        B = x.shape[0]
        
        # Expand learnable queries to batch size: [1, 4, 1024] -> [B, 4, 1024]
        q = self.queries.expand(B, -1, -1)
        
        # Apply layer norm
        q = self.ln_q(q)
        kv = self.ln_kv(x)
        
        # Prepare key_padding_mask for MultiheadAttention
        # MultiheadAttention uses key_padding_mask where True means "ignore"
        key_padding_mask = None
        if attn_mask is not None:
            # Convert: 1=keep, 0=pad -> True=ignore, False=keep
            key_padding_mask = ~attn_mask.bool()  # [B, L]
        
        # Cross-attention: queries attend to SmolVLM2 sequence
        out, _ = self.attn(
            query=q,           # [B, 4, 1024]
            key=kv,            # [B, L, 1024]
            value=kv,          # [B, L, 1024]
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Post-process with MLP (residual connection)
        out = out + self.mlp(out)
        
        return out  # [B, 4, 1024]
