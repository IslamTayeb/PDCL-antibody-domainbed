import torch
import torch.nn as nn
import math

def exists(val):
    """Check if value exists (is not None)"""
    return val is not None

class AbAgRotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings for antibody-antigen sequences
    Based on Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
            
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
            
        return self.cos_cached, self.sin_cached
        
    def _rotary_embedding(self, x):
        seq_len = x.shape[1]
        cos, sin = self.forward(x, seq_len)
        
        # Reshape for applying rotary embedding
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        return torch.cat(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin),
            dim=-1
        )