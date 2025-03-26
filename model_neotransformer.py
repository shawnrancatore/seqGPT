import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from model_base import BaseTransformer, TransformerModelRegistry


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding implementation.
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Create rotary embeddings once
        self.register_buffer(
            "cos_cached", 
            self._compute_cos_sin_cache(max_seq_len, dim)[0]
        )
        self.register_buffer(
            "sin_cached", 
            self._compute_cos_sin_cache(max_seq_len, dim)[1]
        )

    def _compute_cos_sin_cache(self, seq_len: int, dim: int):
        """Compute cos and sin cache for rotary embeddings."""
        # Ensure dim is even
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"
        
        # Create position indices
        position = torch.arange(seq_len).float().unsqueeze(1)  # [seq_len, 1]
        
        # Create dimension indices
        dim_indices = torch.arange(0, dim, 2).float() / dim  # [dim/2]
        
        # Compute theta
        theta = position * torch.exp(-dim_indices * math.log(self.base))  # [seq_len, dim/2]
        
        # Compute cos and sin
        cos = torch.cos(theta)  # [seq_len, dim/2]
        sin = torch.sin(theta)  # [seq_len, dim/2]
        
        # Duplicate to match original dimension
        cos = torch.repeat_interleave(cos, 2, dim=1)  # [seq_len, dim]
        sin = torch.repeat_interleave(sin, 2, dim=1)  # [seq_len, dim]
        
        return cos, sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        return x_rotated

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys."""
        # Get the appropriate part of the cache
        cos = self.cos_cached[:seq_len]  # [seq_len, dim]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim]
        
        # Reshape for broadcasting
        # For q: [batch, heads, seq_len, head_dim]
        # Need cos/sin: [1, 1, seq_len, head_dim]
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        
        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class SelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        dropout: float = 0.1, 
        qk_norm: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        assert dim % n_heads == 0, "Dimension must be divisible by number of heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # QKV projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Optional QK normalization
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with rotary positional embeddings."""
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Apply rotary positional embeddings
        q, k = self.rotary_emb(q, k, seq_len)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = scale * torch.matmul(q, k.transpose(-2, -1))
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Get attention weights with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        out = self.to_out(out)
        
        return out


class SwiGLU(nn.Module):
    """
    SwiGLU activation as used in modern transformers like PaLM.
    Based on the paper: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.to_hidden = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.to_hidden(x)
        hidden, gate = hidden.chunk(2, dim=-1)
        return self.to_out(F.silu(gate) * hidden)


class NeoTransformerBlock(nn.Module):
    """Modern transformer block with pre-norm, SwiGLU, and rotary embeddings."""
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qk_norm: bool = True,
        max_seq_len: int = 2048
    ):
        super().__init__()
        # Attention with pre-norm
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            n_heads=n_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            max_seq_len=max_seq_len
        )
        
        # MLP with pre-norm
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


@TransformerModelRegistry.register('neotransformer')
class NeoTransformer(BaseTransformer):
    """
    Modern transformer model incorporating recent architecture improvements:
    - Rotary Positional Embeddings (RoPE) instead of learned positional embeddings
    - SwiGLU activation function instead of GELU
    - QK-normalization for more stable training
    - Pre-normalization architecture
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qk_norm: bool = True,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, block_size=block_size, model_type='neotransformer')
        
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.qk_norm = qk_norm
        
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            NeoTransformerBlock(
                dim=n_embd,
                n_heads=n_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                qk_norm=qk_norm,
                max_seq_len=block_size
            )
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output head
        self.norm_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"NeoTransformer model with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model with small standard deviation for better stability."""
        if isinstance(module, nn.Linear):
            # Slightly smaller std for more stable training
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        # Get token embeddings (no added positional embeddings - using rotary in attention)
        x = self.tok_emb(idx)  # (B, T, C)
        x = self.drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.norm_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Ensure we don't exceed block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last timestep
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append the new token to the context
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'qk_norm': self.qk_norm,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NeoTransformer':
        """Create a model instance from a configuration."""
        return cls(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config.get('n_embd', 256),
            n_layer=config.get('n_layer', 6),
            n_head=config.get('n_head', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            qk_norm=config.get('qk_norm', True)
        )
