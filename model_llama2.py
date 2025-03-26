"""
Llama 2-inspired transformer model implementation.
Incorporates key architectural features from the Llama 2 architecture
while adapting them for number sequence prediction.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from model_base import BaseTransformer, TransformerModelRegistry


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings as used in Llama 2.
    Based on the paper: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim: int, max_position: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base

        # Create and cache the rotation matrices
        # This is much more efficient than computing them on the fly
        self.register_buffer(
            'inv_freq',
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )
        self.register_buffer(
            'cos_cached',
            self._compute_cos_sin_cache()[0],
            persistent=False
        )
        self.register_buffer(
            'sin_cached',
            self._compute_cos_sin_cache()[1],
            persistent=False
        )

    def _compute_cos_sin_cache(self):
        # Generate rotation matrices for all positions up to max_position
        t = torch.arange(self.max_position, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache the cos and sin values
        cos_cached = emb.cos()
        sin_cached = emb.sin()

        return cos_cached, sin_cached

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Apply rotary embeddings to input tensors."""
        seq_len = x.shape[1]

        if position_ids is None:
            # Use default positional ids if not provided
            position_ids = torch.arange(seq_len, device=x.device)

        # Get the appropriate rotations from the cache
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [seq_len, 1, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [seq_len, 1, dim]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embeddings to query and key tensors."""
    # Reshape for broadcasting
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization as used in Llama 2.
    More numerically stable than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) as used in Llama 2.
    More efficient than standard multi-head attention by using a single KV head.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            dropout: float = 0.0,
            max_position: int = 2048
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Determine head dimension
        self.head_dim = head_dim or dim // num_heads

        # Projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)

        # In MQA, we only have one KV head regardless of number of query heads
        self.k_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim, bias=False)

        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position=max_position)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape query for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # For k and v, we keep them single-headed (this is the key difference in MQA)
        k = k.view(batch_size, seq_len, 1, self.head_dim)
        v = v.view(batch_size, seq_len, 1, self.head_dim)

        # Get rotary embeddings
        cos, sin = self.rotary_emb(x, position_ids)

        # Apply rotary embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch_size, 1, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch_size, 1, seq_len, head_dim]

        # Repeat k and v to match the number of query heads
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch_size, num_heads, seq_len, seq_len]

        # Apply causal mask
        if mask is None:
            # Default to causal mask
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
                diagonal=1
            )

        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output


class SwiGLUMLP(nn.Module):
    """
    SwiGLU MLP as used in Llama 2.
    A more efficient activation function than GELU.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class Llama2Block(nn.Module):
    """
    Transformer block as used in Llama 2.
    Uses RMSNorm, Multi-Query Attention, and SwiGLU.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_hidden_dim: Optional[int] = None,
            dropout: float = 0.0,
            max_position: int = 2048
    ):
        super().__init__()
        # Attention with pre-norm
        self.attn_norm = RMSNorm(dim)
        self.attention = MultiQueryAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            max_position=max_position
        )

        # MLP with pre-norm
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention with residual connection
        x = x + self.attention(self.attn_norm(x), mask)

        # MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))

        return x


@TransformerModelRegistry.register('llama2')
class Llama2Transformer(BaseTransformer):
    """
    Llama 2-inspired transformer model for sequence prediction.

    Key features:
    - RMSNorm instead of LayerNorm
    - Rotary positional embeddings
    - Multi-Query Attention
    - SwiGLU activation
    """

    def __init__(
            self,
            vocab_size: int,
            block_size: int = 128,
            n_embd: int = 256,
            n_layer: int = 6,
            n_head: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            **kwargs
    ):
        super().__init__(vocab_size=vocab_size, block_size=block_size, model_type='llama2')

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Token embeddings
        self.tok_embeddings = nn.Embedding(vocab_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Llama2Block(
                dim=n_embd,
                num_heads=n_head,
                mlp_hidden_dim=int(n_embd * mlp_ratio),
                dropout=dropout,
                max_position=block_size
            )
            for _ in range(n_layer)
        ])

        # Final normalization and output
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"Llama2Transformer with {sum(p.numel() for p in self.parameters())} parameters")

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Use scaled initialization for better training stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def forward(
            self,
            idx: torch.Tensor,
            targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            idx: Input token indices [batch_size, seq_len]
            targets: Target token indices [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Optional loss value
        """
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size, f"Input sequence length {seq_len} exceeds block size {self.block_size}"

        # Get token embeddings
        x = self.tok_embeddings(idx)
        x = self.dropout(x)

        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=idx.device),
            diagonal=1
        )

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Apply final normalization
        x = self.norm(x)

        # Compute logits
        logits = self.output(x)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
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
        """
        Generate new tokens autoregressively.

        Args:
            idx: Context tokens [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Optional top-k sampling parameter

        Returns:
            Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()

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
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Llama2Transformer':
        """Create a model instance from a configuration."""
        return cls(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config.get('n_embd', 256),
            n_layer=config.get('n_layer', 6),
            n_head=config.get('n_head', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.0)
        )