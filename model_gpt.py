import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from model_base import BaseTransformer, TransformerModelRegistry


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.1, resid_pdrop: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Key, query, value projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # Output projection and dropout
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # Mask for causal attention
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1, 1, 1024, 1024))
        )

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, sequence length, embedding dimension

        # Calculate query, key, values for all heads in batch
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Compute attention scores (affinities)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        # Apply causal mask (mask out future positions)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        # Apply softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection and dropout
        y = self.resid_drop(self.proj(y))

        return y


class MLP(nn.Module):
    """Simple MLP block with GELU activation."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.1, resid_pdrop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = MLP(n_embd, resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        # MLP with residual connection
        x = x + self.mlp(self.ln2(x))
        return x


@TransformerModelRegistry.register('gpt2')
class GPT2Transformer(BaseTransformer):
    """GPT-2 style transformer model for number sequence prediction."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, block_size=block_size, model_type='gpt2')
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])

        # Final layer norm and head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"GPT2 model with {sum(p.numel() for p in self.parameters())} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"

        # Get token embeddings and add positional embeddings
        token_embeddings = self.tok_emb(idx)  # (B, T, C)
        position_embeddings = self.pos_emb[:, :T, :]  # (1, T, C)
        x = self.drop(token_embeddings + position_embeddings)  # (B, T, C)

        # Apply transformer blocks
        x = self.blocks(x)  # (B, T, C)

        # Apply final layer norm
        x = self.ln_f(x)  # (B, T, C)

        # Project to vocabulary
        logits = self.head(x)  # (B, T, vocab_size)

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
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the context
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'embd_pdrop': self.embd_pdrop,
            'attn_pdrop': self.attn_pdrop,
            'resid_pdrop': self.resid_pdrop,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GPT2Transformer':
        """Create a model instance from a configuration."""
        return cls(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config.get('n_embd', 256),
            n_layer=config.get('n_layer', 6),
            n_head=config.get('n_head', 8),
            embd_pdrop=config.get('embd_pdrop', 0.1),
            attn_pdrop=config.get('attn_pdrop', 0.1),
            resid_pdrop=config.get('resid_pdrop', 0.1)
        )
