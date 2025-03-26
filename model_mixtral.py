import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from model_base import BaseTransformer, TransformerModelRegistry
from model_neotransformer import RotaryPositionalEmbedding, SelfAttention
from pattern_aware_moe import PatternAwareMixtralBlock

class MoERouter(nn.Module):
    """
    Router for Mixture of Experts layer.
    Implements top-k routing with load balancing.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Routing projections
        self.routing = nn.Linear(dim, num_experts, bias=False)

        # Initialize with small values
        nn.init.normal_(self.routing.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the router.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            expert_weights: Routing weights of shape [batch_size, seq_len, top_k]
            expert_indices: Expert indices of shape [batch_size, seq_len, top_k]
            router_logits: Raw routing logits of shape [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, _ = x.shape

        # Compute routing logits
        router_logits = self.routing(x)  # [batch_size, seq_len, num_experts]

        # Get weights and indices using top-k routing
        expert_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # [batch_size, seq_len, top_k]

        # Apply softmax over the selected experts only
        expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_weights, expert_indices, router_logits
class FFExpert(nn.Module):
    """Expert feed-forward network with SwiGLU activation."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        hidden = self.w1(x)
        gate = F.silu(self.w3(x))
        return self.dropout(self.w2(hidden * gate))


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    Based on Mixtral paper: https://arxiv.org/abs/2401.04088
    """

    def __init__(
            self,
            dim: int,
            num_experts: int = 8,
            hidden_dim: int = None,
            top_k: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Set hidden dimension
        if hidden_dim is None:
            hidden_dim = dim * 4

        # Create router
        self.router = MoERouter(dim, num_experts, top_k)

        # Create experts
        self.experts = nn.ModuleList([
            FFExpert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        # For load balancing loss
        self.router_z_loss_coef = 0.001

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with auxiliary load balancing loss.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]

        Returns:
            y: Output tensor of shape [batch_size, seq_len, dim]
            aux_loss: Dictionary containing auxiliary loss for load balancing
        """
        batch_size, seq_len, dim = x.shape

        # Get routing weights and indices
        expert_weights, expert_indices, router_logits = self.router(x)

        # Initialize output tensor
        y = torch.zeros_like(x)

        # Compute load balancing loss
        # Encourages balanced expert utilization
        router_probs = router_logits.softmax(dim=-1)
        aux_loss = {
            # Z-loss: penalize large logits to improve training stability
            "router_z_loss": torch.mean(torch.square(router_logits)) * self.router_z_loss_coef,
            # Balance loss: encourage uniform expert utilization
            "balance_loss": torch.mean(
                torch.square(router_probs.mean(dim=(0, 1)) - (1.0 / self.num_experts))
            )
        }

        # For each expert, process tokens routed to it
        for expert_idx in range(self.num_experts):
            # Find which positions are routed to this expert
            # For each position and top_k slot, check if the expert index matches
            mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if not mask.any():
                # Skip if no tokens are routed to this expert
                continue

            # Get the weights for this expert from all positions that route to it
            # This is a bit complex because we need to find where the expert appears in the top_k
            expert_positions = (expert_indices == expert_idx).nonzero(as_tuple=True)
            batch_indices, seq_indices, k_indices = expert_positions

            # Extract inputs for this expert using the exact indices where this expert is chosen
            expert_inputs = x[batch_indices, seq_indices]  # [num_tokens, dim]

            # Process inputs through this expert
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Get the weights for this expert
            expert_weights_for_this_expert = expert_weights[batch_indices, seq_indices, k_indices]

            # Apply weighted outputs
            weighted_outputs = expert_outputs * expert_weights_for_this_expert.unsqueeze(-1)

            # Place the weighted outputs back in the right positions using direct indexing
            for i in range(len(batch_indices)):
                b, s = batch_indices[i], seq_indices[i]
                y[b, s] += weighted_outputs[i]

        return y, aux_loss

class MixtralTransformerBlock(nn.Module):
    """Mixtral transformer block with MoE feed-forward network."""
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int, 
        num_experts: int = 8,
        top_k: int = 2,
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
        
        # MoE with pre-norm
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MixtureOfExperts(
            dim=dim, 
            num_experts=num_experts, 
            hidden_dim=int(dim * mlp_ratio),
            top_k=top_k,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        
        # MoE with residual connection
        normalized_x = self.norm2(x)
        moe_output, aux_loss = self.moe(normalized_x)
        x = x + moe_output
        
        return x, aux_loss


@TransformerModelRegistry.register('mixtral')
class MixtralTransformer(BaseTransformer):
    """
    Mixtral-style transformer with Sparse Mixture-of-Experts for FFN layers.
    Based on the Mixtral paper: https://arxiv.org/abs/2401.04088
    """
    
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        num_experts: int = 8,
        top_k: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qk_norm: bool = True,
        aux_loss_weight: float = 0.01,
        pattern_aware: bool = False,
        num_patterns: Optional[int] = None,
        **kwargs
    ):
        super().__init__(vocab_size=vocab_size, block_size=block_size, model_type='mixtral')
        
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_experts = num_experts
        self.top_k = top_k
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.aux_loss_weight = aux_loss_weight
        self.pattern_aware = pattern_aware
        self.num_patterns = num_patterns
        
        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        if pattern_aware and num_patterns is not None:
            # Pattern-aware blocks for specialized routing
            self.blocks = nn.ModuleList([
                PatternAwareMixtralBlock(
                    dim=n_embd,
                    n_heads=n_head,
                    num_experts=num_experts,
                    num_patterns=num_patterns,
                    top_k=top_k,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    max_seq_len=block_size
                )
                for _ in range(n_layer)
            ])
            print(f"Created pattern-aware MixtralTransformer with {num_patterns} pattern types")
        else:
            # Standard blocks without pattern awareness
            self.blocks = nn.ModuleList([
                MixtralTransformerBlock(
                    dim=n_embd,
                    n_heads=n_head,
                    num_experts=num_experts,
                    top_k=top_k,
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
        
        print(f"Mixtral model with {sum(p.numel() for p in self.parameters())} parameters")
        print(f"Effective parameter count: ~{sum(p.numel() for p in self.parameters()) // num_experts * (top_k + 1) // top_k}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the model."""
        if isinstance(module, nn.Linear):
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
        targets: Optional[torch.Tensor] = None,
        pattern_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        # Get token embeddings (no added positional embeddings - using rotary in attention)
        x = self.tok_emb(idx)  # (B, T, C)
        x = self.drop(x)
        
        # Track auxiliary losses from MoE layers
        all_aux_losses = {}
        
        # Apply transformer blocks
        if self.pattern_aware and pattern_ids is not None:
            # Pattern-aware forward pass
            for i, block in enumerate(self.blocks):
                x, aux_loss = block(x, pattern_ids)

                # Add layer index to loss keys for tracking
                for k, v in aux_loss.items():
                    all_aux_losses[f"layer_{i}_{k}"] = v
        else:
            # Standard forward pass
            for i, block in enumerate(self.blocks):
                x, aux_loss = block(x)
            
            # Add layer index to loss keys for tracking
            for k, v in aux_loss.items():
                all_aux_losses[f"layer_{i}_{k}"] = v
        
        # Apply final layer norm
        x = self.norm_f(x)
        
        # Project to vocabulary
        logits = self.head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Main loss: cross entropy
            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding
            )
            
            # Add auxiliary losses (with scaling)
            aux_loss_sum = sum(all_aux_losses.values())
            loss = ce_loss + self.aux_loss_weight * aux_loss_sum
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        pattern_ids: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Ensure we don't exceed block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond, pattern_ids=pattern_ids)
            
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

    def get_expert_pattern_distribution(self) -> Dict[str, torch.Tensor]:
        """
        Get the distribution of experts across pattern types.
        Only available in pattern-aware mode.

        Returns:
            Dictionary with distribution matrices and similarity metrics
        """
        if not self.pattern_aware:
            return {"error": "Not in pattern-aware mode"}

        distributions = {}
        for i, block in enumerate(self.blocks):
            if hasattr(block.moe, "router") and hasattr(block.moe.router, "pattern_routing"):
                distributions[f"layer_{i}_distribution"] = F.softmax(block.moe.router.pattern_routing, dim=-1)
                distributions[f"layer_{i}_similarity"] = block.moe.router.compute_pattern_similarity()
        return distributions

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = super().get_config()
        config.update({
            'n_embd': self.n_embd,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'qk_norm': self.qk_norm,
            'aux_loss_weight': self.aux_loss_weight,
            'pattern_aware': self.pattern_aware,
            'num_patterns': self.num_patterns,
        })
        return config
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MixtralTransformer':
        """Create a model instance from a configuration."""
        return cls(
            vocab_size=config['vocab_size'],
            block_size=config['block_size'],
            n_embd=config.get('n_embd', 256),
            n_layer=config.get('n_layer', 6),
            n_head=config.get('n_head', 8),
            num_experts=config.get('num_experts', 8),
            top_k=config.get('top_k', 2),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1),
            qk_norm=config.get('qk_norm', True),
            aux_loss_weight=config.get('aux_loss_weight', 0.01),
            pattern_aware = config.get('pattern_aware', False),
            num_patterns = config.get('num_patterns', None)
        )
