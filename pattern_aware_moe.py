"""
Pattern-aware Mixture of Experts implementation.
This module provides MoE components that can specialize for different pattern types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

from model_neotransformer import SelfAttention


class PatternAwareMoERouter(nn.Module):
    """
    Router for Mixture of Experts layer with pattern-aware routing.
    Extends the standard MoE router to encourage specialization based on pattern types.
    """

    def __init__(self, dim: int, num_experts: int, num_patterns: int, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_patterns = num_patterns
        self.top_k = top_k

        # Content-based routing projection
        self.content_routing = nn.Linear(dim, num_experts, bias=False)

        # Pattern-based routing bias - learns which experts should handle which patterns
        self.pattern_routing = nn.Parameter(torch.zeros(num_patterns, num_experts))

        # Initialize with small values
        nn.init.normal_(self.content_routing.weight, mean=0.0, std=0.01)

        # Initialize pattern routing to encourage separation
        # We use a slightly higher std to create initial preferences
        nn.init.normal_(self.pattern_routing, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor, pattern_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the pattern-aware router.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            pattern_ids: Pattern type IDs of shape [batch_size]

        Returns:
            expert_weights: Routing weights of shape [batch_size, seq_len, top_k]
            expert_indices: Expert indices of shape [batch_size, seq_len, top_k]
            router_logits: Raw routing logits of shape [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, _ = x.shape

        # Compute content-based routing logits
        content_logits = self.content_routing(x)  # [batch_size, seq_len, num_experts]

        # Get pattern routing bias for each sample in the batch
        pattern_bias = self.pattern_routing[pattern_ids]  # [batch_size, num_experts]

        # Expand pattern bias to match content logits dimensions
        pattern_bias = pattern_bias.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, num_experts]

        # Combine content and pattern routing
        router_logits = content_logits + pattern_bias

        # Get weights and indices using top-k routing
        expert_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # [batch_size, seq_len, top_k]

        # Apply softmax over the selected experts only
        expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_weights, expert_indices, router_logits

    def compute_pattern_similarity(self) -> torch.Tensor:
        """
        Compute a similarity matrix between patterns based on their expert preferences.
        Used for visualization and analysis of expert specialization.

        Returns:
            Similarity matrix of shape [num_patterns, num_patterns]
        """
        # Normalize pattern routing weights along expert dimension
        normalized_weights = F.softmax(self.pattern_routing, dim=-1)

        # Compute cosine similarity between pattern preferences
        similarity = torch.matmul(normalized_weights, normalized_weights.transpose(0, 1))

        return similarity


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


class PatternAwareMixtureOfExperts(nn.Module):
    """
    Pattern-aware Mixture of Experts layer with top-k routing.
    Extends the standard MoE by incorporating pattern types to encourage specialization.
    """

    def __init__(
            self,
            dim: int,
            num_experts: int = 8,
            num_patterns: int = None,  # Number of different pattern types
            hidden_dim: int = None,
            top_k: int = 2,
            dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_patterns = num_patterns
        self.top_k = top_k

        # Set hidden dimension
        if hidden_dim is None:
            hidden_dim = dim * 4

        # Create pattern-aware router if num_patterns is provided
        if num_patterns is not None and num_patterns > 0:
            self.router = PatternAwareMoERouter(dim, num_experts, num_patterns, top_k)
        else:
            raise ValueError("PatternAwareMixtureOfExperts requires num_patterns to be specified")

        # Create experts
        self.experts = nn.ModuleList([
            FFExpert(dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        # For load balancing loss
        self.router_z_loss_coef = 0.001

        # For pattern specialization loss
        self.specialization_coef = 0.1

    def forward(self, x: torch.Tensor, pattern_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with pattern-aware routing and additional losses.

        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            pattern_ids: Pattern type IDs of shape [batch_size]

        Returns:
            y: Output tensor of shape [batch_size, seq_len, dim]
            aux_loss: Dictionary containing auxiliary losses
        """
        batch_size, seq_len, dim = x.shape

        # Get routing weights and indices
        expert_weights, expert_indices, router_logits = self.router(x, pattern_ids)

        # Initialize output tensor
        y = torch.zeros_like(x)

        # Compute standard MoE losses
        router_probs = router_logits.softmax(dim=-1)
        aux_loss = {
            "router_z_loss": torch.mean(torch.square(router_logits)) * self.router_z_loss_coef,
            "balance_loss": torch.mean(
                torch.square(router_probs.mean(dim=(0, 1)) - (1.0 / self.num_experts))
            )
        }

        # Add pattern specialization loss
        if self.num_patterns is not None:
            # Compute similarity between pattern routings
            pattern_similarity = self.router.compute_pattern_similarity()

            # We want the diagonal to be high (self-similarity) and off-diagonal to be low
            # Create an identity matrix as the target
            target = torch.eye(self.num_patterns, device=pattern_similarity.device)

            # Specialization loss: make patterns route to different experts
            # We multiply by (1-target) to ignore the diagonal elements
            specialization_loss = torch.mean(pattern_similarity * (1 - target))
            aux_loss["specialization_loss"] = specialization_loss * self.specialization_coef

        # For each expert, process tokens routed to it
        for expert_idx in range(self.num_experts):
            # Find which positions are routed to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]

            if not mask.any():
                # Skip if no tokens are routed to this expert
                continue

            # Get the weights for this expert from all positions that route to it
            expert_positions = (expert_indices == expert_idx).nonzero(as_tuple=True)
            batch_indices, seq_indices, k_indices = expert_positions

            # Extract inputs for this expert
            expert_inputs = x[batch_indices, seq_indices]  # [num_tokens, dim]

            # Process inputs through this expert
            expert_outputs = self.experts[expert_idx](expert_inputs)

            # Get the weights for this expert
            expert_weights_for_this_expert = expert_weights[batch_indices, seq_indices, k_indices]

            # Apply weighted outputs
            weighted_outputs = expert_outputs * expert_weights_for_this_expert.unsqueeze(-1)

            # Place the weighted outputs back in the right positions
            for i in range(len(batch_indices)):
                b, s = batch_indices[i], seq_indices[i]
                y[b, s] += weighted_outputs[i]

        return y, aux_loss


class PatternAwareMixtralBlock(nn.Module):
    """Mixtral transformer block with pattern-aware MoE feed-forward network."""

    def __init__(
            self,
            dim: int,
            n_heads: int,
            num_experts: int = 8,
            num_patterns: int = None,
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

        # Pattern-aware MoE with pre-norm
        self.norm2 = nn.LayerNorm(dim)
        self.moe = PatternAwareMixtureOfExperts(
            dim=dim,
            num_experts=num_experts,
            num_patterns=num_patterns,
            hidden_dim=int(dim * mlp_ratio),
            top_k=top_k,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, pattern_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))

        # MoE with residual connection
        normalized_x = self.norm2(x)
        moe_output, aux_loss = self.moe(normalized_x, pattern_ids)
        x = x + moe_output

        return x, aux_loss