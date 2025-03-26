import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, Type


class TransformerModelRegistry:
    """Registry of available transformer model implementations."""
    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a transformer model implementation."""
        def wrapper(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get_model_class(cls, name: str) -> Type:
        """Get model class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown model type: {name}. Available models: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def list_available_models(cls) -> list:
        """List all available model types."""
        return list(cls._registry.keys())


class BaseTransformer(nn.Module, ABC):
    """Abstract base class for all transformer models."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        model_type: str,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.model_type = model_type

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Parameters:
        -----------
        idx : torch.Tensor
            Input token indices, shape (B, T)
        targets : torch.Tensor, optional
            Target token indices, shape (B, T)

        Returns:
        --------
        tuple
            (logits, loss) where logits has shape (B, T, vocab_size)
            and loss is a scalar tensor if targets is provided
        """
        pass

    @torch.no_grad()
    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate new tokens from the model.

        Parameters:
        -----------
        idx : torch.Tensor
            Context tokens, shape (B, T)
        max_new_tokens : int
            Maximum number of tokens to generate
        temperature : float
            Temperature for sampling, lower is more conservative
        top_k : int, optional
            If set, only sample from the top k most likely tokens

        Returns:
        --------
        torch.Tensor
            Generated tokens, shape (B, T + max_new_tokens)
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        # Base configuration that all models should provide
        return {
            'model_type': self.model_type,
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
        }

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseTransformer':
        """Create a model instance from a configuration."""
        pass

    def save_pretrained(self, path: str) -> None:
        """Save model to disk with configuration."""
        config = self.get_config()
        state_dict = self.state_dict()
        torch.save({
            'config': config,
            'state_dict': state_dict
        }, path)

    @classmethod
    def load_pretrained(cls, path: str, device: torch.device = None) -> 'BaseTransformer':
        """Load model from disk."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Get appropriate model class from registry based on model_type
        model_type = config.get('model_type')
        model_cls = TransformerModelRegistry.get_model_class(model_type)
        
        # Create model instance
        model = model_cls.from_config(config)
        model.to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
