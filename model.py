"""
Number Sequence Transformer Models

This file provides a unified import point for all transformer models.
"""

from model_base import BaseTransformer, TransformerModelRegistry
from model_gpt import GPT2Transformer
from model_neotransformer import NeoTransformer
from model_mixtral import MixtralTransformer
from model_llama import Llama3Transformer
from model_llama2 import Llama2Transformer

# For backward compatibility
NumberSequenceTransformer = GPT2Transformer

def get_model(
    model_type: str,
    vocab_size: int,
    block_size: int,
    **kwargs
) -> BaseTransformer:
    """
    Factory function to create a model of the specified type.
    
    Args:
        model_type: Type of model to create ('gpt2', 'neotransformer', 'mixtral', 'llama3', etc.)
        vocab_size: Size of the vocabulary
        block_size: Maximum sequence length
        **kwargs: Additional arguments for the specific model type
    
    Returns:
        A transformer model instance
    """
    model_class = TransformerModelRegistry.get_model_class(model_type)
    return model_class(vocab_size=vocab_size, block_size=block_size, **kwargs)

def list_available_models() -> list:
    """List all available model architectures."""
    return TransformerModelRegistry.list_available_models()

def test_model():
    """Test the model with random input."""
    vocab_size = 100
    block_size = 32
    batch_size = 4

    # Test each available model type
    for model_type in list_available_models():
        print(f"\nTesting {model_type} model:")
        
        # Create model
        model = get_model(
            model_type=model_type,
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=128,
            n_layer=4,
            n_head=4
        )

        # Create random input
        import torch
        idx = torch.randint(0, vocab_size, (batch_size, block_size))
        targets = torch.randint(0, vocab_size, (batch_size, block_size))

        # Forward pass
        logits, loss = model(idx, targets)
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item()}")

        # Test generation
        start_tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        generated = model.generate(start_tokens, max_new_tokens=20, temperature=0.8, top_k=50)
        print(f"Generated shape: {generated.shape}")
        print(f"Generated tokens: {generated[0].tolist()}")


if __name__ == "__main__":
    test_model()
