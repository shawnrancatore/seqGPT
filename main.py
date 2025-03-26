#!/usr/bin/env python3
"""
Number Sequence Transformer

This script brings together all components of the number sequence prediction system.
It provides an easy entry point to train, evaluate, and generate from the model.
"""

import os
import sys
import argparse
import torch
import logging
from typing import Dict, List, Tuple, Optional

# Import components
from pattern_generators import create_pattern_library, generate_pattern
from tokenizer import NumberStreamTokenizer
from model import get_model, list_available_models, BaseTransformer
from dataset import create_dataloaders
from train import train_epoch, evaluate, generate_samples


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def train_model(args):
    """Train the model using DataParallel for multi-GPU utilization."""
    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs")
    else:
        device = torch.device("cpu")
        n_gpus = 0
        print("Using CPU - Training may be slow.")

    # Display available models
    print(f"Available model types: {list_available_models()}")
    print(f"Using model type: {args.model_type}")

    # Create tokenizer and pattern library
    tokenizer = NumberStreamTokenizer(max_number=100)
    patterns = create_pattern_library()
    
    # Exclude all token_* patterns due to inconsistent handling of finite/infinite sequences
    # Will implement better handling in the future
    token_patterns = [pattern for pattern in patterns.keys() if pattern.startswith('token_')]
    for pattern in token_patterns:
        del patterns[pattern]
        print(f"Excluded '{pattern}' pattern from training.")
        
    # Also exclude any other known finite patterns
    if 'token_arithmetic' in patterns:  # Safety check in case it wasn't caught above
        del patterns['token_arithmetic']
        print("Excluded 'token_arithmetic' pattern (finite length) from training.")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        patterns=patterns,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size * max(1, n_gpus),  # Scale batch size by number of GPUs
        samples_per_pattern=args.samples_per_pattern,
        num_workers=args.num_workers
    )

    # Initialize variables for resuming training
    start_epoch = 0
    best_val_loss = float('inf')

    # Check if resuming training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

            # Extract model parameters from the checkpoint
            config = checkpoint.get('config', {})
            
            if not config:
                print("Warning: Old-style checkpoint format detected. Using provided model arguments.")
                config = {
                    'model_type': args.model_type,
                    'vocab_size': tokenizer.vocab_size,
                    'block_size': args.max_seq_len,
                    'n_embd': args.n_embd,
                    'n_layer': args.n_layer,
                    'n_head': args.n_head,
                    'dropout': args.dropout,
                }
            else:
                # Ensure vocabulary size matches the tokenizer
                config['vocab_size'] = tokenizer.vocab_size
                
                # Use the model type from the checkpoint unless explicitly overridden
                if args.model_type != 'auto':
                    config['model_type'] = args.model_type
                    print(f"Overriding checkpoint model type with: {args.model_type}")
                
                print(f"Resuming with model type: {config.get('model_type', 'gpt2')}")

            # Create model based on config
            model = get_model(**config).to(device)

            # Load model weights
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Handle old-style checkpoints
                model.load_state_dict(checkpoint['model_state_dict'])

            # Get other training info
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))

            print(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
        else:
            print(f"No checkpoint found at {args.resume}")
            return
    else:
        # Create a new model
        model = get_model(
            model_type=args.model_type,
            vocab_size=tokenizer.vocab_size,
            block_size=args.max_seq_len,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            dropout=args.dropout,
            # Additional model-specific parameters
            mlp_ratio=args.mlp_ratio,
            qk_norm=args.qk_norm,
            num_experts=args.num_experts,
            top_k=args.top_k_experts,
            aux_loss_weight=args.aux_loss_weight
        ).to(device)

    # Wrap model with DataParallel if multiple GPUs are available
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Model wrapped with DataParallel")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Load optimizer state if resuming
    if args.resume and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state from checkpoint")

    # Create learning rate scheduler
    if args.lr_decay:
        total_steps = len(train_loader) * args.epochs // args.grad_accumulation_steps
        warmup_steps = int(args.warmup_ratio * total_steps)

        def lr_lambda(step):
            # Linear warmup followed by cosine decay
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(
                3.14159 * (step - warmup_steps) / (total_steps - warmup_steps)
            )).item())

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accumulation_steps=args.grad_accumulation_steps
        )
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, test_loader, device)
        print(f"Validation loss: {val_loss:.4f}")

        # Save checkpoint
        if args.save_model:
            os.makedirs(args.model_dir, exist_ok=True)

            # Get model config (handles both DataParallel and regular models)
            if isinstance(model, torch.nn.DataParallel):
                config = model.module.get_config()
                model_state_dict = model.module.state_dict()
            else:
                config = model.get_config()
                model_state_dict = model.state_dict()

            # Save model with configuration
            checkpoint = {
                'epoch': epoch,
                'state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }

            torch.save(checkpoint, f"{args.model_dir}/checkpoint_epoch{epoch + 1}.pt")
            print(f"Saved checkpoint for epoch {epoch + 1}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, f"{args.model_dir}/best_model.pt")
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # Generate samples
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            # For generation, use the base model without DataParallel
            generation_model = model.module if isinstance(model, torch.nn.DataParallel) else model

            generate_samples(
                model=generation_model,
                tokenizer=tokenizer,
                device=device,
                num_samples=args.num_samples,
                max_tokens=args.max_seq_len // 2,
                temperature=args.temperature
            )

    print("Training complete!")

    # Final generation
    generation_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    generate_samples(
        model=generation_model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples * 2,
        max_tokens=args.max_seq_len,
        temperature=args.temperature
    )

    return model, tokenizer


def load_model(model_path: str, device):
    """
    Load a saved model from a checkpoint file.

    Parameters:
    -----------
    model_path : str
        Path to the model checkpoint
    device : torch.device
        Device to load the model on

    Returns:
    --------
    tuple
        (model, tokenizer)
    """
    # Create tokenizer (we need to recreate it with the same parameters)
    tokenizer = NumberStreamTokenizer(max_number=100)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    logging.info(f"Loaded checkpoint from {model_path}")

    # Extract model configuration from the checkpoint
    if 'config' in checkpoint:
        # New-style checkpoint with full config
        config = checkpoint['config']
        config['vocab_size'] = tokenizer.vocab_size  # Ensure vocab size matches tokenizer
        
        # Fix dropout if it's an object instead of a float
        if 'dropout' in config and not isinstance(config['dropout'], (int, float)):
            # Extract dropout value or use default
            try:
                if hasattr(config['dropout'], 'p'):
                    config['dropout'] = config['dropout'].p
                else:
                    config['dropout'] = 0.1  # Default value
            except:
                config['dropout'] = 0.1  # Default value
            logging.info(f"Fixed dropout parameter to: {config['dropout']}")
        
        # Log the model configuration
        logging.info(f"Model configuration:")
        for key, value in config.items():
            logging.info(f"- {key}: {value}")
        
        # Create model from config
        model = get_model(**config).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Old-style checkpoint
        logging.warning("Using legacy checkpoint format. Some features may not be available.")
        
        # Extract model parameters
        model_params = checkpoint.get('model_params', {})
        
        if not model_params:
            # Fall back to defaults if the checkpoint doesn't contain model parameters
            logging.warning("Checkpoint doesn't contain model parameters, using defaults")
            model_params = {
                'vocab_size': tokenizer.vocab_size,
                'block_size': 128,
                'n_embd': 128,
                'n_layer': 4,
                'n_head': 4,
                'embd_pdrop': 0.1,
                'attn_pdrop': 0.1,
                'resid_pdrop': 0.1
            }
        else:
            # Ensure vocabulary size matches the tokenizer
            model_params['vocab_size'] = tokenizer.vocab_size
        
        # Create model with legacy parameters (default to GPT2 for backwards compatibility)
        model_params['model_type'] = 'gpt2'
        model = get_model(**model_params).to(device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])

    logging.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")

    return model, tokenizer


def generate_from_model(args):
    """Generate sequences from a saved model."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.model_path, device)
    logging.info(f"Loaded model from {args.model_path}")
    
    if hasattr(model, 'model_type'):
        logging.info(f"Model type: {model.model_type}")

    # Generate samples
    logging.info(f"Generating {args.num_samples} samples:")
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Number Sequence Transformer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Get available model types
    available_models = list_available_models()

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")

    # Model type
    train_parser.add_argument("--model_type", type=str, default='gpt2', choices=available_models,
                              help=f"Type of transformer model to use. Available: {available_models}")

    # Model parameters
    train_parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    train_parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    train_parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Additional model parameters for different architectures
    train_parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP expansion ratio")
    train_parser.add_argument("--qk_norm", action="store_true", help="Use QK normalization")
    train_parser.add_argument("--num_experts", type=int, default=8, help="Number of experts (for MoE models)")
    train_parser.add_argument("--top_k_experts", type=int, default=2, help="Number of experts to route to")
    train_parser.add_argument("--aux_loss_weight", type=float, default=0.01, help="Weight for auxiliary losses")

    # Training parameters
    train_parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    train_parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--lr_decay", action="store_true", help="Use learning rate decay")
    train_parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    # Data parameters
    train_parser.add_argument("--samples_per_pattern", type=int, default=500, help="Samples per pattern")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")

    # Other parameters
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    train_parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    train_parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")
    train_parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    train_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    train_parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    train_parser.add_argument("--log_file", type=str, default=None, help="Log file path")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate from a trained model")
    gen_parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    gen_parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    gen_parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    gen_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    gen_parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    gen_parser.add_argument("--log_file", type=str, default=None, help="Log file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file if hasattr(args, 'log_file') else None)

    if args.command == "train":
        train_model(args)
    elif args.command == "generate":
        generate_from_model(args)
    else:
        parser.print_help()
        
        # Add note about using the dedicated complexity test module
        if not args.command:
            print("\nNote: For complexity testing, use the dedicated module:\n"
                  "  python complexity_test.py run [options]\n"
                  "  python complexity_test.py create_config --output my_config.json\n")


if __name__ == "__main__":
    main()
