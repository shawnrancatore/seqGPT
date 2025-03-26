#!/usr/bin/env python3
"""
Train a pattern-aware Mixtral model.
This script provides specialized training for pattern-aware MoE models
without modifying the core training logic.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from pattern_generators import create_pattern_library
from tokenizer import NumberStreamTokenizer
from dataset import NumberSequenceDataset
from model import get_model
from pattern_extractors import create_pattern_aware_dataloaders, PatternStatistics
from train import evaluate  # Reuse standard evaluation for comparison


def train_pattern_aware_epoch(
        model,
        dataloader,
        optimizer,
        scheduler,
        device,
        grad_accumulation_steps=1
):
    """
    Train pattern-aware model for one epoch.
    """
    model.train()
    total_loss = 0
    total_samples = 0

    # Initialize accumulated gradients
    optimizer.zero_grad()

    for step, (x, y, pattern_names, pattern_ids) in enumerate(dataloader):
        # Move to device
        x = x.to(device)
        y = y.to(device)
        if pattern_ids is not None:
            pattern_ids = pattern_ids.to(device)

        # Forward pass with pattern IDs
        logits, loss = model(x, y, pattern_ids)
        
        # When using DataParallel, the loss may be a tensor with values from each GPU
        # We need to reduce it to a scalar manually
        if isinstance(model, torch.nn.DataParallel) and loss.dim() > 0:
            loss = loss.mean()  # Average the losses from all GPUs

        # Scale loss by accumulation steps
        loss = loss / grad_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights if we've accumulated enough gradients
        if (step + 1) % grad_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Reset gradients
            optimizer.zero_grad()

        # Track statistics
        total_loss += loss.item() * grad_accumulation_steps * x.size(0)
        total_samples += x.size(0)

        # Log progress
        if step % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            logging.info(f"Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

    # Handle any remaining accumulated gradients
    if (step + 1) % grad_accumulation_steps != 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_samples


def evaluate_pattern_aware(
        model,
        dataloader,
        device,
        tokenizer=None,
        pattern_mapping=None
):
    """
    Evaluate pattern-aware model and track per-pattern performance.
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    # Track pattern-specific metrics
    pattern_stats = PatternStatistics(pattern_mapping)

    with torch.no_grad():
        for x, y, pattern_names, pattern_ids in dataloader:
            x, y = x.to(device), y.to(device)
            if pattern_ids is not None:
                pattern_ids = pattern_ids.to(device)

            # Forward pass with pattern IDs
            logits, loss = model(x, y, pattern_ids)
            
            # Handle DataParallel case
            if isinstance(model, torch.nn.DataParallel) and loss.dim() > 0:
                loss = loss.mean()  # Average the losses from all GPUs

            # Accumulate loss
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Track pattern-specific losses
            if pattern_ids is not None:
                # Calculate per-sample losses
                sample_losses = []
                for i in range(len(x)):
                    sample_logits = logits[i:i + 1]
                    sample_targets = y[i:i + 1]
                    # Get vocab_size from model.module if using DataParallel
                    vocab_size = model.module.vocab_size if isinstance(model, torch.nn.DataParallel) else model.vocab_size
                    sample_loss = nn.functional.cross_entropy(
                        sample_logits.view(-1, vocab_size),
                        sample_targets.view(-1),
                        ignore_index=-1,
                        reduction='mean'
                    )
                    sample_losses.append(sample_loss)

                # Update pattern statistics
                pattern_stats.update(pattern_ids.cpu().numpy(),
                                     [l.item() for l in sample_losses])

            # Generate predictions for a few samples
            if tokenizer and total_samples <= 15:  # Only for the first few batches
                sample_idx = 0
                sample_x = x[sample_idx].unsqueeze(0)
                sample_y = y[sample_idx]
                sample_pattern_id = None
                if pattern_ids is not None:
                    sample_pattern_id = pattern_ids[sample_idx].unsqueeze(0)
                pattern_name = pattern_names[sample_idx] if pattern_names[0] is not None else "unknown"

                # Generate predictions
                max_new_tokens = 20
                # For generation, use the base model without DataParallel
                generation_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                generated = generation_model.generate(
                    sample_x,
                    max_new_tokens=max_new_tokens,
                    pattern_ids=sample_pattern_id,
                    temperature=0.8
                )

                # Convert to tokens and sequences for display
                context_tokens = tokenizer.denumericalize(sample_x[0].tolist())
                expected_tokens = tokenizer.denumericalize(sample_y.tolist())
                generated_tokens = tokenizer.denumericalize(generated[0, sample_x.size(1):].tolist())

                # Reconstruct sequences
                context_sequence = tokenizer.reconstruct_sequence(context_tokens)
                expected_sequence = tokenizer.reconstruct_sequence(expected_tokens)
                generated_sequence = tokenizer.reconstruct_sequence(generated_tokens)

                # Print comparison
                pattern_id_str = f"(ID: {sample_pattern_id.item()})" if sample_pattern_id is not None else ""
                print(f"\nSample {total_samples}: Pattern '{pattern_name}' {pattern_id_str}")
                print(f"Starting context: {' '.join(str(n) for n in context_sequence[:10])}")
                print(f"Generated:      {' '.join(str(n) for n in generated_sequence[:20])}")
                print(f"Expected:       {' '.join(str(n) for n in expected_sequence[:20])}")

                # Calculate accuracy
                min_len = min(len(generated_sequence), len(expected_sequence))
                if min_len > 0:
                    correct = sum(1 for i in range(min_len) if generated_sequence[i] == expected_sequence[i])
                    accuracy = correct / min_len * 100
                    print(f"Accuracy: {accuracy:.2f}%")

    # Print pattern-specific metrics
    pattern_stats.print_summary()

    return total_loss / total_samples


def analyze_expert_specialization(
        model,
        pattern_mapping,
        output_dir='expert_analysis'
):
    """
    Analyze and visualize expert specialization across pattern types.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if model has the pattern-aware method
    if not hasattr(model, 'get_expert_pattern_distribution'):
        logging.warning("Model does not support expert pattern distribution analysis")
        return

    # Get expert-pattern distributions from the model
    distributions = model.get_expert_pattern_distribution()

    # Plot the distributions for each layer
    for key, distribution in distributions.items():
        if 'distribution' in key:
            layer_num = key.split('_')[1]

            # Create a heatmap of expert-pattern distribution
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(
                distribution.detach().cpu().numpy(),  # Detach tensor to remove gradients
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=[f"Expert {i}" for i in range(distribution.size(1))],
                yticklabels=[pattern_mapping.get(i, f"Pattern {i}") for i in range(distribution.size(0))]
            )
            plt.title(f"Layer {layer_num}: Expert-Pattern Assignment Distribution")
            plt.ylabel("Pattern Type")
            plt.xlabel("Expert")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/layer_{layer_num}_distribution.png")
            plt.close()

        elif 'similarity' in key:
            layer_num = key.split('_')[1]

            # Create a heatmap of pattern similarity
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                distribution.detach().cpu().numpy(),  # Detach tensor to remove gradients
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                xticklabels=[pattern_mapping.get(i, f"Pattern {i}") for i in range(distribution.size(1))],
                yticklabels=[pattern_mapping.get(i, f"Pattern {i}") for i in range(distribution.size(0))]
            )
            plt.title(f"Layer {layer_num}: Pattern Similarity Matrix")
            plt.ylabel("Pattern Type")
            plt.xlabel("Pattern Type")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/layer_{layer_num}_similarity.png")
            plt.close()

    logging.info(f"Expert specialization analysis saved to {output_dir}/")


def main(args):
    """Main function for pattern-aware training."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"pattern_mixtral_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        logging.info(f"Using {n_gpus} GPUs")
    else:
        device = torch.device("cpu")
        n_gpus = 0
        logging.info("Using CPU - Training may be slow.")

    # Create pattern library
    patterns = create_pattern_library()

    # Always filter out token patterns for consistency with other training
    # Exclude all token_* patterns due to inconsistent handling of finite/infinite sequences
    token_patterns = [pattern for pattern in patterns.keys() if pattern.startswith('token_')]
    for pattern in token_patterns:
        del patterns[pattern]
        logging.info(f"Excluded '{pattern}' pattern from training.")
        
    # Also exclude any other known finite patterns
    if 'token_arithmetic' in patterns:  # Safety check in case it wasn't caught above
        del patterns['token_arithmetic']
        logging.info("Excluded 'token_arithmetic' pattern (finite length) from training.")

    logging.info(f"Using pattern library with {len(patterns)} patterns")

    # Create tokenizer
    tokenizer = NumberStreamTokenizer(max_number=100)
    logging.info(f"Created tokenizer with vocab size: {tokenizer.vocab_size}")

    # Create standard datasets first (using existing code)
    from dataset import NumberSequenceDataset

    train_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='train',
        max_seq_len=args.max_seq_len,
        samples_per_pattern=args.samples_per_pattern
    )

    test_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='test',
        max_seq_len=args.max_seq_len,
        samples_per_pattern=args.samples_per_pattern
    )

    # Now wrap these with our pattern-aware dataloaders
    from pattern_extractors import create_pattern_aware_dataloaders

    train_loader, test_loader, pattern_mapping, num_patterns = create_pattern_aware_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size * max(1, n_gpus),  # Scale batch size by number of GPUs
        num_workers=args.num_workers,
        pin_memory=True
    )

    logging.info(f"Created dataloaders with {len(train_loader)} training batches, {len(test_loader)} test batches")
    logging.info(f"Number of pattern types: {num_patterns}")

    # Initialize variables for resuming training
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Check if resuming training from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            # Create model from the saved configuration
            config = checkpoint['config']
            config['vocab_size'] = tokenizer.vocab_size  # Ensure vocab size matches tokenizer
            config['num_patterns'] = num_patterns  # Ensure pattern count matches
            config['pattern_aware'] = True  # Ensure pattern awareness is enabled
            
            # Create the model with the saved config
            model = get_model(**config).to(device)
            
            # Load the model state
            model.load_state_dict(checkpoint['state_dict'])
            logging.info(f"Loaded model state from checkpoint")
            
            # Wrap model with DataParallel if multiple GPUs are available
            if n_gpus > 1:
                model = torch.nn.DataParallel(model)
                logging.info(f"Model wrapped with DataParallel")
            
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info(f"Loaded optimizer state from checkpoint")
            
            # Set the starting epoch
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            logging.info(f"Resuming from epoch {start_epoch} with validation loss {best_val_loss:.4f}")
        else:
            logging.warning(f"No checkpoint found at {args.resume}, starting from scratch")
            # Create new model since resume failed
            model = get_model(
                model_type="mixtral",
                vocab_size=tokenizer.vocab_size,
                block_size=args.max_seq_len,
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                num_experts=args.num_experts,
                num_patterns=num_patterns,
                top_k=args.top_k_experts,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout,
                qk_norm=args.qk_norm,
                aux_loss_weight=args.aux_loss_weight,
                pattern_aware=True  # Enable pattern awareness
            ).to(device)
            
            # Wrap model with DataParallel if multiple GPUs are available
            if n_gpus > 1:
                model = torch.nn.DataParallel(model)
                logging.info(f"Model wrapped with DataParallel")
                
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Create pattern-aware Mixtral model from scratch
        model = get_model(
            model_type="mixtral",
            vocab_size=tokenizer.vocab_size,
            block_size=args.max_seq_len,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            num_experts=args.num_experts,
            num_patterns=num_patterns,
            top_k=args.top_k_experts,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            qk_norm=args.qk_norm,
            aux_loss_weight=args.aux_loss_weight,
            pattern_aware=True  # Enable pattern awareness
        ).to(device)
        logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Wrap model with DataParallel if multiple GPUs are available
        if n_gpus > 1:
            model = torch.nn.DataParallel(model)
            logging.info(f"Model wrapped with DataParallel")

        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create learning rate scheduler
    if args.lr_decay:
        total_steps = len(train_loader) * args.epochs // args.grad_accumulation_steps
        warmup_steps = int(args.warmup_ratio * total_steps)

        def lr_lambda(step):
            # Linear warmup followed by cosine decay
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Training loop (start from the appropriate epoch if resuming)
    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_pattern_aware_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accumulation_steps=args.grad_accumulation_steps
        )
        logging.info(f"Train loss: {train_loss:.4f}")

        # Evaluate with pattern awareness
        val_loss = evaluate_pattern_aware(
            model=model,
            dataloader=test_loader,
            device=device,
            tokenizer=tokenizer,
            pattern_mapping=pattern_mapping
        )
        logging.info(f"Validation loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_model:
                os.makedirs('models', exist_ok=True)
                model_path = f"models/pattern_mixtral_epoch{epoch + 1}.pt"

                # Get model state dict and config properly (handling DataParallel)
                if isinstance(model, torch.nn.DataParallel):
                    model_state_dict = model.module.state_dict()
                    config = model.module.get_config()
                else:
                    model_state_dict = model.state_dict()
                    config = model.get_config()
                
                # Save model with configuration
                torch.save({
                    'epoch': epoch,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                    'pattern_mapping': pattern_mapping
                }, model_path)

                logging.info(f"Saved model checkpoint at {model_path}")

        # Analyze expert specialization periodically
        if (epoch + 1) % args.analyze_every == 0 or epoch == args.epochs - 1:
            logging.info("Analyzing expert specialization...")
            # Use model.module if DataParallel is used
            analyze_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            analyze_expert_specialization(
                model=analyze_model,
                pattern_mapping=pattern_mapping,
                output_dir=f"expert_analysis_epoch{epoch + 1}"
            )

    logging.info("Training complete!")

    # Final expert specialization analysis
    logging.info("Performing final expert specialization analysis...")
    # Use model.module if DataParallel is used
    analyze_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    analyze_expert_specialization(
        model=analyze_model,
        pattern_mapping=pattern_mapping,
        output_dir="expert_analysis_final"
    )

    return model, tokenizer, pattern_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pattern-aware Mixtral model")

    # Model parameters
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts in the MoE layer")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Number of experts to route to per token")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--qk_norm", action="store_true", help="Use QK normalization")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01, help="Weight for auxiliary losses")

    # Training parameters
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_decay", action="store_true", help="Use learning rate decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for LR scheduler")

    # Data parameters
    parser.add_argument("--samples_per_pattern", type=int, default=500, help="Samples per pattern")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    # Token patterns are now excluded by default for consistency

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--analyze_every", type=int, default=5, help="Analyze expert specialization every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")

    args = parser.parse_args()

    main(args)