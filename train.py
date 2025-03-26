import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import argparse
import logging
from typing import Dict, List, Tuple, Union, Optional

from pattern_generators import generate_pattern, create_pattern_library
from tokenizer import NumberStreamTokenizer
from model import NumberSequenceTransformer
from dataset import create_dataloaders


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        grad_accumulation_steps: int = 1
) -> float:
    """
    Train for one epoch.

    Parameters:
    -----------
    model : nn.Module
        The model to train
    dataloader : DataLoader
        Dataloader for training data
    optimizer : torch.optim.Optimizer
        The optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler
    device : torch.device
        Device to train on
    grad_accumulation_steps : int
        Number of steps to accumulate gradients

    Returns:
    --------
    float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_samples = 0

    # Initialize accumulated gradients
    optimizer.zero_grad()

    for step, (x, y, _) in enumerate(dataloader):
        # Move to device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        logits, loss = model(x, y)

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

        # Log progress (with DEBUG level so it can be filtered out)
        if step % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            logging.debug(f"Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {lr:.6f}")

    # Handle any remaining accumulated gradients
    if (step + 1) % grad_accumulation_steps != 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / total_samples


def evaluate(model, dataloader, device, tokenizer=None):
    """
    Evaluate the model on a dataset.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : torch.utils.data.DataLoader
        The dataloader for the evaluation dataset
    device : torch.device
        The device to run evaluation on
    tokenizer : NumberStreamTokenizer, optional
        Tokenizer for reconstructing sequences

    Returns:
    --------
    float
        Average loss on the dataset
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y, pattern_names in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, loss = model(x, y)

            # Handle DataParallel case - loss might be a tensor with iple values
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            # Accumulate loss
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # If tokenizer is provided, generate predictions for a sample
            if tokenizer and total_samples <= 5:  # Only for the first few batches
                # Get a sample from the batch
                sample_idx = 0
                sample_x = x[sample_idx].unsqueeze(0)
                sample_y = y[sample_idx]
                pattern_name = pattern_names[sample_idx]
                
                # Skip token patterns during evaluation examples
                if pattern_name.startswith('token_'):
                    continue

                # Generate predictions
                max_new_tokens = 30  # Generate 30 new tokens
                generated = model.generate(sample_x, max_new_tokens=max_new_tokens, temperature=0.8)

                # Convert to tokens
                context_tokens = tokenizer.denumericalize(sample_x[0].tolist())
                expected_tokens = tokenizer.denumericalize(sample_y.tolist())
                generated_tokens = tokenizer.denumericalize(generated[0, sample_x.size(1):].tolist())

                # Process tokens to get sequences
                # Reconstruct context sequence
                context_sequence = []
                current_digits = []
                for token in context_tokens:
                    if token.startswith('NUM_'):
                        if current_digits:
                            num_str = ''.join(current_digits)
                            context_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []
                        num = int(token.split('_')[1])
                        context_sequence.append(num)
                    elif token in tokenizer.digits or token == '.' or token == '-':
                        current_digits.append(token)
                    elif token == ',':
                        if current_digits:
                            num_str = ''.join(current_digits)
                            context_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []

                # Reconstruct expected sequence
                expected_sequence = []
                current_digits = []
                for token in expected_tokens:
                    if token.startswith('NUM_'):
                        if current_digits:
                            num_str = ''.join(current_digits)
                            expected_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []
                        num = int(token.split('_')[1])
                        expected_sequence.append(num)
                    elif token in tokenizer.digits or token == '.' or token == '-':
                        current_digits.append(token)
                    elif token == ',':
                        if current_digits:
                            num_str = ''.join(current_digits)
                            expected_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []

                # Reconstruct generated sequence
                generated_sequence = []
                current_digits = []
                has_partial = False # Flag to indicate partial number at the end
                partial_num = ""

                for token in generated_tokens:
                    if token.startswith('NUM_'):
                        if current_digits:
                            num_str = ''.join(current_digits)
                            generated_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []
                        num = int(token.split('_')[1])
                        generated_sequence.append(num)
                    elif token in tokenizer.digits or token == '.' or token == '-':
                        current_digits.append(token)
                    elif token == ',':
                        if current_digits:
                            num_str = ''.join(current_digits)
                            generated_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                            current_digits = []

                # Check if there are remaining digits (partial number)
                if current_digits:
                    has_partial = True
                    partial_num = ''.join(current_digits)

                # Print the comparison
                all_patterns = create_pattern_library()
                pattern_rule = all_patterns.get(pattern_name, "")
                print(f"Sample {total_samples}: Pattern '{pattern_name}' ({pattern_rule})")
                print(f"Starting context: {' '.join(str(n) for n in context_sequence[:10])}")
                print(f"Generated:      {' '.join(str(n) for n in generated_sequence[:30])}")
                print(f"Expected:       {' '.join(str(n) for n in expected_sequence[:100])}")

                # Calculate accuracy on complete numbers only
                min_len = min(len(generated_sequence), len(expected_sequence))
                correct = sum(1 for i in range(min_len) if generated_sequence[i] == expected_sequence[i])
                accuracy = correct / min_len * 100 if min_len > 0 else 0

                # Check for partial match
                if has_partial and min_len < len(expected_sequence):
                    next_expected = str(expected_sequence[min_len])
                    if next_expected.startswith(partial_num):
                        print(f"Partial match detected: '{partial_num}' is prefix of next expected number '{next_expected}'")
                        # If all complete numbers matched and partial matches, award 100%
                        if correct == min_len:
                            accuracy = 100.0

                print(f"Accuracy: {accuracy:.2f}%")

                # Only add partial number note if there actually is a partial number
                if has_partial:
                    print(f"Note: Partial number '{partial_num}' at end of generation")

                print()

    return total_loss / total_samples


def generate_samples(
        model,
        tokenizer,
        device,
        num_samples=5,
        max_tokens=50,
        temperature=0.8,
        top_k=50,
        context_length=10  # Length of starting context to show
):
    """
    Generate samples from the model with proper evaluation of accuracy.

    Parameters:
    -----------
    model : NumberSequenceTransformer
        The model to generate from
    tokenizer : NumberStreamTokenizer
        Tokenizer for converting between tokens and numbers
    device : torch.device
        Device to run generation on
    num_samples : int
        Number of samples to generate
    max_tokens : int
        Maximum number of tokens to generate
    temperature : float
        Temperature for sampling
    top_k : int
        Top-k sampling parameter
    context_length : int
        Length of starting context to show
    """
    # ANSI color codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Unwrap model if it's using DataParallel
    if isinstance(model, torch.nn.DataParallel):
        generation_model = model.module
    else:
        generation_model = model

    model.eval()

    # Get a batch from test set (or create synthetic prompts)
    all_patterns = create_pattern_library()
    
    # Exclude token patterns from evaluation
    patterns = {}
    for pattern_name, pattern_rule in all_patterns.items():
        if not pattern_name.startswith('token_'):
            patterns[pattern_name] = pattern_rule

    for i in range(num_samples):
        # Choose a random pattern
        pattern_name = random.choice(list(patterns.keys()))
        pattern_rule = patterns[pattern_name]

        # Generate a sequence from the pattern
        full_sequence = generate_pattern(pattern_rule, length=max_tokens + context_length)

        # Get the context (first part of the sequence)
        context_sequence = full_sequence[:context_length]
        expected_sequence = full_sequence[context_length:context_length + max_tokens]

        # Tokenize the context
        context_tokens = tokenizer.tokenize_sequence(context_sequence)
        context_ids = tokenizer.numericalize(context_tokens)

        # Convert to tensor and add batch dimension
        context_tensor = torch.tensor([context_ids], dtype=torch.long).to(device)

        # Generate from the model
        with torch.no_grad():
            generated_ids = generation_model.generate(
                context_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

        # Get only the newly generated part (exclude the context)
        new_ids = generated_ids[0, context_tensor.size(1):].tolist()

        # Convert back to tokens and then to numbers
        generated_tokens = tokenizer.denumericalize(new_ids)

        # Analyze the tokens to reconstruct the sequence
        generated_sequence = []
        current_digits = []
        has_partial = False # Flag for tracking partial numbers at the end
        partial_num = ""

        for token in generated_tokens:
            if token.startswith('NUM_'):
                if current_digits:
                    # Process any accumulated digits before handling the NUM token
                    num_str = ''.join(current_digits)
                    generated_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                    current_digits = []

                # Extract number from the NUM token
                num = int(token.split('_')[1])
                generated_sequence.append(num)

            elif token in tokenizer.digits or token == '.' or token == '-':
                current_digits.append(token)

            elif token == ',':
                if current_digits:
                    num_str = ''.join(current_digits)
                    generated_sequence.append(int(num_str) if num_str.isdigit() else float(num_str))
                    current_digits = []

        # Check if there are remaining digits (partial number)
        if current_digits:
            has_partial = True
            partial_num = ''.join(current_digits)

        # Print the results with clear formatting
        print(f"\n{BOLD}Sample {i + 1}: Pattern '{pattern_name}' ({pattern_rule}){RESET}")

        # Print context (starting sequence)
        context_str = ' '.join(str(n) for n in context_sequence)
        print(f"{BOLD}Starting context:{RESET} {BLUE}{context_str}{RESET}")

        # Print generated sequence - show all numbers, not truncated
        generated_str = ' '.join(str(n) for n in generated_sequence)
        print(f"{BOLD}Generated:{RESET}      {GREEN}{generated_str}{RESET}")

        # Print expected sequence - show all numbers, not truncated
        expected_str = ' '.join(str(n) for n in expected_sequence)
        print(f"{BOLD}Expected:{RESET}       {RED}{expected_str}{RESET}")

        # Calculate accuracy on complete numbers only
        min_len = min(len(generated_sequence), len(expected_sequence))
        correct = sum(1 for i in range(min_len) if generated_sequence[i] == expected_sequence[i])
        accuracy = correct / min_len * 100 if min_len > 0 else 0

        # Check for partial match
        if has_partial and min_len < len(expected_sequence):
            next_expected = str(expected_sequence[min_len])
            if next_expected.startswith(partial_num):
                print(f"Partial match detected: '{partial_num}' is prefix of next expected number '{next_expected}'")
                # If all complete numbers matched and partial matches, award 100%
                if correct == min_len:
                    accuracy = 100.0

        print(f"{BOLD}Accuracy:{RESET} {accuracy:.2f}%")

        # Only add partial number note if there actually is a partial number
        if has_partial:
            print(f"{BOLD}Note:{RESET} Partial number '{partial_num}' at end of generation")

        print("-" * 80)


def train_for_complexity_test(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    target_loss: float,
    max_epochs: int,
    grad_accumulation_steps: int = 1
) -> Tuple[int, float]:
    """
    Train model until target loss is reached or max epochs are completed, for complexity testing.

    Parameters:
    -----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader for training data.
    test_loader : DataLoader
        DataLoader for test data (validation).
    optimizer : torch.optim.Optimizer
        Optimizer.
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler.
    device : torch.device
        Device to train on.
    target_loss : float
        Target validation loss to achieve.
    max_epochs : int
        Maximum number of epochs to train.
    grad_accumulation_steps : int
        Number of gradient accumulation steps.

    Returns:
    --------
    Tuple[int, float]
        Number of epochs trained and the final validation loss.
    """
    best_val_loss = float('inf')
    epochs_trained = 0

    for epoch in range(max_epochs):
        epochs_trained = epoch + 1 # Correctly track epochs trained
        logging.debug(f"Complexity Test - Epoch {epoch + 1}/{max_epochs}")

        # Train epoch
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accumulation_steps=grad_accumulation_steps
        )
        logging.debug(f"Complexity Test - Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, test_loader, device)
        # Keep validation loss at INFO level since it's important progress information
        logging.info(f"Complexity Test - Validation loss: {val_loss:.4f}")

        if val_loss <= target_loss:
            logging.info(f"Target validation loss of {target_loss} reached in {epoch + 1} epochs.")
            return epochs_trained, val_loss # Return epochs trained and loss

        # Basic early stopping in case of overfitting (optional, for robustness)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        elif val_loss > best_val_loss * 1.1: # If val_loss increases significantly
            logging.info(f"Validation loss increased significantly, early stopping epoch {epoch + 1}.")
            return epochs_trained, val_loss # Return epochs trained and loss

    logging.info(f"Max epochs reached, target validation loss not reached. Final Validation Loss: {val_loss:.4f}")
    return epochs_trained, val_loss # Return epochs trained and final loss



def main(args):
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create pattern library
    patterns = create_pattern_library()
    logging.info(f"Created pattern library with {len(patterns)} patterns")

    # Create tokenizer
    tokenizer = NumberStreamTokenizer(max_number=100)
    logging.info(f"Created tokenizer with vocab size: {tokenizer.vocab_size}")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        patterns=patterns,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        samples_per_pattern=args.samples_per_pattern,
        num_workers=args.num_workers
    )
    logging.info(f"Created dataloaders with {len(train_loader)} training batches and {len(test_loader)} test batches")

    # Create model
    model = NumberSequenceTransformer(
        vocab_size=tokenizer.vocab_size,
        block_size=args.max_seq_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        resid_pdrop=args.dropout
    ).to(device)
    logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

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

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accumulation_steps=args.grad_accumulation_steps
        )
        logging.info(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_loss = evaluate(model, test_loader, device, tokenizer)
        logging.info(f"Validation loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_model:
                os.makedirs('models', exist_ok=True)

                # Save the model configuration along with the state
                model_config = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'model_params': {
                        'vocab_size': tokenizer.vocab_size,
                        'block_size': args.max_seq_len,
                        'n_embd': args.n_embd,
                        'n_layer': args.n_layer,
                        'n_head': args.n_head,
                        'embd_pdrop': args.dropout,
                        'attn_pdrop': args.dropout,
                        'resid_pdrop': args.dropout
                    }
                }

                torch.save(model_config, f"models/number_sequence_transformer_epoch{epoch + 1}.pt")
                logging.info(f"Saved model checkpoint with configuration at epoch {epoch + 1}")

        # Generate samples
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            logging.info("Generating samples:")
            generate_samples(
                model=model,
                tokenizer=tokenizer,
                device=device,
                num_samples=3,
                max_tokens=args.max_seq_len,
                temperature=0.8
            )

    logging.info("Training complete!")

    # Final generation
    logging.info("Generating final samples:")
    generate_samples(
        model=model,
        tokenizer=tokenizer,
        device=device,
        num_samples=5,
        max_tokens=args.max_seq_len,
        temperature=0.8
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer for number sequence prediction")

    # Model parameters
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

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

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_model", action="store_true", help="Save model checkpoints")
    parser.add_argument("--sample_every", type=int, default=5, help="Generate samples every N epochs")

    args = parser.parse_args()

    main(args)