#!/usr/bin/env python3
"""
Complexity Test Module for SeqGPT

This module provides functionality to test the complexity of different patterns
by evaluating how much compute is needed to learn them. It helps to quantify the
complexity of numerical patterns by testing different model sizes until a target
accuracy is reached.
"""

import os
import sys
import logging
import argparse
import json
from typing import Dict, List, Tuple, Optional
import torch

# Import components
from pattern_generators import create_pattern_library, generate_pattern
from tokenizer import NumberStreamTokenizer
from model import get_model, list_available_models
from dataset import create_dataloaders
from train import train_for_complexity_test, evaluate


def setup_logging(log_file: Optional[str] = None, verbosity: str = "INFO"):
    """Setup logging configuration."""
    # Set log level based on verbosity
    log_level = getattr(logging, verbosity.upper())
    
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def load_patterns_config(config_file: str) -> Dict:
    """
    Load patterns configuration from a JSON file.
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration file
        
    Returns:
    --------
    Dict
        Dictionary containing pattern configurations
    """
    if not os.path.exists(config_file):
        # Return default if config doesn't exist
        logging.warning(f"Config file {config_file} not found. Using default patterns.")
        return {}
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError:
        logging.error(f"Error parsing config file {config_file}. Using default patterns.")
        return {}
    except Exception as e:
        logging.error(f"Error loading config file {config_file}: {str(e)}. Using default patterns.")
        return {}


def evaluate_silently(model, dataloader, device):
    """
    Evaluate the model on a dataset and return only the loss.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate
    dataloader : torch.utils.data.DataLoader
        The dataloader for the evaluation dataset
    device : torch.device
        The device to run evaluation on
        
    Returns:
    --------
    float
        Average loss on the dataset
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            
            # Handle DataParallel case
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
                
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples


def run_complexity_test(args):
    """
    Evaluate the complexity of different patterns by training models of varying sizes.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Tokenizer and patterns
    tokenizer = NumberStreamTokenizer(max_number=100)
    
    # Load patterns configuration if specified
    patterns_config = {}
    if args.patterns_config:
        patterns_config = load_patterns_config(args.patterns_config)
    
    # Determine which patterns to test
    full_pattern_library = create_pattern_library()
    
    if args.pattern_rule:
        # Custom pattern from command line takes precedence
        patterns = {"custom_pattern": args.pattern_rule}
        pattern_names_to_test = ["custom_pattern"]
    elif args.pattern_type:
        # Specific pattern type from command line
        if args.pattern_type not in full_pattern_library:
            print(f"Error: Pattern type '{args.pattern_type}' not found in library.")
            return
        patterns = {args.pattern_type: full_pattern_library[args.pattern_type]}
        pattern_names_to_test = [args.pattern_type]
    elif patterns_config.get('patterns'):
        # Patterns from config file
        selected_patterns = {}
        for pattern_name in patterns_config.get('patterns'):
            if pattern_name in full_pattern_library:
                selected_patterns[pattern_name] = full_pattern_library[pattern_name]
            else:
                logging.warning(f"Pattern '{pattern_name}' from config not found in library. Skipping.")
        
        if not selected_patterns:
            print("Error: No valid patterns found in configuration file.")
            return
            
        patterns = selected_patterns
        pattern_names_to_test = sorted(list(patterns.keys()))
    else:
        # Default: test all patterns
        patterns = full_pattern_library
        pattern_names_to_test = sorted(list(patterns.keys()))

    # Get model configurations
    model_configurations = args.model_sizes
    
    # If config file has model sizes, use those
    if patterns_config.get('model_sizes'):
        model_configurations = patterns_config.get('model_sizes')
    
    # Target loss and max epochs
    target_loss = args.target_loss
    max_epochs = args.max_epochs
    
    # Apply values from config if available
    if patterns_config.get('target_loss'):
        target_loss = patterns_config.get('target_loss')
    if patterns_config.get('max_epochs'):
        max_epochs = patterns_config.get('max_epochs')

    # Print available model types
    print(f"Available model types: {list_available_models()}")
    print(f"Using model type: {args.model_type}")
    print(f"Testing {len(pattern_names_to_test)} patterns with {len(model_configurations)} model sizes")
    print(f"Target loss: {target_loss}, Max epochs: {max_epochs}")

    results = {}
    trained_models = {}  # Store trained models for cross-validation
    
    # First pass: Train models for each pattern
    for pattern_name in pattern_names_to_test:
        rule = patterns[pattern_name]
        print(f"\n--- Testing pattern: {pattern_name} ---")
        results[pattern_name] = {}
        best_config_name = "N/A"
        best_epochs = "N/A"
        best_model = None
        best_val_loss = float('inf')

        for config_name, config in model_configurations.items():
            print(f"\nTrying model size: {config_name}")
            train_loader, test_loader = create_dataloaders(
                patterns={pattern_name: rule}, # Only create dataloader for current pattern
                tokenizer=tokenizer,
                max_seq_len=args.max_seq_len,
                batch_size=args.batch_size,
                samples_per_pattern=args.samples_per_pattern,
                num_workers=args.num_workers
            )

            model = get_model(
                model_type=args.model_type,
                vocab_size=tokenizer.vocab_size,
                block_size=args.max_seq_len,
                n_embd=config['n_embd'],
                n_layer=config['n_layer'],
                n_head=config['n_head'],
                dropout=args.dropout,
                # Additional model-specific parameters
                mlp_ratio=args.mlp_ratio,
                qk_norm=args.qk_norm,
                num_experts=args.num_experts,
                top_k=args.top_k_experts,
                aux_loss_weight=args.aux_loss_weight
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None

            epochs_trained, val_loss = train_for_complexity_test(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                target_loss=target_loss,
                max_epochs=max_epochs,
                grad_accumulation_steps=args.grad_accumulation_steps
            )

            if val_loss <= target_loss:
                best_config_name = config_name
                best_epochs = epochs_trained
                best_model = model  # Save the best model
                best_val_loss = val_loss
                results[pattern_name][config_name] = epochs_trained
                print(f"Target loss reached with model size: {config_name} in {epochs_trained} epochs.")
                break # Move to the next pattern after finding a suitable model size
            elif val_loss < best_val_loss:
                # Keep track of the best model even if it doesn't reach target loss
                best_config_name = config_name
                best_epochs = "Not Reached"
                best_model = model
                best_val_loss = val_loss
                results[pattern_name][config_name] = "Not Reached"
                print(f"Target loss not reached with model size: {config_name} after {max_epochs} epochs (Loss: {val_loss:.4f}).")
            else:
                results[pattern_name][config_name] = "Not Reached"
                print(f"Target loss not reached with model size: {config_name} after {max_epochs} epochs (Loss: {val_loss:.4f}).")

        if best_model is not None:
            print(f"\nPattern: {pattern_name} - Best Model Size: {best_config_name}, Epochs: {best_epochs}, Val Loss: {best_val_loss:.4f}")
            # Store the best model for cross-validation
            trained_models[pattern_name] = {
                'model': best_model,
                'model_size': best_config_name,
                'val_loss': best_val_loss
            }
        else:
            print(f"\nPattern: {pattern_name} - No viable model found.")

    # Output Summary Table
    print("\n--- Complexity Test Summary ---")
    print(f"Target Loss: {target_loss}, Max Epochs: {max_epochs}")
    print(f"Model Sizes Tested: {', '.join(model_configurations.keys())}")
    print(f"Model Type: {args.model_type}")
    print("-" * 50)
    print(f"{'Pattern':<25} {'Model Size':<15} {'Epochs to Target Loss'}")
    print("-" * 50)
    for pattern_name in pattern_names_to_test:
        pattern_results = results[pattern_name]
        best_model_size = "N/A"
        epochs_to_target = "N/A"
        for model_size, epoch_count in pattern_results.items():
            if isinstance(epoch_count, int): # Check if target loss was reached
                best_model_size = model_size
                epochs_to_target = epoch_count
                break # Take the smallest model size that worked
        print(f"{pattern_name:<25} {best_model_size:<15} {epochs_to_target}")
    print("-" * 50)
    
    # Save results to JSON file
    if args.save_results:
        result_file = args.save_results
        full_results = {
            "config": {
                "target_loss": target_loss,
                "max_epochs": max_epochs,
                "model_type": args.model_type,
                "model_sizes": model_configurations,
            },
            "results": results
        }
        
        try:
            with open(result_file, 'w') as f:
                json.dump(full_results, f, indent=2)
            print(f"Results saved to {result_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    # Cross-validation (if enabled)
    if args.cross_validate:
        print("\n--- Cross-Validation Results ---")
        if not trained_models:
            print("No models were successfully trained. Cannot perform cross-validation.")
        else:
            print("Validation Loss of Models Trained on Pattern (row) when Tested on Pattern (column)")
            
            # Print header row
            # First, collect pattern names and ensure they fit within the table
            max_pattern_len = max([len(p) for p in pattern_names_to_test]) + 2
            col_width = max(15, max_pattern_len)
            
            # Create table header
            header = "Trained On \\ Tested On"
            header_row = f"{header:<25}"
            for test_pattern in pattern_names_to_test:
                header_row += f" {test_pattern:<{col_width}}"
            print(header_row)
            print("-" * (25 + (col_width+1) * len(pattern_names_to_test)))
            
            # For each trained model, evaluate on all test datasets
            cross_val_results = {}
            for train_pattern, model_info in trained_models.items():
                trained_model = model_info['model']
                trained_model.eval()  # Set to evaluation mode
                
                row = f"{train_pattern:<25}"
                cross_val_results[train_pattern] = {}
                
                # First collect all val_losses for this row
                val_losses_for_row = {}
                for test_pattern in pattern_names_to_test:
                    if test_pattern == train_pattern:
                        # For diagonal elements, use the stored validation loss
                        val_loss = model_info['val_loss']
                    else:
                        # Create test dataloader for this pattern
                        _, test_loader = create_dataloaders(
                            patterns={test_pattern: patterns[test_pattern]},
                            tokenizer=tokenizer,
                            max_seq_len=args.max_seq_len,
                            batch_size=args.batch_size,
                            samples_per_pattern=min(args.samples_per_pattern, 100),  # Use fewer samples for cross-validation
                            num_workers=args.num_workers
                        )
                        
                        # Evaluate the trained model on this test set
                        val_loss = evaluate_silently(trained_model, test_loader, device)
                    
                    val_losses_for_row[test_pattern] = val_loss
                    cross_val_results[train_pattern][test_pattern] = val_loss
                
                # Now format and print the row
                for test_pattern in pattern_names_to_test:
                    val_loss = val_losses_for_row[test_pattern]
                    row += f" {val_loss:.4f}".ljust(col_width+1)
                
                print(row)
            
            # Add a summary row showing best validation loss for each pattern
            print("-" * (25 + (col_width+1) * len(pattern_names_to_test)))
            summary_row = f"{'Best Loss':<25}"
            for test_pattern in pattern_names_to_test:
                best_loss = float('inf')
                for train_pattern in trained_models:
                    if test_pattern in cross_val_results[train_pattern]:
                        best_loss = min(best_loss, cross_val_results[train_pattern][test_pattern])
                
                if best_loss < float('inf'):
                    summary_row += f" {best_loss:.4f}".ljust(col_width+1)
                else:
                    summary_row += f" {'N/A':<{col_width}}"
            print(summary_row)
            
            print("-" * (25 + (col_width+1) * len(pattern_names_to_test)))
            print("Note: Lower values indicate better generalization")
            
            # Save cross-validation results if requested
            # Pause for clarity between output logs
            print("\nGenerating dataloaders for next model: ")
            
            if args.save_results:
                # Append to existing results file
                try:
                    with open(args.save_results, 'r') as f:
                        full_results = json.load(f)
                    
                    # Convert cross_val_results to a serializable format
                    serializable_cross_val = {}
                    for train_pattern, test_patterns in cross_val_results.items():
                        serializable_cross_val[train_pattern] = {
                            test_pattern: loss for test_pattern, loss in test_patterns.items()
                        }
                    
                    full_results["cross_validation"] = serializable_cross_val
                    
                    with open(args.save_results, 'w') as f:
                        json.dump(full_results, f, indent=2)
                    print(f"Cross-validation results appended to {args.save_results}")
                except Exception as e:
                    print(f"Error saving cross-validation results: {str(e)}")
    
    print("Test complete.")


def create_default_config(config_path: str):
    """
    Create a default configuration file.
    
    Parameters:
    -----------
    config_path : str
        Path to save the configuration file
    """
    default_config = {
        "patterns": ["linear", "even_numbers", "fibonacci_type", "powers_of_2", "modulo_5"],
        "model_sizes": {
            "tiny": {"n_embd": 32, "n_layer": 1, "n_head": 1},
            "small": {"n_embd": 64, "n_layer": 2, "n_head": 2},
            "medium": {"n_embd": 128, "n_layer": 4, "n_head": 4},
            "large": {"n_embd": 256, "n_layer": 6, "n_head": 8}
        },
        "target_loss": 0.02,
        "max_epochs": 20
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Default configuration file created at {config_path}")
    except Exception as e:
        print(f"Error creating configuration file: {str(e)}")


def main():
    """Main entry point for complexity testing."""
    parser = argparse.ArgumentParser(description="Complexity Test for Number Sequence Prediction")
    
    # Create config subcommand
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create config command
    config_parser = subparsers.add_parser("create_config", help="Create a default configuration file")
    config_parser.add_argument("--output", type=str, default="complexity_config.json", 
                             help="Output path for the configuration file")
    
    # Run test command
    test_parser = subparsers.add_parser("run", help="Run complexity tests")
    
    # Get available model types
    available_models = list_available_models()
    
    # Model type selection
    test_parser.add_argument("--model_type", type=str, default='gpt2', choices=available_models,
                          help=f"Type of transformer model to use. Available: {available_models}")
    
    # Verbosity setting
    test_parser.add_argument("--verbosity", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                          help="Set the logging verbosity level")
    
    # Configuration file
    test_parser.add_argument("--patterns_config", type=str, default=None,
                          help="Path to patterns configuration file (JSON)")
    
    # Test specific patterns
    test_parser.add_argument("--pattern_type", type=str, default=None, 
                          help="Specific pattern type to test (from library)")
    test_parser.add_argument("--pattern_rule", type=str, default=None, 
                          help="Custom pattern rule to test, overrides pattern_type")
    
    # Cross-validation
    test_parser.add_argument("--cross_validate", action="store_true", 
                          help="Evaluate how models trained on each pattern perform on all other patterns")
    
    # Results storage
    test_parser.add_argument("--save_results", type=str, default=None,
                          help="Path to save results in JSON format")
    
    # Testing parameters
    test_parser.add_argument("--max_seq_len", type=int, default=64, 
                          help="Maximum sequence length for complexity test")
    test_parser.add_argument("--batch_size", type=int, default=64, 
                          help="Batch size for complexity test")
    test_parser.add_argument("--lr", type=float, default=1e-3, 
                          help="Learning rate for complexity test")
    test_parser.add_argument("--weight_decay", type=float, default=0.01, 
                          help="Weight decay")
    test_parser.add_argument("--grad_accumulation_steps", type=int, default=1, 
                          help="Gradient accumulation steps")
    test_parser.add_argument("--samples_per_pattern", type=int, default=2000,
                          help="Reduced samples per pattern for faster testing")
    test_parser.add_argument("--num_workers", type=int, default=0, 
                          help="Number of dataloader workers")
    test_parser.add_argument("--seed", type=int, default=42, 
                          help="Random seed")
    test_parser.add_argument("--cpu", action="store_true", 
                          help="Force CPU usage for testing")
    test_parser.add_argument("--log_file", type=str, default="complexity_test.log", 
                          help="Log file path for complexity test")
    test_parser.add_argument("--target_loss", type=float, default=0.02, 
                          help="Target validation loss for complexity test")
    test_parser.add_argument("--max_epochs", type=int, default=20, 
                          help="Maximum epochs to train per model size")
    
    # Model size arguments
    default_model_sizes = {
        'small': {'n_embd': 64, 'n_layer': 2, 'n_head': 2}, 
        'medium': {'n_embd': 128, 'n_layer': 4, 'n_head': 4}, 
        'large': {'n_embd': 256, 'n_layer': 6, 'n_head': 8}
    }
    
    test_parser.add_argument("--dropout", type=float, default=0.1, 
                          help="Dropout probability for complexity test")
    test_parser.add_argument('--model_sizes', type=eval, default=default_model_sizes,
                          help='Model configurations to test, e.g., \'{"small": {"n_embd": 64, "n_layer": 2, "n_head": 2}, "medium": ..., "large": ...}\'')
    test_parser.add_argument("--n_embd", type=int, default=None, 
                          help="Embedding dimension (overrides model_sizes for single size test)")
    test_parser.add_argument("--n_layer", type=int, default=None, 
                          help="Number of transformer layers (overrides model_sizes for single size test)")
    test_parser.add_argument("--n_head", type=int, default=None, 
                          help="Number of attention heads (overrides model_sizes for single size test)")
    
    # Additional model parameters for different architectures
    test_parser.add_argument("--mlp_ratio", type=float, default=4.0, 
                          help="MLP expansion ratio")
    test_parser.add_argument("--qk_norm", action="store_true", 
                          help="Use QK normalization")
    test_parser.add_argument("--num_experts", type=int, default=8, 
                          help="Number of experts (for MoE models)")
    test_parser.add_argument("--top_k_experts", type=int, default=2, 
                          help="Number of experts to route to")
    test_parser.add_argument("--aux_loss_weight", type=float, default=0.01, 
                          help="Weight for auxiliary losses")

    args = parser.parse_args()
    
    # Override model_sizes if n_embd, n_layer, n_head are provided directly for single size testing
    if hasattr(args, 'n_embd') and args.n_embd is not None and args.n_layer is not None and args.n_head is not None:
        args.model_sizes = {"single_size": {'n_embd': args.n_embd, 'n_layer': args.n_layer, 'n_head': args.n_head}}

    # Setup logging
    if args.command == "run" or not args.command:
        # Use verbosity for the run command
        setup_logging(
            args.log_file if hasattr(args, 'log_file') else None,
            args.verbosity if hasattr(args, 'verbosity') else "INFO"
        )
    else:
        # Default INFO for other commands
        setup_logging(args.log_file if hasattr(args, 'log_file') else None)

    # Run appropriate command
    if args.command == "create_config":
        create_default_config(args.output)
    elif args.command == "run" or not args.command:  # Default to run if no command specified
        run_complexity_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()