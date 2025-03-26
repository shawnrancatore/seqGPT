import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import random

from pattern_generators import generate_pattern, create_pattern_library
from tokenizer import NumberStreamTokenizer


class NumberSequenceDataset(Dataset):
    """Dataset for number sequences generated from pattern rules."""

    def __init__(
        self,
        patterns: Dict[str, str],
        tokenizer: NumberStreamTokenizer,
        split: str = 'train',
        max_seq_len: int = 128,
        samples_per_pattern: int = 500,
        test_size: float = 0.2,
        seed: int = 42,
        recursive_pattern_multiplier: int = 4
    ):
        """
        Initialize the dataset.

        Parameters:
        -----------
        patterns : dict
            Dictionary mapping pattern names to pattern rules
        tokenizer : NumberStreamTokenizer
            Tokenizer for the sequences
        split : str
            'train' or 'test' split
        max_seq_len : int
            Maximum sequence length
        samples_per_pattern : int
            Number of samples to generate for each pattern
        test_size : float
            Fraction of data to use for testing
        seed : int
            Random seed for reproducibility
        recursive_pattern_multiplier : int
            Multiplier for samples of recursive patterns
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Fix the random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Generate and process sequences
        self.data = []
        self.pattern_labels = []

        for pattern_name, rule in patterns.items():
            print(f"Generating sequences for pattern: {pattern_name}")

            # Determine how many samples to generate for this pattern
            pattern_samples = samples_per_pattern
            if "fibonacci" in pattern_name or "recursive" in pattern_name:
                pattern_samples *= recursive_pattern_multiplier

            pattern_data = []
            for i in range(samples_per_pattern):
                # Generate a sequence a bit longer than needed to allow for proper prediction
                seq_length = max_seq_len + 20 # Increased sequence length for more context
                sequence = generate_pattern(rule, length=seq_length)

                # Tokenize the sequence
                tokens = self.tokenizer.tokenize_sequence(sequence)
                token_ids = self.tokenizer.numericalize(tokens)

                # Create input-target pairs
                if len(token_ids) > max_seq_len + 1:
                    # For longer sequences, randomly sample a sliding window
                    start_idx = random.randint(0, len(token_ids) - max_seq_len - 1)
                    end_idx = start_idx + max_seq_len + 1
                    token_ids = token_ids[start_idx:end_idx]

                # Ensure we have enough tokens
                if len(token_ids) >= max_seq_len + 1:
                    x = token_ids[:max_seq_len]
                    y = token_ids[1:max_seq_len+1]

                    pattern_data.append((x, y, pattern_name))

            # Split the pattern data into train and test
            random.shuffle(pattern_data)
            split_idx = int(len(pattern_data) * (1 - test_size))

            if split == 'train':
                self.data.extend(pattern_data[:split_idx])
            else:
                self.data.extend(pattern_data[split_idx:])

        # Shuffle the final dataset
        random.shuffle(self.data)

        print(f"Created {split} dataset with {len(self.data)} samples")

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a sample from the dataset.

        Parameters:
        -----------
        idx : int
            Index of the sample

        Returns:
        --------
        tuple
            (x, y, pattern_name) where:
            - x is the input sequence (tensor of token IDs)
            - y is the target sequence (tensor of token IDs)
            - pattern_name is the name of the pattern
        """
        x, y, pattern_name = self.data[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), pattern_name


def create_dataloaders(
    patterns: Dict[str, str],
    tokenizer: NumberStreamTokenizer,
    max_seq_len: int = 128,
    batch_size: int = 64,
    samples_per_pattern: int = 500,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Parameters:
    -----------
    patterns : dict
        Dictionary mapping pattern names to pattern rules
    tokenizer : NumberStreamTokenizer
        Tokenizer for the sequences
    max_seq_len : int
        Maximum sequence length
    batch_size : int
        Batch size for the dataloaders
    samples_per_pattern : int
        Number of samples to generate for each pattern
    num_workers : int
        Number of workers for the dataloaders

    Returns:
    --------
    tuple
        (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='train',
        max_seq_len=max_seq_len,
        samples_per_pattern=samples_per_pattern
    )

    test_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='test',
        max_seq_len=max_seq_len,
        samples_per_pattern=samples_per_pattern
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def test_dataset():
    """Test the dataset and dataloader functionality."""
    # Create a small pattern library for testing
    patterns = {
        "linear": "A + 1",
        "modulo_3": "A % 3",
        "alternating": "[0, 1]"
    }

    # Create tokenizer
    tokenizer = NumberStreamTokenizer(max_number=100)

    # Create datasets
    train_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='train',
        max_seq_len=32,
        samples_per_pattern=10
    )

    test_dataset = NumberSequenceDataset(
        patterns=patterns,
        tokenizer=tokenizer,
        split='test',
        max_seq_len=32,
        samples_per_pattern=10
    )

    # Test dataset
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Test dataloader
    train_loader, test_loader = create_dataloaders(
        patterns=patterns,
        tokenizer=tokenizer,
        max_seq_len=32,
        batch_size=2,
        samples_per_pattern=10,
        num_workers=0
    )

    # Get a batch
    x_batch, y_batch, pattern_names = next(iter(train_loader))
    print(f"Batch x shape: {x_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print(f"Pattern names: {pattern_names}")

    # Detokenize a sample
    sample_idx = 0
    x_tokens = tokenizer.denumericalize(x_batch[sample_idx].tolist())
    y_tokens = tokenizer.denumericalize(y_batch[sample_idx].tolist())

    print(f"Sample x tokens: {x_tokens[:20]}...")
    print(f"Sample y tokens: {y_tokens[:20]}...")

    # Reconstruct the sequence
    x_seq = tokenizer.reconstruct_sequence(x_tokens)
    print(f"Reconstructed x sequence: {x_seq[:10]}...")


if __name__ == "__main__":
    test_dataset()