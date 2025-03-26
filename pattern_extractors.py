"""
Utilities for extracting and managing pattern information from datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union


class PatternAwareNumberSequenceDataset(Dataset):
    """
    Wrapper for the NumberSequenceDataset that adds pattern IDs.
    Compatible with the existing dataset structure.
    """

    def __init__(self, dataset):
        self.dataset = dataset

        # Extract unique pattern names from the dataset
        self.pattern_names = []
        self.pattern_to_id = {}

        # Sample a few items to identify patterns
        sample_size = min(100, len(dataset))
        for i in range(sample_size):
            _, _, pattern_name = dataset[i]
            if pattern_name not in self.pattern_to_id:
                self.pattern_to_id[pattern_name] = len(self.pattern_to_id)
                self.pattern_names.append(pattern_name)

        self.id_to_pattern = {id: name for name, id in self.pattern_to_id.items()}
        print(f"Detected {len(self.pattern_names)} unique pattern types")
        for name, id in self.pattern_to_id.items():
            print(f"  Pattern {id}: {name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get original item (x, y, pattern_name)
        x, y, pattern_name = self.dataset[idx]

        # Get pattern ID
        pattern_id = self.pattern_to_id.get(pattern_name, 0)

        return x, y, pattern_name, torch.tensor(pattern_id, dtype=torch.long)

    def get_num_patterns(self):
        """Get the number of unique pattern types."""
        return len(self.pattern_to_id)


def create_pattern_aware_dataloaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Dict[int, str], int]:
    """
    Create pattern-aware dataloaders from datasets.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Additional arguments for DataLoader

    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
        pattern_mapping: Mapping from pattern IDs to names
        num_patterns: Number of unique patterns
    """
    # Wrap datasets with pattern extractor
    train_dataset = PatternAwareNumberSequenceDataset(train_dataset)
    test_dataset = PatternAwareNumberSequenceDataset(test_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    # Get pattern mapping
    pattern_mapping = train_dataset.id_to_pattern

    # Get number of patterns
    num_patterns = train_dataset.get_num_patterns()

    return train_loader, test_loader, pattern_mapping, num_patterns


class PatternStatistics:
    """
    Utility for tracking pattern-specific statistics during training.
    """

    def __init__(self, pattern_mapping: Dict[int, str] = None):
        self.pattern_mapping = pattern_mapping or {}
        self.stats = {}
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.stats = {
            pattern_id: {
                'loss': 0.0,
                'count': 0,
                'name': pattern_name
            }
            for pattern_id, pattern_name in self.pattern_mapping.items()
        }

        # Add entry for unknown patterns
        if -1 not in self.stats:
            self.stats[-1] = {
                'loss': 0.0,
                'count': 0,
                'name': 'unknown'
            }

    def update(self, pattern_ids, losses):
        """
        Update statistics for specific patterns.

        Args:
            pattern_ids: Tensor or list of pattern IDs
            losses: Tensor or list of corresponding losses
        """
        if isinstance(pattern_ids, torch.Tensor):
            pattern_ids = pattern_ids.cpu().numpy()

        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()

        for i, pattern_id in enumerate(pattern_ids):
            # Use -1 for None or unknown patterns
            pid = pattern_id if pattern_id is not None else -1

            # Ensure we have an entry for this pattern
            if pid not in self.stats:
                name = self.pattern_mapping.get(pid, f'pattern_{pid}')
                self.stats[pid] = {'loss': 0.0, 'count': 0, 'name': name}

            # Update statistics
            self.stats[pid]['loss'] += float(losses[i])
            self.stats[pid]['count'] += 1

    def get_summary(self):
        """
        Get summary statistics.

        Returns:
            Dictionary of pattern statistics
        """
        summary = {}

        for pattern_id, data in self.stats.items():
            if data['count'] > 0:
                avg_loss = data['loss'] / data['count']
                summary[pattern_id] = {
                    'avg_loss': avg_loss,
                    'count': data['count'],
                    'name': data['name']
                }

        return summary

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()

        print("\nPerformance by pattern type:")
        for pattern_id, data in sorted(summary.items()):
            print(f"  {data['name']} (ID: {pattern_id}): "
                  f"Loss = {data['avg_loss']:.4f}, "
                  f"Samples = {data['count']}")