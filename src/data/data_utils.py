"""Module containg utility functions for creating dataloaders."""
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


def get_dataloaders(
    root_dir: str, 
    dataset_class,
    val_split: float = 0.2,
    batch_size: int = 64,
    random_state: int = 42,
    stretch: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:   
    """Get dataloaders for a given dataset.

    Args:
        root_dir: directory containing the dataset.
        dataset_class: class to use for the dataset. Options are 'StrongLensingDataset' or 
        'LensFindingDataset'.
        val_split: percentage of the training data to use for validation.
        batch_size: mini-batch size.
        random_state: random seed for reproducibility.
        apply_sqrt_stretch: whether to apply square-root stretch to the images.

    Returns:
        A tuple of four dataloders: (full_train_dataloader, train_dataloader, val_dataloader, 
        test_dataloader). Respectively, they are the contain the original training data, the 
        training data after the train-val split, the validation data, and the test data.
    """
    # Load dataset
    full_train_dataset = dataset_class(root_dir=root_dir, split='train', stretch=stretch)
    test_dataset = dataset_class(root_dir=root_dir, split='test', stretch=stretch)

    # Perform a train-val stratified split
    targets = [full_train_dataset.data[i][1] for i in range(len(full_train_dataset))]
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=val_split,
        random_state=random_state,
        stratify=targets  # ensures original class distribution
    )
    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)
    
    # Dataloaders
    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    train_dataloader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    return (full_train_dataloader, train_dataloader, val_dataloader, test_dataloader)