"""Module containing logic for loading the Common Test I - Multi-class Classification dataset."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import os

class MultiClassClassificationDataset(Dataset):
    """Dataset class for Common Test I - Multi-Class Classification. 

    The dataset consists of three classes, strong lensing images with no substructure, subhalo 
    substructure, and vortex substructure. Data is already min-max normalized.
    Structure:
        train/
            # No substructure
            no/
                1.npy, 2.npy, ...
            # Subhalo substructure
            sphere/
                1.npy, 2.npy, ...
            # Vortex substructure
            vort/
                1.npy, 2.npy, ...
        test/
            # No substructure
            no/
                1.npy, 2.npy, ...
            # Subhalo substructure
            sphere/
                1.npy, 2.npy, ...
            # Vortex substructure
            vort/
                1.npy, 2.npy, ...

    Attributes:
        root_dir: a string representing the directory containing the dataset.
        stretch: whether to apply square-root stretch to images or not.
        data: a list containing the paths to the images and their corresponding labels.
        class_to_idx: a dictionary mapping class names to class integers.
    """

    def __init__(
        self, 
        root_dir: str = 'datasets/multiclass_classification', 
        split: str = 'train', 
        stretch: bool = True
    ) -> None:
        """
        Args:
            root_dir: directory containing the dataset.
            split: split of the dataset to load. Options are 'train' or 'test'.
            stretch: whether to apply square-root stretch to images or not.

        Raises:
            AssertionError: if split is not 'train' or 'test'.
        """
        assert split in ['train', 'test'], "split must be either 'train' or 'test'"
        if split == 'test':
            split = 'val'
        self.root_dir = Path(root_dir)
        self.stretch = stretch

        self.data = []
        self.class_to_idx = {'no': 0, 'sphere': 1, 'vort': 2} # classes mapping

        split_dir = self.root_dir / split
        # Load data
        for subdir in ['no', 'sphere', 'vort']:
            class_dir = split_dir / subdir
            if not class_dir.exists():
                continue
                
            class_idx = self.class_to_idx[subdir]
            for file_path in sorted(class_dir.glob('*.npy'), key=lambda x: int(x.stem)):
                self.data.append((str(file_path), class_idx))
    
    def __len__(self) -> int:
        "Return size of the dataset."
        return len(self.data)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """Return data sample at given index."""
        image, label = self.data[index]
        image = torch.from_numpy(np.load(image)).float()
        if self.stretch:
            image = torch.sqrt(image)
        return image, label