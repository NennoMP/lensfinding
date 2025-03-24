"""Module containing logic for loading the Specific Test II - Lens Finding dataset."""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LensFindingDataset(Dataset):
    """Dataset class for Specific Test II - Lens Finding. 

    The dataset consists of observational data of strong lenses and non-lensed galaxies.
    Structure:
        # Strong lenses
        train_lenses/
            1.npy, 2.npy, ...
        test_lenses/
            1.npy, 2.npy, ...
        # Non-lensed galaxies
        train_nonlenses/
            1.npy, 2.npy, ...
        test_nonlenses/
            1.npy, 2.npy, ...

    Attributes:
        root_dir: a string representing the directory containing the dataset.
        stretch: whether to apply square-root stretch to images or not.
        data: a list containing the paths to the images and their corresponding labels.
        class_to_idx: a dictionary mapping class names to class integers.
    """

    def __init__(
        self, 
        root_dir: str = 'datasets/lensfinding', 
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
        self.root_dir = Path(root_dir)
        self.stretch = stretch

        self.data = []
        self.class_to_idx = {'nonlenses': 0, 'lenses': 1} # classes mapping

        # Load lenses
        lenses_dir = self.root_dir / f"{split}_lenses"
        if lenses_dir.exists():
            for file_path in sorted(lenses_dir.glob('*.npy'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem):
                self.data.append((str(file_path), self.class_to_idx['lenses']))
        # Load non-lenses
        nonlenses_dir = self.root_dir / f"{split}_nonlenses"
        if nonlenses_dir.exists():
            for file_path in sorted(nonlenses_dir.glob('*.npy'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem):
                self.data.append((str(file_path), self.class_to_idx['nonlenses']))
    
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