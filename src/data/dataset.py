"""
Dataset utilities for loading paired fundus images.
"""

import os
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torch

class FundusDataset(Dataset):
    """Dataset for paired left/right fundus images with multi-label annotations."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load the annotation CSV.
        self.data = pd.read_csv(csv_file)

        # Drop rows with NaN in any of the label columns.
        self.data = self.data.dropna(subset=self.data.columns[3:])

        # Restore a contiguous 0-based index for clean integer indexing.
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Resolve the left/right image paths from the CSV row.
        left_img_path  = os.path.join(self.root_dir, self.data.iloc[idx, 1])
        right_img_path = os.path.join(self.root_dir, self.data.iloc[idx, 2])

        # Load RGB images.
        left_img  = Image.open(left_img_path).convert("RGB")
        right_img = Image.open(right_img_path).convert("RGB")

        # Apply the requested data transforms / augmentations.
        if self.transform:
            left_img  = self.transform(left_img)
            right_img = self.transform(right_img)

        # Extract the 8-column label vector.
        labels = self.data.iloc[idx, 3:].values

        # Coerce labels to numeric values.
        labels = labels.astype(float)

        # Convert to a float tensor.
        labels = torch.FloatTensor(labels)

        return left_img, right_img, labels
