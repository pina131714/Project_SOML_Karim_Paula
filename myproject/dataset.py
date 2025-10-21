# myproject/dataset.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
import random
import pytorch_lightning as pl
import os

class CarBikeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preparing a binary image classification dataset
    consisting of car and bike images.

    (Refactored for Practice 5)
    This module now assumes a pre-split 'train' and 'test' folder structure
    created by 'preprocess.py'.

    Parameters
    ----------
    data_dir : str, optional
        Path to the 'processed' folder WHICH CONTAINS 'train' and 'test'.
        Defaults to '../data/processed'.
    batch_size : int, optional
        Number of samples per batch to load. Defaults to 8.
    """

    def __init__(self, data_dir=os.path.join(os.path.dirname(__file__), '..', r'data/processed'), batch_size=8):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

    def prepare_data(self):
        """
        Data is assumed to be prepared by 'myproject/preprocess.py'.
        This function just checks if it exists.
        """
        train_path = os.path.join(self._data_dir, 'train')
        if not os.path.exists(train_path):
            print(f"Error: Training directory not found at: {train_path}")
            print("Please run 'uv run python -m myproject.preprocess' first.")
            raise FileNotFoundError(f"Directory not found: {train_path}")

    def setup(self, stage=None):
        """
        Sets up the training and validation (test) datasets.
        
        Loads datasets directly from 'train' and 'test' folders.
        """
        train_path = os.path.join(self._data_dir, 'train')
        test_path = os.path.join(self._data_dir, 'test')
        
        self.train_dataset = datasets.ImageFolder(train_path, transform=self._transform)
        self.test_dataset = datasets.ImageFolder(test_path, transform=self._transform)
        
        print(f"Data loaded: {len(self.train_dataset)} train images, {len(self.test_dataset)} test images.")

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.
        """
        # num_workers > 0 speeds up data loading
        return DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation (test) dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=4)
