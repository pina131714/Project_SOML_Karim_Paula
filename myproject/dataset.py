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

    This module assumes the dataset is organized in the standard ImageFolder format, with separate 
    subdirectories for each class.

    Parameters
    ----------
    data_dir : str, optional
        Path to the directory containing the dataset. Defaults to '../data/processed/Car-Bike-Dataset'.
    batch_size : int, optional
        Number of samples per batch to load. Defaults to 8.
    """

    def __init__(self, data_dir=os.path.join(os.path.dirname(__file__), '..', r'data/processed/Car-Bike-Dataset'), batch_size=8):
        super().__init__()
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

    def prepare_data(self):
        """
        Placeholder for data preparation logic such as downloading.

        In this case, the dataset is assumed to be local and already prepared.
        """
        pass

    def setup(self, stage=None):
        """
        Sets up the training and validation datasets.

        It loads the full dataset from disk, applies transformations, and creates train/test splits 
        with a fixed random seed for reproducibility.

        Parameters
        ----------
        stage : str or None, optional
            Stage to set up ("fit", "validate", "test", or "predict"). Not used in this implementation.
        """
        full_dataset = datasets.ImageFolder(self._data_dir, transform=self._transform)

        bike_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]  # Bike = 0
        car_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 1]  # Car = 1

        random.seed(42)

        train_indices = bike_indices[:40] + car_indices[:40]
        test_indices = bike_indices[40:] + car_indices[40:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns
        -------
        DataLoader
            DataLoader object for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns
        -------
        DataLoader
            DataLoader object for the validation set.
        """
        return DataLoader(self.test_dataset, batch_size=8, shuffle=False)

