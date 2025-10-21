# carbike_classification/dataset.py
import os
from torchvision import datasets, models
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class CarBikeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for car vs bike classification.
    
    Parameters
    ----------
    processed_dir : str or Path, optional
        Path to the 'processed' folder containing 'train' and 'test'.
        Defaults to '../data/processed' relative to this file.
    batch_size : int, optional
        Batch size for training. Defaults to 8.
    """
    def __init__(self, batch_size=8, processed_dir=None):
        super().__init__()
        default_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        self._data_dir = str(processed_dir) if processed_dir else default_dir
        self._batch_size = batch_size
        self._transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()

    def prepare_data(self):
        """
        Data is assumed to be prepared.
        This function just checks if it exists.
        """
        train_path = os.path.join(self._data_dir, 'train')
        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Training directory not found at {train_path}. "
                "Run preprocess.py first or provide a valid processed_dir."
            )

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
