"""
Train and validate the CarBikeClassifier model using PyTorch Lightning.

The training uses the CarBikeDataModule data loader and logs metrics via TensorBoard.
After training and validation, the trained model weights are saved to the 'models' directory.

Steps performed:
1. Initialize the data module and model.
2. Configure TensorBoard logger for metrics tracking.
3. Set up a PyTorch Lightning trainer with a max of 3 epochs.
4. Train and validate the model.
5. Save the trained model weights to disk.

The model is saved at: '../models/car_bike_model.pth'
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
import random
import pytorch_lightning as pl
import os
from myproject.dataset import CarBikeDataModule
from myproject.model import CarBikeClassifier
from pytorch_lightning.loggers import TensorBoardLogger


if __name__ == '__main__':
    data_module = CarBikeDataModule()
    model = CarBikeClassifier()

    # Configure TensorBoard logger for tracking metrics and losses
    logger = TensorBoardLogger("tb_logs", name="car_bike_model")

    # Initialize the trainer with 3 epochs and the defined logger
    trainer = pl.Trainer(max_epochs=3, log_every_n_steps=1, logger=logger)
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Validate the model
    trainer.validate(model, datamodule=data_module)

    # Directory to save the trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    model_dir = os.path.join(current_dir, '..', 'models')     # 'models' folder in project root
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Full path to save the model
    model_path = os.path.join(model_dir, 'car_bike_model.pth')
    
    # Save only the model state dict (weights)
    torch.save(model._model.state_dict(), model_path)

    print(f"Model saved at: {model_path}")
