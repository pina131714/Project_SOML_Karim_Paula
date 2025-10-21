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

def run_training(epochs: int, learning_rate: float, batch_size: int):
    """
    Main function to run the training and validation process.
    
    Args:
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for the data loaders.
        
    Returns:
        dict: A dictionary containing the validation results.
    """
    # We pass the batch_size and learning_rate to our modules
    # (Ensure your classes are updated to accept these arguments)
    data_module = CarBikeDataModule(batch_size=batch_size)
    model = CarBikeClassifier(learning_rate=learning_rate)

    # Configure TensorBoard logger for tracking metrics and losses
    # Using a 'demo' name for runs initiated from the app
    logger = TensorBoardLogger("tb_logs", name="car_bike_model")

    # Initialize the trainer with the given epochs and the defined logger
    trainer = pl.Trainer(
        max_epochs=epochs, 
        log_every_n_steps=1, 
        logger=logger,
        accelerator="auto" # Auto-detect GPU/CPU
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Validate the model
    val_results = trainer.validate(model, datamodule=data_module)

    # Directory to save the trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Current script directory
    model_dir = os.path.join(current_dir, '..', 'models')     # 'models' folder in project root
    os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Full path to save the model
    # We save this as a 'demo' model to avoid overwriting your main 'car_bike_model.pth'
    model_path = os.path.join(model_dir, 'car_bike_model.pth')
    
    # Save only the model state dict (weights)
    torch.save(model._model.state_dict(), model_path)

    print(f"Model (demo) saved at: {model_path}")
    
    # Return the validation metrics for Gradio
    if val_results:
        return val_results[0]
    return {}


if __name__ == '__main__':
    
    # Define default parameters for script execution
    DEFAULT_EPOCHS = 3
    DEFAULT_LR = 1e-3
    DEFAULT_BATCH_SIZE = 32

    print("--- Running train.py as a script ---")

    # Run the training
    final_metrics = run_training(
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE
    )
    
    print("--- Script execution finished ---")
    print("Final Validation Metrics:")
    print(final_metrics)
