# carbike_classification/train.py
"""
Train and validate the CarBikeClassifier model using PyTorch Lightning.

Training uses CarBikeDataModule and logs metrics via TensorBoard.
After training, model weights are saved to the specified models directory.

Default save paths:
- Model: ../models/car_bike_model.pth
- TensorBoard logs: ./tb_logs (inside source folder)
"""

import torch
import os
from pathlib import Path
import pytorch_lightning as pl
from carbike_classification.dataset import CarBikeDataModule
from carbike_classification.model import CarBikeClassifier
from pytorch_lightning.loggers import TensorBoardLogger

def run_training(
    epochs: int,
    learning_rate: float,
    batch_size: int,
    tb_logs_dir: str = None,
    models_dir: str = None,
    processed_dir: str = None
):
    """
    Train the model with optional directories for logs and model saving.

    Args:
        epochs (int)
        learning_rate (float)
        batch_size (int)
        tb_logs_dir (str, optional): Directory to save TensorBoard logs.
            Defaults to tb_logs/ inside source folder.
        models_dir (str, optional): Directory to save the trained model.
            Defaults to ../models/ relative to this script.
        processed_dir (str, optional): Directory with preprocessed data.
            Defaults to ../data/processed relative to project root.
    """
    # Paths
    current_dir = Path(__file__).resolve().parent
    tb_logs_dir = Path(tb_logs_dir) if tb_logs_dir else current_dir / "tb_logs"
    models_dir = Path(models_dir) if models_dir else current_dir.parent / "models"
    processed_dir = Path(processed_dir) if processed_dir else current_dir.parent / "data" / "processed"

    # Data module
    data_module = CarBikeDataModule(batch_size=batch_size, processed_dir=processed_dir)
    # Model
    model = CarBikeClassifier(learning_rate=learning_rate)

    # Logger
    logger = TensorBoardLogger(tb_logs_dir, name="car_bike_model")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=1,
        logger=logger,
        accelerator="auto"
    )

    # Train
    trainer.fit(model, datamodule=data_module)
    val_results = trainer.validate(model, datamodule=data_module)

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / "car_bike_model.pth"
    torch.save(model._model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")
    print(f"TensorBoard logs saved at: {tb_logs_dir}")

    return val_results[0] if val_results else {}

# Script execution
if __name__ == "__main__":
    DEFAULT_EPOCHS = 3
    DEFAULT_LR = 1e-3
    DEFAULT_BATCH_SIZE = 32

    print("--- Running train.py as a script ---")

    final_metrics = run_training(
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LR,
        batch_size=DEFAULT_BATCH_SIZE
    )

    print("--- Script execution finished ---")
    print("Final Validation Metrics:")
    print(final_metrics)
