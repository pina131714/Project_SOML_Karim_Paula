import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
import random
import pytorch_lightning as pl
import os
from dataset import CarBikeDataModule
from pytorch_lightning.loggers import TensorBoardLogger

class CarBikeClassifier(pl.LightningModule):
    """
    PyTorch Lightning module for binary image classification (Car vs. Bike) using VGG11.

    This model uses a pretrained VGG11 backbone from torchvision, with the final fully connected
    layer adapted for two output classes. Feature extraction layers are frozen during training.

    Attributes
    ----------
    model : torchvision.models.VGG
        Pretrained VGG11 model with modified classifier head.
    loss_fn : nn.Module
        Loss function used during training (CrossEntropyLoss).
    """

    def __init__(self):
        """
        Initializes the CarBikeClassifier model, replaces the final classifier layer,
        and freezes the convolutional feature extractor.
        """
        super().__init__()
        self.model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        self.model.classifier[6] = nn.Linear(4096, 2)

        # Freeze feature layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, 2).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input images and their corresponding labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Loss value for the batch.
        """
        xb, yb = batch
        out = self(xb)
        loss = self.loss_fn(out, yb)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A batch of input images and their corresponding labels.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Accuracy of the batch.
        """
        xb, yb = batch
        out = self(xb)
        preds = torch.argmax(out, dim=1)
        acc = (preds == yb).float().mean()
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer for training the model.
        """
        return optim.Adam(self.parameters())
