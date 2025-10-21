# carbike_classification/evaluate_model.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.figure

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.calibration import CalibrationDisplay
from torchvision import models
from carbike_classification.dataset import CarBikeDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

def unnormalize(img_tensor):
    """
    Unnormalize a tensor image using ImageNet statistics.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image tensor of shape (C, H, W) normalized using ImageNet mean and std.

    Returns
    -------
    torch.Tensor
        Unnormalized image tensor with the same shape.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img_tensor * std + mean

def load_model_and_data(model_path=None, processed_dir=None):
    """
    Load the trained model and validation dataset.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved PyTorch model (.pth). Defaults to '../models/car_bike_model.pth'.
    processed_dir : str, optional
        Path to the processed dataset folder containing 'train' and 'test'.
        Defaults to '../data/processed'.

    Returns
    -------
    model : torch.nn.Module
        Loaded PyTorch model.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation dataset.
    class_names : list of str
        List of class names.
    device : torch.device
        Device on which the model is loaded (CPU or GPU).
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_bike_model.pth')
    if processed_dir is None:
        processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg11(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    data_module = CarBikeDataModule(processed_dir=processed_dir)
    data_module.setup(stage='test')
    val_loader = data_module.val_dataloader()
    class_names = list(val_loader.dataset.classes)
    
    return model, val_loader, class_names, device

def run_evaluation(model, val_loader, device, num_classes):
    """
    Run evaluation loop to collect predictions, probabilities, losses, and labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.
    val_loader : DataLoader
        Validation DataLoader.
    device : torch.device
        Device to run the evaluation on.
    num_classes : int
        Number of classes.

    Returns
    -------
    all_labels : list of int
        True labels.
    all_preds : list of int
        Predicted labels.
    all_probs_per_class : list of list of float
        Probabilities per class.
    all_losses : list of float
        Cross-entropy losses per sample.
    all_images : list of torch.Tensor
        Original input images (unnormalized later).
    """
    all_preds = []
    all_probs_per_class = [[] for _ in range(num_classes)]
    all_labels = []
    all_losses = []
    all_images = []

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            losses = loss_fn(logits, yb)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_losses.extend(losses.cpu().numpy())
            all_images.extend(xb.cpu())

            for c in range(num_classes):
                all_probs_per_class[c].extend(probs[:, c].cpu().numpy())
                
    return all_labels, all_preds, all_probs_per_class, all_losses, all_images

def get_accuracy(all_labels, all_preds):
    """
    Compute accuracy given true and predicted labels.

    Parameters
    ----------
    all_labels : list of int
        True labels.
    all_preds : list of int
        Predicted labels.

    Returns
    -------
    float
        Classification accuracy.
    """
    return accuracy_score(all_labels, all_preds)

def plot_confusion_matrix(all_labels, all_preds, class_names) -> matplotlib.figure.Figure:
    """
    Generate a confusion matrix plot.

    Parameters
    ----------
    all_labels : list of int
        True labels.
    all_preds : list of int
        Predicted labels.
    class_names : list of str
        Class names.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the confusion matrix.
    """
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    return fig

def plot_calibration_curves(all_labels, all_probs_per_class, class_names) -> matplotlib.figure.Figure:
    """
    Generate calibration curves per class.

    Parameters
    ----------
    all_labels : list of int
        True labels.
    all_probs_per_class : list of list of float
        Probabilities per class.
    class_names : list of str
        Class names.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing calibration curves.
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(6*num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for c, class_name in enumerate(class_names):
        y_true_binary = [1 if label == c else 0 for label in all_labels]
        CalibrationDisplay.from_predictions(
            y_true=y_true_binary,
            y_prob=all_probs_per_class[c],
            n_bins=10,
            name=class_name,
            ax=axes[c]
        )
        axes[c].set_xlabel(f'Mean Predicted Probability ({class_name})')
        axes[c].set_ylabel(f'Fraction of positives ({class_name})')
        axes[c].set_title(f'Calibration Curve: {class_name}')

    plt.tight_layout()
    return fig

def plot_highest_loss_samples(all_labels, all_preds, all_losses, all_images, class_names, top_k=5) -> matplotlib.figure.Figure:
    """
    Plot the top-k samples with the highest loss.

    Parameters
    ----------
    all_labels : list of int
        True labels.
    all_preds : list of int
        Predicted labels.
    all_losses : list of float
        Loss per sample.
    all_images : list of torch.Tensor
        Original input images.
    class_names : list of str
        Class names.
    top_k : int, optional
        Number of top loss samples to plot. Default is 5.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the top-k loss samples.
    """
    top_k_indices = np.argsort(all_losses)[-top_k:]
    to_pil = ToPILImage()

    fig, axs = plt.subplots(1, top_k, figsize=(15, 4))
    if top_k == 1:
        axs = [axs]

    for i, idx in enumerate(top_k_indices):
        img = unnormalize(all_images[idx])
        img = to_pil(img)
        true_label = class_names[all_labels[idx]]
        pred_label = class_names[all_preds[idx]]
        loss = all_losses[idx]
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'True: {true_label}\nPred: {pred_label}\nLoss: {loss:.2f}')

    plt.tight_layout()
    return fig

def main(model_path=None, figures_dir=None):
    """
    Run full evaluation and save visualizations.

    Parameters
    ----------
    model_path : str, optional
        Path to the trained model (.pth). Defaults to '../models/car_bike_model.pth'.
    figures_dir : str, optional
        Directory to save the generated figures. Defaults to '../reports/figures'.
    """
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_bike_model.pth')
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading model and data...")
    model, val_loader, class_names, device = load_model_and_data(model_path=model_path)
    num_classes = len(class_names)

    print("Running evaluation...")
    all_labels, all_preds, all_probs, all_losses, all_images = run_evaluation(
        model, val_loader, device, num_classes
    )

    accuracy = get_accuracy(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')

    print("Generating and saving plots...")
    fig_cm = plot_confusion_matrix(all_labels, all_preds, class_names)
    fig_cm.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
    plt.close(fig_cm)

    fig_cal = plot_calibration_curves(all_labels, all_probs, class_names)
    fig_cal.savefig(os.path.join(figures_dir, 'calibration_curves_per_class.png'))
    plt.close(fig_cal)

    fig_loss = plot_highest_loss_samples(all_labels, all_preds, all_losses, all_images, class_names)
    fig_loss.savefig(os.path.join(figures_dir, 'highest_loss_samples.png'))
    plt.close(fig_loss)

    print(f"All visualizations saved in: {figures_dir}")

if __name__ == '__main__':
    main()
