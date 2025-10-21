import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.figure

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.calibration import CalibrationDisplay
from torchvision import models
from myproject.dataset import CarBikeDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

# === CONFIG ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_bike_model.pth')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
# We still ensure the directory exists for when the script is run directly
os.makedirs(FIGURES_DIR, exist_ok=True)


# === Helper Function ===
def unnormalize(img_tensor):
    """
    Reverses normalization on a tensor image using ImageNet stats.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Image tensor normalized with ImageNet statistics, of shape (C, H, W).

    Returns
    -------
    torch.Tensor
        Unnormalized image tensor with same shape.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return img_tensor * std + mean

# === Core Logic Functions ===

def load_model_and_data():
    """Loads the model, data, and class names for evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg11(weights=None)
    model.classifier[6] = torch.nn.Linear(4096, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    data_module = CarBikeDataModule()
    data_module.setup(stage='test')
    val_loader = data_module.val_dataloader()
    class_names = list(val_loader.dataset.classes)
    
    return model, val_loader, class_names, device

def run_evaluation(model, val_loader, device, num_classes):
    """Runs the evaluation loop to get predictions, losses, and labels."""
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

            # Store per-class probabilities
            for c in range(num_classes):
                all_probs_per_class[c].extend(probs[:, c].cpu().numpy())
                
    return all_labels, all_preds, all_probs_per_class, all_losses, all_images

def get_accuracy(all_labels, all_preds):
    """Calculates and returns the accuracy."""
    return accuracy_score(all_labels, all_preds)

# === Plotting Functions (for Gradio) ===

def plot_confusion_matrix(all_labels, all_preds, class_names) -> matplotlib.figure.Figure:
    """Generates and returns the confusion matrix plot."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Create a figure and plot to it
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    
    return fig # Return the figure object

def plot_calibration_curves(all_labels, all_probs_per_class, class_names) -> matplotlib.figure.Figure:
    """Generates and returns the calibration curves plot."""
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(6*num_classes, 5))
    
    # Handle the case of 1 class (though unlikely, good practice)
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
    return fig # Return the figure object

def plot_highest_loss_samples(all_labels, all_preds, all_losses, all_images, class_names, top_k=5) -> matplotlib.figure.Figure:
    """Generates and returns a plot of the top-k highest loss samples."""
    top_k_indices = np.argsort(all_losses)[-top_k:]
    to_pil = ToPILImage()

    fig, axs = plt.subplots(1, top_k, figsize=(15, 4))
    if top_k == 1: # Handle single subplot case
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
    return fig # Return the figure object

# === Main execution block (for running as a script) ===

if __name__ == '__main__':
    
    # 1. Load model and data
    print("Loading model and data...")
    model, val_loader, class_names, device = load_model_and_data()
    num_classes = len(class_names)
    
    # 2. Run evaluation
    print("Running evaluation...")
    all_labels, all_preds, all_probs, all_losses, all_images = run_evaluation(
        model, val_loader, device, num_classes
    )
    
    # 3. Print Accuracy
    accuracy = get_accuracy(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}')

    # 4. Generate and SAVE plots (original script behavior)
    print("Generating and saving plots...")
    
    # Confusion Matrix
    fig_cm = plot_confusion_matrix(all_labels, all_preds, class_names)
    fig_cm.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
    plt.close(fig_cm) # Close figure to free memory

    # Calibration Curves
    fig_cal = plot_calibration_curves(all_labels, all_probs, class_names)
    fig_cal.savefig(os.path.join(FIGURES_DIR, 'calibration_curves_per_class.png'))
    plt.close(fig_cal)
    
    # High-Loss Samples
    fig_loss = plot_highest_loss_samples(all_labels, all_preds, all_losses, all_images, class_names)
    fig_loss.savefig(os.path.join(FIGURES_DIR, 'highest_loss_samples.png'))
    plt.close(fig_loss)

    print(f"All visualizations have been saved into: {FIGURES_DIR}")
