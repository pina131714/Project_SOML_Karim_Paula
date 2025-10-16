import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.calibration import CalibrationDisplay
from torchvision import models
from myproject.dataset import CarBikeDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

# === CONFIG ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'car_bike_model.pth')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg11(weights=None)
model.classifier[6] = torch.nn.Linear(4096, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


# === Load data ===
data_module = CarBikeDataModule()
data_module.setup(stage='test')
val_loader = data_module.val_dataloader()

# Get class names directly from dataset
class_names = list(val_loader.dataset.dataset.classes)
num_classes = 2  # Bike and Car


# === Evaluation ===
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


# === Accuracy ===
accuracy = accuracy_score(all_labels, all_preds)
print(f'Validation Accuracy: {accuracy:.4f}')


# === Confusion Matrix ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'))
plt.close()


# === Calibration Curves (one per class) ===
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibrationDisplay

fig, axes = plt.subplots(1, num_classes, figsize=(6*num_classes, 5))

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
plt.savefig(os.path.join(FIGURES_DIR, 'calibration_curves_per_class.png'))
plt.close()


# === Show top-k highest loss samples ===
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

top_k = 5
top_k_indices = np.argsort(all_losses)[-top_k:]

to_pil = ToPILImage()

fig, axs = plt.subplots(1, top_k, figsize=(15, 4))
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
plt.savefig(os.path.join(FIGURES_DIR, 'highest_loss_samples.png'))
plt.close()

print(f"All visualizations have been saved into: {FIGURES_DIR}")
