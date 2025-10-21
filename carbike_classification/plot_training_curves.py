# carbike_classification/plot_training_curves.py
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.figure
from tensorboard.backend.event_processing import event_accumulator
from glob import glob

def events_to_df(events):
    """
    Convert a list of TensorBoard scalar events to a pandas DataFrame.

    Parameters
    ----------
    events : List[tensorboard.EventAccumulator.ScalarEvent]
        List of scalar events extracted from TensorBoard logs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing 'step' and 'value' columns.
    """
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({'step': steps, 'value': values})


def get_latest_version_dir(base_dir: str) -> str:
    """
    Get the latest version_* folder inside a TensorBoard log folder.

    Parameters
    ----------
    base_dir : str
        Base directory containing versioned TensorBoard logs (e.g., tb_logs/car_bike_model).

    Returns
    -------
    str
        Path to the latest version directory.
    """
    version_dirs = sorted(glob(os.path.join(base_dir, 'version_*')), key=os.path.getmtime)
    if not version_dirs:
        raise FileNotFoundError(f"No version directories found in {base_dir}")
    return version_dirs[-1]

def plot_training_curves(log_dir: str = None, figures_dir: str = None) -> tuple:
    """
    Load TensorBoard logs and plot training loss and validation accuracy.

    Parameters
    ----------
    log_dir : str, optional
        TensorBoard log directory containing 'version_*' subfolders.
        Default: last version inside 'tb_logs/car_bike_model' relative to this file.
    figures_dir : str, optional
        Directory to save figures. Default: '../reports/figures' relative to this file.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the plots.
    figures_dir : str
        Directory where figures can be saved.
    """
    # Default log_dir
    if log_dir is None:
        base_tb_logs = os.path.join(os.path.dirname(__file__), 'tb_logs', 'car_bike_model')
        log_dir = get_latest_version_dir(base_tb_logs)
    else:
        # Always pick the latest version inside the folder passed
        log_dir = get_latest_version_dir(log_dir)

    # Default figures_dir
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Load TensorBoard events
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()
    print(f"Available tags in the log ({log_dir}):", tags)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training loss
    if 'train_loss' in tags.get('scalars', []):
        scalars = ea.Scalars('train_loss')
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        ax1.plot(steps, values, label='Train Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss per Epoch')
        ax1.grid(True)
        ax1.legend()
    else:
        ax1.set_title('Training Loss (No Data Found)')
        print("Warning: 'train_loss' not found in logs.")

    # Validation accuracy
    if 'val_acc' in tags.get('scalars', []):
        scalars = ea.Scalars('val_acc')
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        ax2.plot(steps, values, label='Validation Accuracy', color='orange')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy per Epoch')
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.set_title('Validation Accuracy (No Data Found)')
        print("Warning: 'val_acc' not found in logs.")

    plt.tight_layout()
    return fig, figures_dir


if __name__ == '__main__':
    print("Running plot_training_curves.py as a script...")

    # Generate figure using defaults
    training_figure, figures_dir = plot_training_curves()

    # Save the figure inside figures_dir
    save_path = os.path.join(figures_dir, 'training_curves_combined.png')
    training_figure.savefig(save_path)
    plt.close(training_figure)

    print(f"Combined training curves plot saved to: {save_path}")
