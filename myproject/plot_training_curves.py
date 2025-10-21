import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.figure
from tensorboard.backend.event_processing import event_accumulator

# === Paths ===
# NOTE: This assumes a specific log version. 
# For Gradio, you might want to find the 'latest' version dynamically.
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'tb_logs', 'car_bike_model', 'version_0')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


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
    
    Examples
    --------
    >>> df = events_to_df(ea.Scalars('train_loss'))
    >>> df.head()
       step   value
    0     1  0.6932
    1     2  0.5874
    ...
    """
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return pd.DataFrame({'step': steps, 'value': values})


def plot_training_curves(log_dir=LOG_DIR) -> matplotlib.figure.Figure:
    """
    Loads TensorBoard logs and generates a single figure with training 
    loss and validation accuracy plots.

    Args:
        log_dir (str): Path to the specific TensorBoard log directory (e.g., .../version_0).

    Returns:
        matplotlib.figure.Figure: A figure object containing the plots.
    """
    
    # === Load TensorBoard events ===
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    tags = ea.Tags()
    print("Available tags in the log:", tags) # Keep console output for debugging

    # --- Create a single figure with two subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # === Plot training loss ===
    if 'train_loss' in tags.get('scalars', []):
        train_loss_events = ea.Scalars('train_loss')
        df_train_loss = events_to_df(train_loss_events)

        ax1.plot(df_train_loss['step'], df_train_loss['value'], label='Train Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss per Epoch')
        ax1.grid(True)
        ax1.legend()
    else:
        ax1.set_title('Training Loss (No Data Found)')
        print("Warning: 'train_loss' not found in logs.")

    # === Plot validation accuracy ===
    if 'val_acc' in tags.get('scalars', []):
        val_acc_events = ea.Scalars('val_acc')
        df_val_acc = events_to_df(val_acc_events)

        ax2.plot(df_val_acc['step'], df_val_acc['value'], label='Validation Accuracy', color='orange')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy per Epoch')
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.set_title('Validation Accuracy (No Data Found)')
        print("Warning: 'val_acc' not found in logs.")

    plt.tight_layout()
    return fig # Return the figure object


if __name__ == '__main__':
    
    print("Running plot_training_curves.py as a script...")
    
    # Generate the plot
    training_figure = plot_training_curves()
    
    # Save the generated figure (original script behavior)
    save_path = os.path.join(FIGURES_DIR, 'training_curves_combined.png')
    training_figure.savefig(save_path)
    plt.close(training_figure) # Close figure to free memory
    
    print(f"Combined training curves plot saved to: {save_path}")
