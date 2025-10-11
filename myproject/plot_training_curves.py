import os
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# === Paths ===
LOG_DIR = os.path.join(os.path.dirname(__file__), 'tb_logs', 'car_bike_model', 'version_0')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# === Load TensorBoard events ===
ea = event_accumulator.EventAccumulator(LOG_DIR)
ea.Reload()

# Print available scalar tags (e.g., 'train_loss', 'val_acc')
tags = ea.Tags()
print("Available tags in the log:", tags)


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


# === Plot training loss ===
if 'train_loss' in tags.get('scalars', []):
    train_loss_events = ea.Scalars('train_loss')
    df_train_loss = events_to_df(train_loss_events)

    plt.figure(figsize=(8, 5))
    plt.plot(df_train_loss['step'], df_train_loss['value'], label='Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'train_loss_curve.png'))
    plt.close()
    print("Train loss curve saved.")
else:
    print("Warning: 'train_loss' not found in logs. No train loss curve will be plotted.")


# === Plot validation accuracy ===
if 'val_acc' in tags.get('scalars', []):
    val_acc_events = ea.Scalars('val_acc')
    df_val_acc = events_to_df(val_acc_events)

    plt.figure(figsize=(8, 5))
    plt.plot(df_val_acc['step'], df_val_acc['value'], label='Validation Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'val_accuracy_curve.png'))
    plt.close()
    print("Validation accuracy curve saved.")
else:
    print("Warning: 'val_acc' not found in logs. No validation accuracy curve will be plotted.")

