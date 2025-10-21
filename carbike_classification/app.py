"""
Gradio Demo for Cars vs Bike Classification

Self-contained demo that:
1. Preprocesses dataset into a temporary folder.
2. Runs demo training with adjustable hyperparameters.
3. Evaluates the model and visualizes predictions.

Uses temporary directories for all outputs.
"""

import gradio as gr
import random
import tempfile
from pathlib import Path

from carbike_classification.preprocess import main as run_preprocessing
from carbike_classification.train import run_training
from carbike_classification.evaluate_model import (
    load_model_and_data, run_evaluation, get_accuracy,
    plot_confusion_matrix, plot_calibration_curves, plot_highest_loss_samples
)
from carbike_classification.plot_training_curves import plot_training_curves

# -------------------------
# TEMPORARY ENV SETUP
# -------------------------
OUTPUT_DIR = Path(tempfile.mkdtemp())
PROCESSED_DIR = OUTPUT_DIR / "processed"
TB_LOGS_DIR = OUTPUT_DIR / "tb_logs"
MODELS_DIR = OUTPUT_DIR / "models"

for path in [PROCESSED_DIR, TB_LOGS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# -------------------------
# Helper Functions
# -------------------------
def check_data_exists():
    """
    Check if preprocessed data exists.

    Returns
    -------
    bool
        True if training data folder exists, False otherwise.
    """
    return (PROCESSED_DIR / "train").exists()

# -------------------------
# Data Exploration Tab
# -------------------------
def create_data_exploration_tab():
    """
    Create an enhanced Gradio tab for data exploration.

    Allows users to:
    - View basic statistics about the dataset.
    - Visualize random samples by class.
    - Select number of samples and class to display.
    - See histograms of image dimensions (width & height) by class.

    Returns
    -------
    gr.Blocks
        Gradio Blocks object containing the enhanced data exploration interface.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with gr.Blocks() as tab:
        gr.Markdown("## 1. Data Exploration (Enhanced)")
        gr.Markdown(f"Explore random samples and statistics from the dataset (`{PROCESSED_DIR}`).")

        with gr.Row():
            stats_output = gr.JSON(label="Dataset Statistics")

        with gr.Row():
            class_dropdown = gr.Dropdown(["Car", "Bike"], value="Car", label="Select Class")
            num_samples_slider = gr.Slider(1, 20, value=10, step=1, label="Number of Samples")

        with gr.Row():
            gallery = gr.Gallery(label="Random Samples", columns=5, height="auto")

        with gr.Row():
            hist_plot = gr.Plot(label="Image Size Distributions by Class")

        def get_dataset_stats():
            car_path = PROCESSED_DIR / "train" / "Car"
            bike_path = PROCESSED_DIR / "train" / "Bike"

            car_images = [p for p in car_path.glob("*") if p.suffix.lower() in ['.jpg', '.png']]
            bike_images = [p for p in bike_path.glob("*") if p.suffix.lower() in ['.jpg', '.png']]

            total = len(car_images) + len(bike_images)
            stats = {
                "Car": len(car_images),
                "Bike": len(bike_images),
                "Total": total,
                "Percentages": {
                    "Car": round(len(car_images)/total*100, 2) if total else 0,
                    "Bike": round(len(bike_images)/total*100, 2) if total else 0
                }
            }
            return stats

        def get_samples_and_plot(selected_class, num_samples):
            from PIL import Image

            # Get class images for gallery
            class_path = PROCESSED_DIR / "train" / selected_class
            images = [p for p in class_path.glob("*") if p.suffix.lower() in ['.jpg', '.png']]
            np.random.shuffle(images)
            selected_images = images[:num_samples]

            # Collect widths and heights by class
            widths, heights = {"Car": [], "Bike": []}, {"Car": [], "Bike": []}
            for cls in ["Car", "Bike"]:
                cls_path = PROCESSED_DIR / "train" / cls
                for img_path in cls_path.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.png']:
                        with Image.open(img_path) as img:
                            widths[cls].append(img.width)
                            heights[cls].append(img.height)

            # Plot superimposed histograms
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Width histogram
            ax[0].hist(widths["Car"], bins=10, alpha=0.6, label="Car", color="skyblue")
            ax[0].hist(widths["Bike"], bins=10, alpha=0.6, label="Bike", color="orange")
            ax[0].set_title("Width Distribution by Class")
            ax[0].set_xlabel("Width (px)")
            ax[0].set_ylabel("Count")
            ax[0].legend()

            # Height histogram
            ax[1].hist(heights["Car"], bins=10, alpha=0.6, label="Car", color="skyblue")
            ax[1].hist(heights["Bike"], bins=10, alpha=0.6, label="Bike", color="orange")
            ax[1].set_title("Height Distribution by Class")
            ax[1].set_xlabel("Height (px)")
            ax[1].set_ylabel("Count")
            ax[1].legend()

            plt.tight_layout()

            return [str(p) for p in selected_images], fig

        tab.load(get_dataset_stats, outputs=[stats_output])
        class_dropdown.change(
            get_samples_and_plot,
            inputs=[class_dropdown, num_samples_slider],
            outputs=[gallery, hist_plot]
        )
        num_samples_slider.change(
            get_samples_and_plot,
            inputs=[class_dropdown, num_samples_slider],
            outputs=[gallery, hist_plot]
        )

    return tab


# -------------------------
# Training Tab
# -------------------------
# Global variables to hold the latest trained model and data
CURRENT_MODEL = None
CURRENT_VAL_LOADER = None
CURRENT_CLASS_NAMES = None
CURRENT_DEVICE = None

def create_training_tab():
    """
    Create Gradio tab for training interface.

    Provides sliders to adjust hyperparameters and a button to run demo training.

    Returns
    -------
    gr.Blocks
        Gradio Blocks object containing the training interface.
    """
    global CURRENT_MODEL, CURRENT_VAL_LOADER, CURRENT_CLASS_NAMES, CURRENT_DEVICE

    with gr.Blocks() as tab:
        gr.Markdown("## 2. Training Interface (Demo)")
        gr.Markdown("Adjust hyperparameters and run a short training demo.")

        with gr.Row():
            with gr.Column(scale=1):
                lr_slider = gr.Slider(1e-5, 1e-2, value=1e-3, step=1e-5, label="Learning Rate")
                epochs_slider = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                batch_slider = gr.Slider(4, 32, value=8, step=4, label="Batch Size")
                train_button = gr.Button("Run Demo Training", variant="primary")

            with gr.Column(scale=2):
                metrics_output = gr.JSON(label="Validation Metrics")

        def training_wrapper(lr, epochs, batch):
            """
            Wrapper function to run training with given hyperparameters.
            Stores trained model and validation data globally.
            """
            global CURRENT_MODEL, CURRENT_VAL_LOADER, CURRENT_CLASS_NAMES, CURRENT_DEVICE

            gr.Info("Starting demo training... (Check console for progress)")
            # Run training
            val_metrics = run_training(
                epochs=int(epochs),
                learning_rate=float(lr),
                batch_size=int(batch),
                tb_logs_dir=str(TB_LOGS_DIR),
                models_dir=str(MODELS_DIR),
                processed_dir=str(PROCESSED_DIR)
            )

            # Load model and validation data for evaluation
            CURRENT_MODEL, CURRENT_VAL_LOADER, CURRENT_CLASS_NAMES, CURRENT_DEVICE = load_model_and_data(
                processed_dir=PROCESSED_DIR,
                model_path=MODELS_DIR / "car_bike_model.pth"
            )

            gr.Info("Demo training complete.")
            return val_metrics

        train_button.click(
            training_wrapper,
            inputs=[lr_slider, epochs_slider, batch_slider],
            outputs=[metrics_output]
        )

    return tab


# -------------------------
# Evaluation Tab
# -------------------------
def create_evaluation_tab():
    """
    Create Gradio tab for model evaluation.

    Uses the last trained model stored in memory.
    """
    global CURRENT_MODEL, CURRENT_VAL_LOADER, CURRENT_CLASS_NAMES, CURRENT_DEVICE

    with gr.Blocks() as tab:
        gr.Markdown("## 3. Model Evaluation")
        gr.Markdown("Evaluate the last trained model and visualize metrics.")

        eval_button = gr.Button("Run Evaluation", variant="primary")

        accuracy_output = gr.Textbox(label="Validation Accuracy", interactive=False)
        training_plot = gr.Plot(label="Training Curves")
        cm_plot = gr.Plot(label="Confusion Matrix")
        calib_plot = gr.Plot(label="Calibration Curves")
        loss_plot = gr.Plot(label="Highest Loss Samples")

        def evaluation_wrapper():
            if CURRENT_MODEL is None:
                return "No trained model found!", None, None, None, None

            labels, preds, probs, losses, images = run_evaluation(
                CURRENT_MODEL, CURRENT_VAL_LOADER, CURRENT_DEVICE, len(CURRENT_CLASS_NAMES)
            )
            acc = get_accuracy(labels, preds)

            # Training curves still from TB logs
            fig_train, _ = plot_training_curves(log_dir=str(TB_LOGS_DIR / "car_bike_model"))
            fig_cm = plot_confusion_matrix(labels, preds, CURRENT_CLASS_NAMES)
            fig_cal = plot_calibration_curves(labels, probs, CURRENT_CLASS_NAMES)
            fig_loss = plot_highest_loss_samples(labels, preds, losses, images, CURRENT_CLASS_NAMES)

            return f"{acc:.4f}", fig_train, fig_cm, fig_cal, fig_loss

        eval_button.click(
            evaluation_wrapper,
            outputs=[accuracy_output, training_plot, cm_plot, calib_plot, loss_plot]
        )

    return tab



# -------------------------
# Main Gradio App
# -------------------------
def main():
    """
    Launch the Gradio demo application.

    Checks if data exists and runs preprocessing if needed, then
    initializes Gradio tabs for data exploration, training, and evaluation.
    """
    if not check_data_exists():
        print("Preprocessed data not found. Running preprocessing...")
        run_preprocessing(processed_dir=str(PROCESSED_DIR))

    with gr.Blocks(title="Cars vs Bike Classification Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Cars vs Bike Classification Demo")
        gr.Markdown("Explore dataset, train, and evaluate the model interactively.")

        with gr.Tabs():
            with gr.TabItem("Data Exploration"):
                create_data_exploration_tab()
            with gr.TabItem("Training"):
                create_training_tab()
            with gr.TabItem("Evaluation"):
                create_evaluation_tab()

    demo.launch()

if __name__ == "__main__":
    main()
