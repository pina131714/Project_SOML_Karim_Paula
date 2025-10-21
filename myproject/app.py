# myproject/app.py
import gradio as gr
import os
import sys
import random
from pathlib import Path

# --- 1. Import necessary functions and paths ---
try:
    # Import the 'main' from preprocess AND RENAME IT
    from myproject.preprocess import PROCESSED_DIR, main as run_preprocessing
    
    # Import the absolute LOG_DIR (for the evaluation tab)
    # (This assumes myproject/paths.py exists)
    from myproject.paths import LOG_DIR 
except ImportError as e:
    print(f"Fatal Error: Cannot import from myproject modules: {e}", file=sys.stderr)
    print("Ensure 'myproject/paths.py' exists and all scripts use absolute paths.")
    sys.exit(1)

# Import the rest of the functions
from myproject.train import run_training
from myproject.evaluate_model import (
    load_model_and_data, run_evaluation, get_accuracy,
    plot_confusion_matrix, plot_calibration_curves, plot_highest_loss_samples
)
from myproject.plot_training_curves import plot_training_curves

# --- 2. Data Check Function ---
def check_data_exists():
    """Checks if the preprocessed data exists."""
    train_path = PROCESSED_DIR / "train"
    return train_path.exists()

# --- 3. Tab Definitions ---

def create_data_exploration_tab():
    """Creates the Data Exploration tab."""
    with gr.Blocks() as tab:
        gr.Markdown("## 1. Data Exploration")
        gr.Markdown(f"Showing random samples from the preprocessed dataset (`{PROCESSED_DIR}`).")
        
        with gr.Row():
            car_gallery = gr.Gallery(label="Sample 'Car' Images (Train Set)", elem_id="car_gallery", columns=5, height="auto")
            bike_gallery = gr.Gallery(label="Sample 'Bike' Images (Train Set)", elem_id="bike_gallery", columns=5, height="auto")

        def get_samples():
            car_path = PROCESSED_DIR / "train" / "Car"
            bike_path = PROCESSED_DIR / "train" / "Bike"
            car_images = [str(p) for p in car_path.glob("*") if p.suffix in ['.jpg', '.png']]
            bike_images = [str(p) for p in bike_path.glob("*") if p.suffix in ['.jpg', '.png']]
            random.shuffle(car_images)
            random.shuffle(bike_images)
            return car_images[:10], bike_images[:10]

        tab.load(get_samples, outputs=[car_gallery, bike_gallery])
    return tab

def create_training_tab():
    """Creates the Training Interface tab."""
    with gr.Blocks() as tab:
        gr.Markdown("## 2. Training Interface (Demo)")
        gr.Markdown("Run a short training session with custom hyperparameters.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Hyperparameters")
                lr_slider = gr.Slider(minimum=1e-5, maximum=1e-2, value=1e-3, label="Learning Rate", step=1e-5, interactive=True)
                epochs_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Epochs (Demo)", interactive=True)
                batch_slider = gr.Slider(minimum=4, maximum=32, value=8, step=4, label="Batch Size", interactive=True)
                train_button = gr.Button("Start Demo Training", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### Training Results")
                output_metrics = gr.JSON(label="Final Validation Metrics")

        def training_wrapper(lr, epochs, b_size):
            gr.Info("Starting demo training... (Check console for progress)")
            metrics = run_training(int(epochs), lr, int(b_size)) 
            gr.Info("Demo training complete.")
            return metrics

        train_button.click(
            fn=training_wrapper,
            inputs=[lr_slider, epochs_slider, batch_slider],
            outputs=[output_metrics]
        )
    return tab

def create_evaluation_tab():
    """Creates the Model Evaluation tab."""
    with gr.Blocks() as tab:
        gr.Markdown("## 3. Model Evaluation")
        gr.Markdown("Evaluate the **best trained model** and load training curves.")
        
        eval_button = gr.Button("Load & Run Evaluation", variant="primary")
        
        gr.Markdown("### Performance Metrics")
        accuracy_output = gr.Textbox(label="Validation Accuracy (from best model)", interactive=False)
        
        gr.Markdown(f"### Training History (from main log)")
        training_plot_output = gr.Plot(label="Training/Validation Curves")

        gr.Markdown("### Evaluation Plots (from best model)")
        with gr.Row():
            cm_plot_output = gr.Plot(label="Confusion Matrix")
            calib_plot_output = gr.Plot(label="Calibration Curves")
        
        gr.Markdown("### Error Analysis (from best model)")
        loss_plot_output = gr.Plot(label="Highest Loss Samples")

        def evaluation_wrapper():
            gr.Info("Loading 'best model' and data...")
            model, val_loader, class_names, device = load_model_and_data()
            
            gr.Info("Running evaluation...")
            all_labels, all_preds, all_probs, all_losses, all_images = run_evaluation(
                model, val_loader, device, len(class_names)
            )
            
            gr.Info("Generating plots...")
            acc = get_accuracy(all_labels, all_preds)
            fig_train = plot_training_curves() # Uses default MAIN_LOG_DIR
            fig_cm = plot_confusion_matrix(all_labels, all_preds, class_names)
            fig_cal = plot_calibration_curves(all_labels, all_probs, class_names)
            fig_loss = plot_highest_loss_samples(all_labels, all_preds, all_losses, all_images, class_names)
            
            gr.Info("Evaluation complete.")
            return f"{acc:.4f}", fig_train, fig_cm, fig_cal, fig_loss

        eval_button.click(
            fn=evaluation_wrapper,
            outputs=[
                accuracy_output,
                training_plot_output,
                cm_plot_output,
                calib_plot_output,
                loss_plot_output
            ]
        )
    return tab


# --- 4. Main Function (Entry Point) ---

def main():
    """
    Main entry point. Checks data, runs preprocess if needed, 
    and launches the Gradio app.
    """
    
    # --- THIS IS THE LOGIC YOU REQUESTED ---
    # Step 1: Check if data exists
    if not check_data_exists():
        print("--------------------------------------------------")
        print("Processed data not found.")
        print("Running one-time data preprocessing...")
        print(f"Data will be installed in: {PROCESSED_DIR}")
        print("This may take a moment and will require Kaggle login...")
        print("--------------------------------------------------")
        
        try:
            # Call the main() function from preprocess.py
            run_preprocessing() 
        except Exception as e:
            print(f"ERROR: Data preprocessing failed: {e}", file=sys.stderr)
            print("Please try running 'python -m myproject.preprocess' manually.", file=sys.stderr)
            sys.exit(1)
        
        # Re-check after running
        if not check_data_exists():
            print("ERROR: Preprocessing ran, but data is still not found.", file=sys.stderr)
            sys.exit(1)
        
        print("Data preprocessing successful.")
    # --- END OF CHECKING LOGIC ---

    # Step 2: Build the interface
    with gr.Blocks(title="Project SOML: Car v Bike", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Project SOML: Car vs Bike Classification")
        gr.Markdown("By Karim Abushams & Paula Pina")
        
        with gr.Tabs():
            with gr.TabItem("1. Data Exploration"):
                create_data_exploration_tab()
            with gr.TabItem("2. Model Evaluation"):
                create_evaluation_tab()
            with gr.TabItem("3. Training (Demo)"):
                create_training_tab()
                
    # Step 3: Launch the app
    print("Launching Gradio demo... Access it at http://127.0.0.1:7860")
    demo.launch()

if __name__ == "__main__":
    main()
