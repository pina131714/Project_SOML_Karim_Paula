## Cars vs Bike image classification
This project's objective is to classify color images of bikes and cars using a pre-trained neural netwrok (VGG11).

This project has been packaged as a PyPI module and includes an interactive Gradio demo.

## Requisites
- Python dependencies are listed in 'pyproject.toml'.

How to sync environment with uv on Linux terminal:
1) Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2) Restart terminal
3) Open terminal and change directory into project folder
4) Run: uv sync

## Data source
The dataset is provided by Kaggle.
- Link: https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset
- Original raw data: 'data/raw' (zip file)
- Subsampled data used for training: 'data/processed'

## How to run the code
1) Train model: Run the train.py script located in 'myproject/'.
2) Evaluate model and generate figures: Run the evaluate_model.py script located in 'myproject/'.
3) Plot training curves (loss & accuracy): Run the plot_training_curves.py script located in 'myproject/'.

Terminal commands:
uv run python -m myproject.preprocess
uv run python -m myproject.train
uv run -m myproject.evaluate_model
uv run -m myproject.plot_training_curves

uv run python -m myproject.app