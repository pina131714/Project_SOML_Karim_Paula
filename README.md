# Cars vs Bike image classification
This project's objective is to classify color images of bikes and cars using a pre-trained neural netwrok (VGG11).
It has been packaged as a Gradio demo interface and uploaded to TestPyPi for easy visualization.
TestPyPI link: https://test.pypi.org/project/carbike-classification/#description
Documentation link: https://pina131714.github.io/Project_SOML_Karim_Paula/

## Interactive demo from TestPyPI
To run from TestPyPI, we run the following command: 
```bash
uvx --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ carbike-classification
```
This will download the package from TestPyPI and run the app.py interface directly, per the pyproject.toml (project.scripts).
For this command to work, you need to have uv installed, available in: https://docs.astral.sh/uv/

## Requisites
- Python dependencies are listed in 'pyproject.toml'.

How to sync environment with uv on Linux terminal:
1) Install uv: 
``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2) Restart terminal
3) Open terminal and change directory into project folder
4) Run: 
``` bash
uv sync
```
5) (Optional but recommended) Activate the uv environment on terminal: 
``` bash
source .venv/bin/activate
```

## Data source
The dataset is provided by Kaggle.
- Link: https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset
- Subsampled the data used for training. 80 images for train (40 bike, 40 car) and 20 for test (10 bike and 10 car). Data source located in: 'data/processed'

## How to run the code with scripts
0) Preprocess data: Run preprocess.py script located in 'carbike_classification/'. 
``` bash
uv run python -m carbike_classification.preprocess
```
1) Train model: Run the train.py script located in 'carbike_classification/'. 
``` bash
uv run python -m carbike_classification.train
```
2) Evaluate model and generate figures: Run the evaluate_model.py script located in 'carbike_classification/'. 
``` bash
uv run -m carbike_classification.evaluate_model
```
3) Plot training curves (loss & accuracy): Run the plot_training_curves.py script located in 'carbike_classification/'. 
``` bash
uv run -m carbike_classification.plot_training_curves
```
4) Run Gradio interface demo: Run the app.py script located in 'carbike_classification/'. 
``` bash
uv run -m carbike_classification.app
```



