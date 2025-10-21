# myproject/preprocess.py
import os
import kagglehub
import zipfile
import shutil
import random
from pathlib import Path

# --- Path Configuration ---
# Project root (one level up from 'myproject')
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Target directories (clean train/test structure)
TRAIN_DIR = PROCESSED_DIR / "train"
TEST_DIR = PROCESSED_DIR / "test"
TRAIN_BIKE_DIR = TRAIN_DIR / "Bike"
TRAIN_CAR_DIR = TRAIN_DIR / "Car"
TEST_BIKE_DIR = TEST_DIR / "Bike"
TEST_CAR_DIR = TEST_DIR / "Car"

KAGGLE_DATASET = "utkarshsaxenadn/car-vs-bike-classification-dataset"

def process_class(source_dir: Path, train_dest: Path, test_dest: Path, class_name: str, n_train=40, n_test=20):
    """
    Copies a random selection of files from a class to the train and test folders.
    """
    print(f"Processing class: {class_name}")
    
    # Use .glob to find images (you can add more extensions if needed)
    files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    if len(files) < (n_train + n_test):
        print(f"  Warning: Not enough images for {class_name}. "
              f"Found {len(files)}, needed {n_train + n_test}.")
        return

    # Random shuffle for selection
    random.seed(42) # For reproducibility
    random.shuffle(files)
    
    # File selection
    selected_files = files[:(n_train + n_test)]
    train_files = selected_files[:n_train]
    test_files = selected_files[n_train:]

    # Copy to 'train'
    for f in train_files:
        shutil.copy(f, train_dest / f.name)
    
    # Copy to 'test'
    for f in test_files:
        shutil.copy(f, test_dest / f.name)
        
    print(f"  Class {class_name}: {len(train_files)} train, {len(test_files)} test.")

# myproject/preprocess.py

# ... (keep all your imports and the 'process_class' function) ...

def main():
    """
    Main script to download and process the dataset.
    (Refactored to read directly from Kaggle cache)
    """
    print("Starting data preprocessing...")
    
    # 1. Ensure Kaggle auth is ready
    try:
        kagglehub.login() # Ensures you are authenticated
    except Exception as e:
        print(f"Error authenticating with Kaggle: {e}")
        print("Please ensure your 'kaggle.json' is in '~/.kaggle/kaggle.json'")
        return

    # 2. Clean up old processed directory
    if PROCESSED_DIR.exists():
        print(f"Removing old 'processed' directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)
    
    # 3. Create new destination structure
    for path in [TRAIN_BIKE_DIR, TRAIN_CAR_DIR, TEST_BIKE_DIR, TEST_CAR_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    print(f"Train/test directories created in: {PROCESSED_DIR}")

    # 4. Download AND extract the dataset to cache
    print(f"Downloading and extracting dataset '{KAGGLE_DATASET}' to cache...")
    
    # This call downloads AND extracts.
    # It returns the path to the EXTRACTED dataset directory.
    cached_dataset_path_str = kagglehub.dataset_download(KAGGLE_DATASET)
    cached_dataset_path = Path(cached_dataset_path_str)
    
    print(f"Dataset available in cache at: {cached_dataset_path}")

    # 5. Find the source data (it's inside the cached path)
    # This path was correct in your original dataset.py
    source_data_dir = cached_dataset_path / "Car-Bike-Dataset" 
    source_bike_dir = source_data_dir / "Bike"
    source_car_dir = source_data_dir / "Car"
    
    if not source_data_dir.exists() or not source_bike_dir.exists() or not source_car_dir.exists():
        print(f"Error: Dataset structure not as expected.")
        print(f"Was looking for 'Car-Bike-Dataset' folder inside: {cached_dataset_path}")
        return
        
    # 6. Process and copy files from cache to PROCESSED_DIR
    print("Copying and splitting files from cache to 'data/processed'...")
    process_class(source_bike_dir, TRAIN_BIKE_DIR, TEST_BIKE_DIR, "Bike")
    process_class(source_car_dir, TRAIN_CAR_DIR, TEST_CAR_DIR, "Car")
    
    print("\nPreprocessing complete.")

if __name__ == "__main__":
    main()
