# carbike_classification/preprocess.py
import kagglehub
import shutil
import random
from pathlib import Path

def process_class(source_dir: Path, train_dest: Path, test_dest: Path, class_name: str, n_train=40, n_test=20):
    """Copy and split images for a class into train and test folders."""
    files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    if len(files) < (n_train + n_test):
        print(f"Not enough images for {class_name}. Found {len(files)}.")
        return
    random.shuffle(files)
    selected_files = files[:n_train + n_test]
    for f in selected_files[:n_train]:
        shutil.copy(f, train_dest / f.name)
    for f in selected_files[n_train:]:
        shutil.copy(f, test_dest / f.name)

def main(processed_dir=None):
    """
    Download dataset from Kaggle and split into train/test.

    Args:
        processed_dir (str or Path, optional): Directory to save processed data.
            Defaults to '../data/processed' relative to this script.
    """
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    else:
        processed_dir = Path(processed_dir)

    train_dir = processed_dir / "train"
    test_dir = processed_dir / "test"
    TRAIN_BIKE_DIR = train_dir / "Bike"
    TRAIN_CAR_DIR = train_dir / "Car"
    TEST_BIKE_DIR = test_dir / "Bike"
    TEST_CAR_DIR = test_dir / "Car"

    for path in [TRAIN_BIKE_DIR, TRAIN_CAR_DIR, TEST_BIKE_DIR, TEST_CAR_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(kagglehub.dataset_download("utkarshsaxenadn/car-vs-bike-classification-dataset"))
    
    source_data_dir = dataset_path / "Car-Bike-Dataset"
    
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Expected dataset folder not found in {dataset_path}")

    process_class(source_data_dir / "Bike", TRAIN_BIKE_DIR, TEST_BIKE_DIR, "Bike")
    process_class(source_data_dir / "Car", TRAIN_CAR_DIR, TEST_CAR_DIR, "Car")

    print(f"Data preprocessing complete. Train/Test split saved in: {processed_dir}")


if __name__ == "__main__":
    main()
