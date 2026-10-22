from pathlib import Path

IMAGE_SIZE = (32, 32)
IMAGE_SHAPE = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
MODEL_DIR = ROOT_DIR / "saved_models"
