"""
Configuration settings for the NASA Turbofan Predictive Maintenance project.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"

# Local data directory - CHANGE THIS TO YOUR LOCAL DIRECTORY WITH THE FILES
# For example: Path("/path/to/your/nasa_dataset")
LOCAL_DATA_DIR = (
    DATA_DIR / "CMAPSSData"
)  # Change this path to point to your local directory with the files

MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset URLs (used only if local files not found)
DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DATASET_FILENAME = "CMAPSS_Data.zip"

# Dataset subdirectories
RAW_DATA_DIR = LOCAL_DATA_DIR  # Use the local directory
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model parameters
SEQUENCE_LENGTH = 50
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Feature configuration
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]  # 21 sensors
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]  # 3 operational settings
TARGET_COLUMN = "RUL"
TIME_COLUMN = "time"
ID_COLUMN = "unit_id"

# Features to use (after feature selection)
FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
    "s_2",
    "s_3",
    "s_4",
    "s_7",
    "s_8",
    "s_9",
    "s_11",
    "s_12",
    "s_13",
    "s_14",
    "s_15",
    "s_17",
    "s_20",
    "s_21",
]

# Dashboard configuration
DASHBOARD_TITLE = "NASA Turbofan Predictive Maintenance"
RUL_THRESHOLD_WARNING = 50
RUL_THRESHOLD_CRITICAL = 20
