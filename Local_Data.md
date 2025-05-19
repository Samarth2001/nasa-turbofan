# Using Local NASA Turbofan Dataset Files

This guide explains how to use your local NASA Turbofan dataset files directly with the project.

## Local File Structure

Your local dataset appears to have the following files:
- `train_FD001`, `train_FD002`, `train_FD003`, `train_FD004` (training datasets)
- `test_FD001`, `test_FD002`, `test_FD003`, `test_FD004` (test datasets)
- `RUL_FD001`, `RUL_FD002`, `RUL_FD003`, `RUL_FD004` (RUL values for test)
- `readme` (documentation)
- `Damage Propagation Modeling` (PDF document)

## Setup Instructions

### Option 1: Update the Config File

1. Open `src/config.py` and modify the `LOCAL_DATA_DIR` variable to point to the directory containing your dataset files:

```python
# Change this to your local directory with the files
LOCAL_DATA_DIR = Path("/path/to/your/nasa_dataset") 
```

2. Save the file and run the training script normally:

```bash
python train.py --subset FD001
```

### Option 2: Use Command Line Arguments

You can specify the data directory directly when running the training script:

```bash
python train.py --subset FD001 --data-dir /path/to/your/nasa_dataset
```

## Additional Options

- `--subset`: Specify which dataset subset to use (FD001, FD002, FD003, or FD004)
- `--force-preprocess`: Force data preprocessing even if processed data exists
- `--download`: Use the download mode instead of local files (in case you want to use the original download mechanism)
- `--batch-size`: Set the batch size for training (default: 64)
- `--epochs`: Set the maximum number of training epochs (default: 100)
- `--patience`: Set the patience for early stopping (default: 10)
- `--learning-rate`: Set the learning rate for the optimizer (default: 0.001)
- `--lstm-units`: Set the number of units in LSTM layers (default: 64)
- `--skip-explainability`: Skip the explainability analysis

## Example Command

```bash
python train.py --data-dir /path/to/your/nasa_dataset --subset FD001 --epochs 50 --batch-size 32
```

## Note on File Format

The code has been updated to work with your local files directly, without requiring file extensions (.txt). It will automatically detect and use the files as shown in your directory structure.