"""
Main script to train the RUL prediction model on the NASA Turbofan dataset.
"""

import argparse
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf
import os

# Suppress TensorFlow oneDNN and INFO logs for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 0 = all logs, 1 = INFO muted, 2 = WARN muted, 3 = ERROR muted

from src.data.data_loader import CMAPSSDataLoader
from src.data.preprocessor import CMAPSSPreprocessor
from src.models.lstm_model import RULPredictor
from src.models.explainer import LSTMExplainer
from src.utils.metrics import evaluate_predictions
from src.config import (
    DATA_DIR,
    LOCAL_DATA_DIR,
    MODELS_DIR,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RUL prediction model on NASA Turbofan dataset"
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Dataset subset to use (default: FD001)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the dataset files (default: config.LOCAL_DATA_DIR)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use download mode instead of local files",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Force preprocessing of data even if processed data exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum number of epochs for training (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Patience for early stopping (default: {EARLY_STOPPING_PATIENCE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate for optimizer (default: {LEARNING_RATE})",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=64,
        help="Number of units in LSTM layers (default: 64)",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip all explainability analysis (attention and SHAP)",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explanation analysis (use attention only if explainability is not skipped)",
    )
    parser.add_argument(
        "--force-kernel-explainer",
        action="store_true",
        help="Force using KernelExplainer instead of DeepExplainer for SHAP (slower but more compatible)",
    )

    return parser.parse_args()


def main():
    """Main function to run the training pipeline."""
    try:
        # Parse arguments
        args = parse_args()

        # Set data directory
        data_dir = Path(args.data_dir) if args.data_dir else LOCAL_DATA_DIR
        logger.info(f"Using data directory: {data_dir}")

        # Create directories if they don't exist
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        MODELS_DIR.mkdir(exist_ok=True, parents=True)

        # Step 1: Load data
        logger.info(f"Loading dataset subset {args.subset}...")
        data_loader = CMAPSSDataLoader(data_dir=data_dir)

        # Determine whether to use local files or download
        use_local_files = not args.download
        if use_local_files:
            logger.info("Using local files mode")
        else:
            logger.info("Using download mode")
            data_loader.download_dataset(force_download=True)
            data_loader.extract_dataset()

        train_df, test_df, test_rul = data_loader.load_dataset(
            args.subset, use_local_files=use_local_files
        )
        logger.info(
            f"Dataset loaded: {len(train_df)} training samples, {len(test_df)} test samples"
        )

        # Step 2: Preprocess data
        logger.info("Preprocessing data...")
        preprocessor = CMAPSSPreprocessor()

        # Check if processed data exists
        processed_data_files = [
            DATA_DIR / "processed" / "X_train.npy",
            DATA_DIR / "processed" / "y_train.npy",
            DATA_DIR / "processed" / "X_val.npy",
            DATA_DIR / "processed" / "y_val.npy",
            DATA_DIR / "processed" / "X_test.npy",
            DATA_DIR / "processed" / "y_test.npy",
            DATA_DIR / "processed" / "engine_ids.npy",
        ]

        if all(f.exists() for f in processed_data_files) and not args.force_preprocess:
            logger.info("Loading existing processed data...")
            processed_data = {
                "X_train": np.load(DATA_DIR / "processed" / "X_train.npy"),
                "y_train": np.load(DATA_DIR / "processed" / "y_train.npy"),
                "X_val": np.load(DATA_DIR / "processed" / "X_val.npy"),
                "y_val": np.load(DATA_DIR / "processed" / "y_val.npy"),
                "X_test": np.load(DATA_DIR / "processed" / "X_test.npy"),
                "y_test": np.load(DATA_DIR / "processed" / "y_test.npy"),
                "engine_ids": np.load(DATA_DIR / "processed" / "engine_ids.npy"),
            }
        else:
            logger.info("Processing data...")
            processed_data = preprocessor.process_data(train_df, test_df, test_rul)

        # Step 3: Build and train model
        logger.info("Building model...")
        model = RULPredictor(
            sequence_length=SEQUENCE_LENGTH,
            n_features=processed_data["X_train"].shape[2],
            lstm_units=args.lstm_units,
            learning_rate=args.learning_rate,
        )

        model.build_model()

        logger.info("Training model...")
        history = model.train(
            processed_data["X_train"],
            processed_data["y_train"],
            processed_data["X_val"],
            processed_data["y_val"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
        )

        # Step 4: Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate(processed_data["X_test"], processed_data["y_test"])

        # Make predictions
        y_pred = model.predict(processed_data["X_test"])

        # Evaluate predictions
        evaluation = evaluate_predictions(
            processed_data["y_test"],
            y_pred,
            processed_data["engine_ids"],
            output_dir=model.model_dir / "evaluation",
        )

        # Step 5: Generate explanations
        if not args.skip_explainability:
            logger.info("Generating model explanations...")
            explainer = LSTMExplainer(model, output_dir=model.model_dir / "explainability")

            # Determine background data for SHAP
            shap_background_data = processed_data["X_train"][:100]
            if args.skip_shap:
                logger.info("SHAP explanations will be skipped as per --skip-shap flag.")
                shap_background_data = None
            
            # Pass force-kernel-explainer option to the explainer
            if args.force_kernel_explainer and not args.skip_shap:
                logger.info("Will use KernelExplainer instead of DeepExplainer for SHAP.")
                # We'll modify the explainer class to handle this parameter
                explainer.use_kernel_explainer = True

            explanations = explainer.explain_prediction(
                processed_data["X_test"][:100],  # Limit to 100 test examples
                background_data=shap_background_data,
            )

            logger.info("Explanations generated successfully")

        logger.info(
            f"Model training and evaluation completed successfully. Model saved to {model.model_dir}"
        )
        logger.info("To launch the dashboard, run: streamlit run dashboard/app.py")

    except FileNotFoundError as e:
        logger.error(f"File not found during training pipeline: {e}")
        logger.error("Please ensure all necessary data and model paths are correct.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the training pipeline: {e}", exc_info=True)
        # exc_info=True will log the full traceback


if __name__ == "__main__":
    main()
