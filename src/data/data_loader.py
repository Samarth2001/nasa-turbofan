"""
Data loader for NASA Turbofan Engine Degradation Simulation dataset.
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import logging

from src.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    DATASET_URL,
    DATASET_FILENAME,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    TIME_COLUMN,
    ID_COLUMN,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CMAPSSDataLoader:
    """
    Data loader for the NASA CMAPSS Turbofan Engine Degradation Simulation dataset.

    This class provides functionality to load the dataset from local files or
    download and extract if needed.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory where the data files are located. If None, uses RAW_DATA_DIR from config.
        """
        self.data_dir = data_dir if data_dir is not None else RAW_DATA_DIR
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.dataset_path = self.data_dir / DATASET_FILENAME

        # Column names for the dataset
        self.columns = [ID_COLUMN, TIME_COLUMN, *SETTING_COLUMNS, *SENSOR_COLUMNS]

    def download_dataset(self, force_download: bool = False) -> None:
        """
        Download the CMAPSS dataset if it doesn't exist locally.

        Args:
            force_download: If True, download the dataset even if it already exists
        """
        if self.dataset_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {self.dataset_path}")
            return

        logger.info(f"Downloading dataset from {DATASET_URL} to {self.dataset_path}")
        try:
            urllib.request.urlretrieve(DATASET_URL, self.dataset_path)
            logger.info("Download completed successfully")
        except Exception as e:
            logger.error(f"Failed to download dataset: {str(e)}")
            raise

    def extract_dataset(self) -> None:
        """Extract the dataset zip file."""
        if not self.dataset_path.exists():
            logger.error(f"Dataset file not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        logger.info(f"Extracting dataset to {self.data_dir}")
        try:
            with zipfile.ZipFile(self.dataset_path, "r") as zip_ref:
                # Extract all members. The dataset might be nested, e.g., an outer zip
                # containing a directory which holds another zip (like "CMAPSSData.zip").
                zip_ref.extractall(self.data_dir)
                logger.info(f"Initial extraction to {self.data_dir} completed.")

            created_dirs = [item for item in self.data_dir.iterdir() if item.is_dir()]

            # Attempt to find and extract a known nested zip file ("CMAPSSData.zip")
            # if the primary text files are not found directly after initial extraction.
            nested_zip_path = None
            for created_dir_path in created_dirs:
                potential_nested_zip = created_dir_path / "CMAPSSData.zip"
                if potential_nested_zip.exists():
                    logger.info(f"Found nested CMAPSSData.zip at {potential_nested_zip}")
                    nested_zip_path = potential_nested_zip
                    break
                # Check if data files are already in a subdirectory from the initial extraction.
                if (created_dir_path / "train_FD001.txt").exists():
                    logger.info(f"Found .txt files in {created_dir_path} from initial extraction.")
                    pass # _find_data_paths will attempt to locate these.

            if nested_zip_path:
                logger.info(f"Extracting nested zip {nested_zip_path} into {self.data_dir}...")
                try:
                    with zipfile.ZipFile(nested_zip_path, "r") as nested_zip_ref:
                        nested_zip_ref.extractall(self.data_dir) # Extract to the main data_dir
                    logger.info(f"Nested zip extraction to {self.data_dir} completed.")
                except Exception as e_nested:
                    logger.error(f"Failed to extract nested zip {nested_zip_path}: {str(e_nested)}")
                    # Continue, _find_data_paths might still find files.

            # Log final directory contents for debugging if needed
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Final contents of extraction directory ({self.data_dir}):")
                for item in self.data_dir.iterdir():
                    logger.debug(f"  {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
                    if item.is_dir():
                        for sub_item in item.iterdir():
                             if sub_item.is_file():
                                logger.debug(f"    [FILE] {sub_item.name}")

        except Exception as e:
            logger.error(f"Failed to extract dataset: {str(e)}")
            raise

    def check_local_files(self, subset: str = "FD001") -> bool:
        """
        Check if the required local files exist.

        Args:
            subset: Dataset subset to check ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            True if all required files exist, False otherwise
        """
        train_file = self.data_dir / f"train_{subset}"
        test_file = self.data_dir / f"test_{subset}"
        rul_file = self.data_dir / f"RUL_{subset}"

        return train_file.exists() and test_file.exists() and rul_file.exists()

    def load_dataset(self, subset: str, use_local_files: bool = True):
        """
        Loads a specific subset of the CMAPSS dataset.

        Args:
            subset (str): The dataset subset to load (e.g., "FD001").
            use_local_files (bool): If True, tries to load from local files first.
                                    If False, assumes download/extraction was handled by caller.

        Returns:
            tuple: (train_df, test_df, test_rul_df) DataFrames.
        """
        logger.info(f"Attempting to load dataset subset {subset} from {self.data_dir}")

        def _find_data_paths(base_dir: Path, current_subset: str, dataset_zip_filename: str):
            """Helper to find dataset text files, checking direct and common subdir paths."""
            direct_train_path = base_dir / f"train_{current_subset}.txt"
            direct_test_path = base_dir / f"test_{current_subset}.txt"
            direct_rul_path = base_dir / f"RUL_{current_subset}.txt"

            if direct_train_path.exists() and direct_test_path.exists() and direct_rul_path.exists():
                logger.info(f"Found dataset files directly in {base_dir}")
                return direct_train_path, direct_test_path, direct_rul_path

            # Check common subdirectory (e.g., CMAPSS_Data if zip is CMAPSS_Data.zip)
            potential_subdir_name = Path(dataset_zip_filename).stem
            subdir_path = base_dir / potential_subdir_name
            
            subdir_train_path = subdir_path / f"train_{current_subset}.txt"
            subdir_test_path = subdir_path / f"test_{current_subset}.txt"
            subdir_rul_path = subdir_path / f"RUL_{current_subset}.txt"

            if subdir_train_path.exists() and subdir_test_path.exists() and subdir_rul_path.exists():
                logger.info(f"Found dataset files in subdirectory: {subdir_path}")
                return subdir_train_path, subdir_test_path, subdir_rul_path
            
            logger.warning(
                f"Dataset files (e.g., train_{current_subset}.txt) not found directly in {base_dir} "
                f"or in common subdirectory {subdir_path}."
            )
            return None, None, None

        train_path, test_path, rul_path = _find_data_paths(self.data_dir, subset, DATASET_FILENAME)

        if not train_path: # Files were not initially found
            if use_local_files: # True if --download was NOT specified; try to get files
                logger.info(
                    f"Local files for {subset} not found. Attempting to download and extract dataset."
                )
                try:
                    self.download_dataset(force_download=False) # Don't force if zip exists
                    self.extract_dataset()
                    # After download/extract, try to find paths again
                    logger.info(f"Re-checking for dataset files for {subset} after download/extraction.")
                    train_path, test_path, rul_path = _find_data_paths(self.data_dir, subset, DATASET_FILENAME)
                except Exception as e:
                    logger.error(f"An error occurred during download/extraction: {str(e)}")
                    # train_path will remain None, leading to the FileNotFoundError below
            # If use_local_files is False, it means --download was specified.
            # train.py should have handled download/extract. If files are still not found,
            # it's a problem with the downloaded/extracted content or paths.

        # Final check for files
        if not train_path or not test_path or not rul_path:
            error_msg = (
                f"Dataset files for {subset} (e.g., train_{subset}.txt) not found in {self.data_dir} "
                f"or its common subdirectory {self.data_dir / Path(DATASET_FILENAME).stem}, "
                "even after download/extraction attempts. Please check dataset integrity and paths."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Proceeding to load training data from {train_path}")
        train_df = pd.read_csv(
            train_path, sep="\s+", header=None, names=self.columns, index_col=False
        )
        # train_df = train_df.drop(
        #     columns=[col for col in train_df.columns if "Unnamed" in str(col)]
        # )

        logger.info(f"Proceeding to load test data from {test_path}")
        test_df = pd.read_csv(
            test_path, sep="\s+", header=None, names=self.columns, index_col=False
        )
        # test_df = test_df.drop(
        #     columns=[col for col in test_df.columns if "Unnamed" in str(col)]
        # )

        logger.info(f"Proceeding to load RUL data from {rul_path}")
        rul = pd.read_csv(rul_path, header=None).squeeze()

        logger.info(
            f"Dataset loaded successfully: {len(train_df)} training samples, {len(test_df)} test samples"
        )

        return train_df, test_df, rul

    def load_all_datasets(
        self, use_local_files: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Load all CMAPSS dataset subsets.

        Args:
            use_local_files: If True, use files directly from data_dir

        Returns:
            Dictionary mapping subset names to tuples containing training DataFrame,
            test DataFrame, and RUL Series for test data
        """
        subsets = ["FD001", "FD002", "FD003", "FD004"]
        all_data = {}

        for subset in subsets:
            logger.info(f"Loading dataset subset {subset}")
            all_data[subset] = self.load_dataset(
                subset, use_local_files=use_local_files
            )

        return all_data


if __name__ == "__main__":
    # Simple test to verify the data loader works
    data_loader = CMAPSSDataLoader()

    # Try loading with local files first
    try:
        train_df, test_df, rul = data_loader.load_dataset("FD001", use_local_files=True)
        print("Successfully loaded local files!")
    except Exception as e:
        print(f"Failed to load local files: {str(e)}")
        print("Attempting to download dataset...")
        data_loader.download_dataset()
        data_loader.extract_dataset()
        train_df, test_df, rul = data_loader.load_dataset(
            "FD001", use_local_files=False
        )

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"RUL values shape: {rul.shape}")
    print("\nFirst few rows of training data:")
    print(train_df.head())
