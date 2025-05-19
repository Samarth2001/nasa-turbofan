"""
Data preprocessor for NASA Turbofan Engine Degradation Simulation dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
import logging

from src.config import (
    PROCESSED_DATA_DIR, 
    SENSOR_COLUMNS, 
    SETTING_COLUMNS, 
    FEATURE_COLUMNS,
    TIME_COLUMN, 
    ID_COLUMN, 
    TARGET_COLUMN,
    SEQUENCE_LENGTH,
    RANDOM_SEED
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CMAPSSPreprocessor:
    """
    Preprocessor for the NASA CMAPSS Turbofan Engine Degradation Simulation dataset.
    
    This class provides functionality to preprocess the raw data by:
    1. Adding RUL (Remaining Useful Life) values to the training data
    2. Normalizing features
    3. Creating sequence data for LSTM/GRU models
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)
    
    def add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL (Remaining Useful Life) column to the dataframe.
        
        Args:
            df: Input DataFrame with engine run data
            
        Returns:
            DataFrame with added RUL column
        """
        # Group by engine ID and calculate max cycle for each engine
        max_cycles = df.groupby(ID_COLUMN)[TIME_COLUMN].max().reset_index()
        max_cycles.columns = [ID_COLUMN, 'max_cycle']
        
        # Merge with original dataframe
        df_with_max = df.merge(max_cycles, on=ID_COLUMN, how='left')
        
        # Calculate RUL
        df_with_max[TARGET_COLUMN] = df_with_max['max_cycle'] - df_with_max[TIME_COLUMN]
        
        # Drop the max_cycle column
        df_with_max = df_with_max.drop('max_cycle', axis=1)
        
        return df_with_max
    
    def normalize_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using StandardScaler.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of normalized training and test DataFrames
        """
        # Fit scaler on training data features only
        self.scaler.fit(train_df[FEATURE_COLUMNS])
        
        # Transform both training and test data
        train_df_norm = train_df.copy()
        test_df_norm = test_df.copy()
        
        train_df_norm[FEATURE_COLUMNS] = self.scaler.transform(train_df[FEATURE_COLUMNS])
        test_df_norm[FEATURE_COLUMNS] = self.scaler.transform(test_df[FEATURE_COLUMNS])
        
        return train_df_norm, test_df_norm
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence data for time series modeling.
        
        Args:
            df: Input DataFrame with RUL values
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) where X is the sequence data and y is the target RUL values
        """
        X = []
        y = []
        
        # Group by engine ID
        grouped = df.groupby(ID_COLUMN)
        
        for engine_id, group in grouped:
            # Sort by time
            group = group.sort_values(by=TIME_COLUMN)
            
            # Extract features and target
            feature_data = group[FEATURE_COLUMNS].values
            target_data = group[TARGET_COLUMN].values
            
            # Create sequences
            for i in range(len(group) - sequence_length + 1):
                X.append(feature_data[i:i+sequence_length])
                y.append(target_data[i+sequence_length-1])  # RUL at the end of sequence
        
        return np.array(X), np.array(y)
    
    def preprocess_test_data(self, test_df: pd.DataFrame, test_rul: pd.Series, sequence_length: int = SEQUENCE_LENGTH) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Preprocess test data for model evaluation.
        
        For test data, we need to:
        1. Add the true RUL values from the separate RUL file
        2. Create sequences for each engine
        3. Keep track of engine IDs for result analysis
        
        Args:
            test_df: Test DataFrame
            test_rul: Series with true RUL values for each engine
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X_test, y_test, engine_ids) where engine_ids maps sequences to engines
        """
        # Group by engine ID and get the last sequence for each engine
        X_test = []
        y_test = []
        engine_ids = []
        
        # Group by engine ID
        grouped = test_df.groupby(ID_COLUMN)
        unique_engine_ids = test_df[ID_COLUMN].unique()
        
        for i, engine_id in enumerate(unique_engine_ids):
            group = grouped.get_group(engine_id)
            
            # Sort by time
            group = group.sort_values(by=TIME_COLUMN)
            
            # Get the last sequence
            if len(group) >= sequence_length:
                # Extract features for the last sequence
                last_sequence = group[FEATURE_COLUMNS].values[-sequence_length:]
                X_test.append(last_sequence)
                
                # Add the true RUL from the RUL file
                y_test.append(test_rul.iloc[i])
                
                # Keep track of the engine ID
                engine_ids.append(engine_id)
            else:
                logger.warning(f"Engine {engine_id} has less than {sequence_length} cycles. Skipping.")
        
        return np.array(X_test), np.array(y_test), engine_ids
    
    def process_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, test_rul: pd.Series) -> Dict:
        """
        Process data for model training and evaluation.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            test_rul: Series with true RUL values for each engine in test data
            
        Returns:
            Dictionary with processed data:
            {
                'X_train': Training sequences,
                'y_train': Training targets,
                'X_val': Validation sequences,
                'y_val': Validation targets,
                'X_test': Test sequences,
                'y_test': Test targets,
                'engine_ids': Engine IDs for test sequences
            }
        """
        # Step 1: Add RUL values to training data
        train_df_with_rul = self.add_rul(train_df)
        
        # Step 2: Normalize features
        train_df_norm, test_df_norm = self.normalize_features(train_df_with_rul, test_df)
        
        # Step 3: Create sequences for training data
        X, y = self.create_sequences(train_df_norm)
        
        # Step 4: Split training data into training and validation sets
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(X))
        split_idx = int(len(indices) * 0.8)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Step 5: Process test data
        X_test, y_test, engine_ids = self.preprocess_test_data(test_df_norm, test_rul)
        
        # Save processed data for future use
        np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
        np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
        np.save(PROCESSED_DATA_DIR / 'X_val.npy', X_val)
        np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
        np.save(PROCESSED_DATA_DIR / 'X_test.npy', X_test)
        np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)
        np.save(PROCESSED_DATA_DIR / 'engine_ids.npy', np.array(engine_ids))
        
        logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")
        logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        logger.info(f"Test data shape: {X_test.shape}, {y_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'engine_ids': engine_ids
        }


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import CMAPSSDataLoader
    
    logger.info("Testing the preprocessor")
    
    # Load data
    data_loader = CMAPSSDataLoader()
    train_df, test_df, test_rul = data_loader.load_dataset("FD001")
    
    # Preprocess data
    preprocessor = CMAPSSPreprocessor()
    processed_data = preprocessor.process_data(train_df, test_df, test_rul)
    
    # Print information about the processed data
    for key, value in processed_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key} shape: {value.shape}")
        else:
            print(f"{key} length: {len(value)}")
