"""
Model explainer for LSTM-based RUL prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional
import os
import shap
import tensorflow as tf
import logging
from pathlib import Path
import joblib

from src.config import (
    FEATURE_COLUMNS, SEQUENCE_LENGTH, MODELS_DIR
)
from src.models.lstm_model import RULPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMExplainer:
    """
    Explainer for LSTM-based RUL prediction models.
    
    This class provides functionality to explain model predictions using:
    1. Attention weights from the model
    2. SHAP values for feature importance
    """
    
    def __init__(self, model: RULPredictor, output_dir: Optional[Path] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained RUL predictor model
            output_dir: Directory to save explainability results
        """
        self.model = model
        
        if output_dir is None:
            if model.model_dir is not None:
                self.output_dir = model.model_dir / 'explainability'
            else:
                self.output_dir = MODELS_DIR / 'explainability'
        else:
            self.output_dir = output_dir
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.shap_explainer = None
        self.use_kernel_explainer = False  # Default to using DeepExplainer
        
    def explain_with_attention(self, X: np.ndarray, sequence_ids: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Explain predictions using attention weights.
        
        Args:
            X: Input sequences
            sequence_ids: Optional IDs for each sequence (e.g., engine IDs)
            
        Returns:
            Tuple of (predictions, attention weights)
        """
        # Check if model has a model attribute (the TF model)
        if not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        logger.info("Generating explanations with attention weights...")
        
        # Get predictions and attention weights
        # This now uses the predict_with_attention method that the LSTMModel provides
        # which returns both RUL predictions and attention weights
        predictions, attention_weights = self.model.predict_with_attention(X)
        
        # Save attention weights for visualization
        np.save(self.output_dir / 'attention_weights.npy', attention_weights)
        np.save(self.output_dir / 'predictions.npy', predictions)
        if sequence_ids is not None:
            np.save(self.output_dir / 'sequence_ids.npy', np.array(sequence_ids))
        
        logger.info(f"Attention explanations saved to {self.output_dir}")
        
        return predictions, attention_weights
    
    def visualize_attention(self, X: np.ndarray, attention_weights: np.ndarray, sequence_idx: int = 0) -> None:
        """
        Visualize attention weights for a specific sequence.
        
        Args:
            X: Input sequences
            attention_weights: Attention weights from the model
            sequence_idx: Index of the sequence to visualize
        """
        # Extract the specific sequence and its attention weights
        sequence = X[sequence_idx]
        weights = attention_weights[sequence_idx].flatten()
        
        plt.figure(figsize=(12, 6))
        
        # Plot attention weights
        plt.subplot(1, 2, 1)
        plt.bar(range(len(weights)), weights)
        plt.title('Attention Weights Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.tight_layout()
        
        # Plot attention heatmap over sequence
        plt.subplot(1, 2, 2)
        df = pd.DataFrame(sequence, columns=FEATURE_COLUMNS)
        
        # Normalize feature values for better heatmap visualization
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_norm['Attention'] = weights # Add attention for heatmap context
        
        sns.heatmap(df_norm.T, cmap='viridis')
        plt.title('Feature Values and Attention Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'attention_viz_sequence_{sequence_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention visualization saved for sequence {sequence_idx}")
    
    def initialize_shap_explainer(self, background_data: np.ndarray) -> None:
        """
        Initialize the SHAP explainer with the provided background data.
        
        Args:
            background_data: Background data for SHAP explainer
        """
        logger.info("Initializing SHAP explainer...")
        
        # Create a simplified model for SHAP that only outputs the RUL prediction (first output)
        # This is needed because SHAP expects a model with a single output
        input_layer = tf.keras.layers.Input(shape=self.model.model.input_shape[1:])
        full_model_output = self.model.model(input_layer)
        # Extract only the RUL prediction output (first output)
        rul_output = full_model_output[0]  # Assuming the first output is the RUL prediction
        simplified_model = tf.keras.Model(inputs=input_layer, outputs=rul_output)

        if self.use_kernel_explainer:
            logger.info("Using KernelExplainer as requested")
            # Create a prediction function for the model
            def model_predict(x):
                # Extract only RUL predictions from the model output
                return simplified_model.predict(x)
                
            # Use KernelExplainer (slower but more compatible)
            try:
                # Reshape if needed
                if len(background_data.shape) == 3:
                    # For sequence data, we need to flatten the time dimension
                    num_samples = background_data.shape[0]
                    flattened_background = background_data.reshape(num_samples, -1)
                    self.shap_explainer = shap.KernelExplainer(model_predict, flattened_background[:100])
                else:
                    # Use a smaller sample for KernelExplainer (for performance)
                    self.shap_explainer = shap.KernelExplainer(model_predict, background_data[:100])
                logger.info("SHAP explainer initialized using KernelExplainer")
            except Exception as e:
                logger.error(f"KernelExplainer initialization failed: {str(e)}")
                self.shap_explainer = None
        else:
            try:
                # Try to use DeepExplainer first (faster but may fail with newer TensorFlow)
                self.shap_explainer = shap.DeepExplainer(simplified_model, background_data)
                logger.info("SHAP explainer initialized using DeepExplainer")
            except Exception as e:
                logger.warning(f"DeepExplainer failed: {str(e)}")
                logger.warning("Falling back to KernelExplainer (slower but more compatible)")
                
                # Create a prediction function for the model
                def model_predict(x):
                    # Extract only RUL predictions from the model output
                    return simplified_model.predict(x)
                    
                # Use KernelExplainer as a fallback (slower but more compatible)
                try:
                    # Reshape if needed
                    if len(background_data.shape) == 3:
                        # For sequence data, we need to flatten the time dimension
                        num_samples = background_data.shape[0]
                        flattened_background = background_data.reshape(num_samples, -1)
                        self.shap_explainer = shap.KernelExplainer(model_predict, flattened_background[:100])
                    else:
                        # Use a smaller sample for KernelExplainer (for performance)
                        self.shap_explainer = shap.KernelExplainer(model_predict, background_data[:100])
                    logger.info("SHAP explainer initialized using KernelExplainer")
                except Exception as e2:
                    logger.error(f"All SHAP explainer methods failed. Last error: {str(e2)}")
                    self.shap_explainer = None
    
    def explain_with_shap(self, X: np.ndarray, background_data: Optional[np.ndarray] = None, max_samples: int = 20) -> np.ndarray:
        """
        Generate SHAP explanations for the model predictions.
        
        Args:
            X: Input data to explain
            background_data: Background data for SHAP explainer (typically subset of training data)
            max_samples: Maximum number of samples to use for SHAP analysis
            
        Returns:
            SHAP values for each feature and each time step
        """
        if background_data is None:
            logger.warning("No background data provided for SHAP explainer, skipping SHAP analysis")
            return np.zeros((min(len(X), max_samples), *X.shape[1:]))
            
        # Initialize SHAP explainer if not already initialized
        if self.shap_explainer is None:
            self.initialize_shap_explainer(background_data)
            
        # Take a subset of samples for SHAP analysis
        X_sample = X[:max_samples] if len(X) > max_samples else X
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Convert to numpy array if needed
            if isinstance(shap_values, list):
                # For multi-output models, SHAP returns a list of arrays
                shap_values = np.array(shap_values[0])
            
            logger.info(f"Generated SHAP values with shape {shap_values.shape}")
            return shap_values
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {str(e)}")
            logger.warning("Returning zero matrix instead of SHAP values")
            # Return zeros with the expected shape
            return np.zeros((len(X_sample), *X_sample.shape[1:]))
    
    def visualize_shap_values(self, X: np.ndarray, shap_values: np.ndarray, sequence_idx: int = 0) -> None:
        """
        Visualize SHAP values for a specific sequence.
        
        Args:
            X: Input sequences
            shap_values: SHAP values from the explainer
            sequence_idx: Index of the sequence to visualize
        """
        # Extract the specific sequence and its SHAP values
        sequence = X[sequence_idx]
        values = shap_values[sequence_idx]
        
        # Create feature names with time steps
        feature_names = []
        for t in range(sequence.shape[0]):
            for f in FEATURE_COLUMNS:
                feature_names.append(f"{f}_t{t}")
        
        # Flatten the data for visualization
        flattened_sequence = sequence.flatten()
        flattened_values = values.flatten()
        
        # Create a DataFrame for SHAP summary
        df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': flattened_values,
            'Feature Value': flattened_sequence
        })
        
        # Sort by absolute SHAP value for prominence
        df['Abs SHAP'] = df['SHAP Value'].abs()
        df = df.sort_values('Abs SHAP', ascending=False)
        
        # Display top N features for clarity
        df = df.head(30)
        
        plt.figure(figsize=(12, 10))
        
        # Plot SHAP values, color-coded by positive/negative impact
        colors = np.array(['blue' if x < 0 else 'red' for x in df['SHAP Value']])
        plt.barh(range(len(df)), df['SHAP Value'], color=colors)
        plt.yticks(range(len(df)), df['Feature'])
        plt.title(f'Top 30 SHAP Values for Sequence {sequence_idx}')
        plt.xlabel('SHAP Value (Impact on Prediction)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'shap_viz_sequence_{sequence_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP visualization saved for sequence {sequence_idx}")
    
    def generate_global_feature_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Generate global feature importance based on SHAP values.
        
        Args:
            shap_values: SHAP values from the explainer
            
        Returns:
            DataFrame with global feature importance
        """
        logger.info("Generating global feature importance...")
        
        # Reshape SHAP values to group by feature
        # Shape: (n_samples, sequence_length, n_features) -> (n_samples, n_features)
        n_samples, sequence_length, n_features = shap_values.shape
        feature_importance = np.zeros((n_samples, n_features))
        
        # Sum absolute SHAP values across time steps
        for i in range(n_features):
            feature_importance[:, i] = np.abs(shap_values[:, :, i]).sum(axis=1)
        
        # Calculate mean importance across samples
        mean_importance = feature_importance.mean(axis=0)
        
        # Create a DataFrame
        importance_df = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Importance': mean_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save the importance scores
        importance_df.to_csv(self.output_dir / 'global_feature_importance.csv', index=False)
        
        # Visualize
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Global Feature Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'global_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Global feature importance saved to {self.output_dir}")
        
        return importance_df
    
    def visualize_shap_summary_plot(self, X_sample: np.ndarray, shap_values: np.ndarray) -> None:
        """
        Visualize SHAP summary plot (e.g., bee swarm).

        Args:
            X_sample: The samples for which SHAP values were computed.
            shap_values: SHAP values from the explainer.
        """
        logger.info("Generating SHAP summary plot...")

        # If shap_values is a list (e.g. for multi-output models), and we have one output:
        current_shap_values = shap_values
        if isinstance(shap_values, list):
            current_shap_values = shap_values[0] # Assuming the first output for RUL

        # For LSTM, X_sample is (n_samples, sequence_length, n_features)
        # shap_values is (n_samples, sequence_length, n_features)
        # We need to reshape X_sample to (n_samples * sequence_length, n_features)
        # and shap_values to (n_samples * sequence_length, n_features)
        # or provide feature names that match the original structure if the plot function supports it.
        # For a simple summary plot, often an average impact per feature is shown.
        # Let's try a standard summary_plot.
        
        # Reshape X_sample if it's 3D: (samples, timesteps, features)
        # The feature names need to align with the last dimension of shap_values
        reshaped_X_sample_for_plot = X_sample
        if X_sample.ndim == 3:
            # Option 1: Average SHAP values over timesteps for each feature
            # mean_shap_values = np.mean(current_shap_values, axis=1)
            # mean_X_sample = np.mean(X_sample, axis=1) # This might not be meaningful for feature values
            # shap.summary_plot(mean_shap_values, mean_X_sample, feature_names=FEATURE_COLUMNS, show=False)
            
            # Option 2: Flatten and use time-step specific feature names (can be too many features)
            # This is complex for beeswarm directly. Let's use the global bar plot.
            
            # For summary_plot (bar), usually global mean absolute SHAP values are used.
             shap.summary_plot(current_shap_values, X_sample, feature_names=FEATURE_COLUMNS, plot_type="bar", show=False)

        else: # Assuming X_sample is 2D
            shap.summary_plot(current_shap_values, X_sample, feature_names=FEATURE_COLUMNS, plot_type="bar", show=False)

        plt.title('Global SHAP Feature Importance (Mean Absolute SHAP Value)')
        plt.savefig(self.output_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP summary plot saved to {self.output_dir / 'shap_summary_plot.png'}")


    def explain_prediction(
        self, 
        X: np.ndarray, 
        background_data: Optional[np.ndarray] = None,
        sequence_ids: Optional[List[int]] = None
    ) -> Dict:
        """
        Generate comprehensive explanations for model predictions.
        
        Args:
            X: Input data to explain
            background_data: Background data for SHAP explainer
            sequence_ids: Optional IDs for each sequence (e.g., engine IDs)
            
        Returns:
            Dictionary with explanation results
        """
        logger.info("Generating comprehensive explanations...")
        
        # Generate explanations with attention weights
        logger.info("Generating explanations with attention weights...")
        predictions, attention_weights = self.explain_with_attention(X, sequence_ids)
        
        # Save attention weights for visualization
        np.save(self.output_dir / 'attention_weights.npy', attention_weights)
        np.save(self.output_dir / 'attention_explained_samples.npy', X)
        
        # Visualize attention for a few examples
        for i in range(min(5, len(X))):
            self.visualize_attention(X, attention_weights, sequence_idx=i)
        
        # Generate explanations with SHAP
        shap_values = None
        feature_importance_df = None
        
        if background_data is not None:
            try:
                logger.info("Generating SHAP explanations for 20 samples...")
                shap_values = self.explain_with_shap(X, background_data)
                
                # Only proceed with SHAP visualizations if we got non-zero values
                if np.any(shap_values):
                    # Save SHAP values for visualization
                    np.save(self.output_dir / 'shap_values.npy', shap_values)
                    np.save(self.output_dir / 'shap_explained_samples.npy', X[:20])
                    
                    # Visualize SHAP values for a few examples
                    for i in range(min(5, len(shap_values))):
                        self.visualize_shap_values(X, shap_values, sequence_idx=i)
                    
                    # Generate feature importance based on SHAP
                    feature_importance_df = self.generate_global_feature_importance(shap_values)
            except Exception as e:
                logger.warning(f"Error in SHAP explanation or visualization: {str(e)}")
                logger.warning("Continuing with attention-based explanations only.")
        else:
            logger.warning("Skipping SHAP explanation: no background data provided")
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'shap_values': shap_values,
            'feature_importance': feature_importance_df
        }


if __name__ == "__main__":
    # Simple test to verify the explainer works
    from src.models.lstm_model import RULPredictor
    
    # Create a model
    model = RULPredictor()
    model.build_model()
    
    # Create some dummy data
    X_dummy = np.random.rand(20, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    
    # Create an explainer
    explainer = LSTMExplainer(model)
    
    # Explain with attention
    predictions, attention_weights = explainer.explain_with_attention(X_dummy)
    print("Predictions shape:", predictions.shape)
    print("Attention weights shape:", attention_weights.shape)
    
    # Visualize attention for the first sequence
    explainer.visualize_attention(X_dummy, attention_weights, sequence_idx=0)
    
    # Explain with SHAP
    background_data = X_dummy[:10]  # Use some samples as background
    shap_values = explainer.explain_with_shap(X_dummy[10:15], background_data)
    print("SHAP values shape:", shap_values.shape)
    
    # Visualize SHAP values for the first sequence
    explainer.visualize_shap_values(X_dummy[10:15], shap_values, sequence_idx=0)
    
    # Generate global feature importance
    importance_df = explainer.generate_global_feature_importance(shap_values)
    print("Feature importance:\n", importance_df)
