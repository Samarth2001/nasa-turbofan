"""
Utility functions for model evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metric names and values
    """
    # Ensure inputs are flattened
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error) with protection against division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if any(mask) else np.inf
    
    # Calculate Score function as used in PHM Data Challenge
    # Score = sum(exp(error/13) - 1) if error < 0 else sum(exp(error/10) - 1)
    errors = y_pred - y_true
    s1 = np.sum(np.exp(errors[errors < 0] / 13) - 1)
    s2 = np.sum(np.exp(errors[errors >= 0] / 10) - 1)
    phm_score = s1 + s2
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'PHM_Score': phm_score
    }


def plot_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    engine_ids: Optional[List[int]] = None,
    title: str = "RUL Predictions vs True Values",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot predictions against true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        engine_ids: Optional engine IDs for each data point
        title: Plot title
        output_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    ax1.set_xlabel('True RUL')
    ax1.set_ylabel('Predicted RUL')
    ax1.set_title('Predicted vs True RUL')
    ax1.grid(True)
    
    # Calculate error
    error = y_pred - y_true
    
    # Plot error distribution
    sns.histplot(error, kde=True, ax=ax2)
    ax2.axvline(0, color='r', linestyle='--')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Error Distribution')
    ax2.grid(True)
    
    # Add metrics as text
    metrics = calculate_metrics(y_true, y_pred)
    metrics_text = "\n".join([f"{name}: {value:.4f}" for name, value in metrics.items() if name != 'PHM_Score'])
    
    # Position the text in the top right of the error plot
    ax2.text(
        0.95, 0.95, metrics_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    return fig


def plot_rul_vs_cycle(
    true_rul: np.ndarray,
    pred_rul: np.ndarray,
    engine_id: int,
    max_cycle: int,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot RUL vs cycle for a specific engine.
    
    Args:
        true_rul: True RUL values
        pred_rul: Predicted RUL values
        engine_id: Engine ID
        max_cycle: Maximum cycle for the engine
        output_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate cycle numbers
    cycles = np.arange(1, len(true_rul) + 1)
    
    # Plot true and predicted RUL
    ax.plot(cycles, true_rul, 'b-', label='True RUL')
    ax.plot(cycles, pred_rul, 'r-', label='Predicted RUL')
    
    # Add warning thresholds
    ax.axhline(50, color='orange', linestyle='--', label='Warning Threshold (50 cycles)')
    ax.axhline(20, color='red', linestyle='--', label='Critical Threshold (20 cycles)')
    
    # Add labels and title
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Remaining Useful Life (cycles)')
    ax.set_title(f'RUL vs Cycle for Engine {engine_id}')
    ax.grid(True)
    ax.legend()
    
    # Set y-axis limit to ensure RUL=0 is visible
    ax.set_ylim(bottom=-5)
    
    # Save if output path is provided
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    return fig


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    engine_ids: Optional[List[int]] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Comprehensive evaluation of model predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        engine_ids: Optional engine IDs for each data point
        output_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results
    """
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    logger.info("Model evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    # Create output directory if provided
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics to CSV
        pd.DataFrame(metrics, index=[0]).to_csv(output_dir / 'metrics.csv', index=False)
        
        # Plot predictions
        plot_predictions(
            y_true, y_pred, engine_ids,
            output_path=output_dir / 'predictions.png'
        )
        
        # Create a DataFrame with results
        results_df = pd.DataFrame({
            'Engine_ID': engine_ids if engine_ids is not None else np.arange(len(y_true)),
            'True_RUL': y_true.flatten(),
            'Predicted_RUL': y_pred.flatten(),
            'Error': y_pred.flatten() - y_true.flatten()
        })
        
        # Save results to CSV
        results_df.to_csv(output_dir / 'prediction_results.csv', index=False)
    
    return {
        'metrics': metrics
    }


if __name__ == "__main__":
    # Simple test to verify the metrics functions
    y_true = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    y_pred = np.array([95, 92, 85, 68, 58, 55, 45, 28, 15, 5])
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Create a simple plot
    fig = plot_predictions(y_true, y_pred, title="Test Predictions")
    plt.show()
