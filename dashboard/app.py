"""
Streamlit dashboard for NASA Turbofan Predictive Maintenance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import os
from pathlib import Path
import tensorflow as tf
import sys
import logging
from typing import Dict, List, Tuple, Optional
import joblib
from scipy import stats

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DASHBOARD_TITLE, PROCESSED_DATA_DIR, MODELS_DIR,
    RUL_THRESHOLD_WARNING, RUL_THRESHOLD_CRITICAL,
    FEATURE_COLUMNS, SEQUENCE_LENGTH
)
from src.data.data_loader import CMAPSSDataLoader
from src.data.preprocessor import CMAPSSPreprocessor
from src.models.lstm_model import RULPredictor
from src.models.explainer import LSTMExplainer
from src.utils.metrics import calculate_metrics, plot_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Set page config
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_data():
    """Load processed data for dashboard."""
    try:
        # Load processed data
        X_train = np.load(PROCESSED_DATA_DIR / 'X_train.npy')
        y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
        X_val = np.load(PROCESSED_DATA_DIR / 'X_val.npy')
        y_val = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
        X_test = np.load(PROCESSED_DATA_DIR / 'X_test.npy')
        y_test = np.load(PROCESSED_DATA_DIR / 'y_test.npy')
        engine_ids = np.load(PROCESSED_DATA_DIR / 'engine_ids.npy')
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'engine_ids': engine_ids
        }
    except FileNotFoundError:
        st.error("Processed data not found. Please run the data processing pipeline first.")
        return None


def load_model(model_path):
    """Load a trained model for prediction."""
    try:
        # Create a model instance
        model = RULPredictor()
        # Load weights from saved model
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def get_available_models():
    """Get list of available trained models."""
    models_dir = Path(MODELS_DIR)
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / 'final_model.h5').exists()]
    return model_dirs


def load_multiple_models(model_dirs):
    """Load multiple models for comparison."""
    models = {}
    for model_dir in model_dirs:
        try:
            model_name = model_dir.name
            model_path = model_dir / 'final_model.h5'
            
            model = RULPredictor()
            model.load_model(model_path)
            models[model_name] = model
            
        except Exception as e:
            logger.error(f"Error loading model {model_dir.name}: {e}")
    
    return models


def make_predictions(model, data):
    """Make predictions using the loaded model."""
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred.flatten())
    
    return y_pred, metrics


def compare_models(models, data):
    """Compare performance of multiple models."""
    results = {}
    
    for model_name, model in models.items():
        y_pred, metrics = make_predictions(model, data)
        results[model_name] = {
            'y_pred': y_pred,
            'metrics': metrics
        }
    
    return results


def generate_confidence_intervals(model, X, n_samples=100, noise_level=0.05):
    """Generate confidence intervals using Monte Carlo dropout or noise injection."""
    predictions = []
    
    # Run multiple forward passes with noise
    for _ in range(n_samples):
        # Add noise to input to simulate uncertainty
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        
        # Get prediction
        y_pred = model.predict(X_noisy)
        predictions.append(y_pred)
    
    # Stack predictions
    predictions = np.hstack(predictions)
    
    # Calculate mean and confidence intervals
    mean_pred = np.mean(predictions, axis=1)
    lower_bound = np.percentile(predictions, 5, axis=1)
    upper_bound = np.percentile(predictions, 95, axis=1)
    
    return mean_pred, lower_bound, upper_bound


def plot_predictions_with_uncertainty(y_true, y_pred, lower_bound, upper_bound):
    """Plot predictions with uncertainty bands."""
    df = pd.DataFrame({
        'Engine ID': range(len(y_true)),
        'True RUL': y_true.flatten(),
        'Predicted RUL': y_pred,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    })
    
    fig = go.Figure()
    
    # Add scatter plot for predictions
    fig.add_trace(
        go.Scatter(
            x=df['Engine ID'], 
            y=df['True RUL'],
            mode='markers',
            name='True RUL',
            marker=dict(color='blue')
        )
    )
    
    # Add predictions with confidence intervals
    fig.add_trace(
        go.Scatter(
            x=df['Engine ID'],
            y=df['Predicted RUL'],
            mode='markers',
            name='Predicted RUL',
            marker=dict(color='red')
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([df['Engine ID'], df['Engine ID'][::-1]]),
            y=np.concatenate([df['Upper Bound'], df['Lower Bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='RUL Predictions with Uncertainty',
        xaxis_title='Engine ID',
        yaxis_title='RUL (cycles)',
        height=500
    )
    
    return fig


def what_if_analysis(model, X_sample, feature_ranges):
    """Perform what-if analysis by adjusting feature values."""
    # Get baseline prediction
    baseline_pred = model.predict(np.array([X_sample]))[0]
    
    results = {}
    
    # Adjust each feature and observe the effect
    for feature_idx, feature_name in enumerate(FEATURE_COLUMNS):
        if feature_name not in feature_ranges:
            continue
        
        min_val, max_val = feature_ranges[feature_name]
        
        # Generate range of values for this feature
        values = np.linspace(min_val, max_val, 10)
        predictions = []
        
        for val in values:
            # Create a copy of the sample and modify the feature
            modified_sample = X_sample.copy()
            modified_sample[:, feature_idx] = val
            
            # Get prediction
            pred = model.predict(np.array([modified_sample]))[0]
            predictions.append(pred)
        
        results[feature_name] = {
            'values': values,
            'predictions': np.array(predictions).flatten()
        }
    
    return baseline_pred, results


def plot_what_if_results(feature_name, values, predictions, baseline_pred):
    """Plot results of what-if analysis for a specific feature."""
    df = pd.DataFrame({
        'Feature Value': values,
        'Predicted RUL': predictions
    })
    
    fig = px.line(
        df, 
        x='Feature Value', 
        y='Predicted RUL',
        title=f'Effect of {feature_name} on Predicted RUL',
        markers=True
    )
    
    # Add baseline
    fig.add_hline(
        y=baseline_pred, 
        line_width=2, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Baseline Prediction",
        annotation_position="top right"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=f"{feature_name} Value",
        yaxis_title="Predicted RUL (cycles)"
    )
    
    return fig


def plot_rul_distribution(y_true, y_pred):
    """Plot distribution of true and predicted RUL values."""
    fig = make_subplots(rows=1, cols=2, 
                         subplot_titles=("True RUL Distribution", "Predicted RUL Distribution"))
    
    # True RUL distribution
    fig.add_trace(
        go.Histogram(x=y_true, name="True RUL", marker_color="blue", opacity=0.7),
        row=1, col=1
    )
    
    # Predicted RUL distribution
    fig.add_trace(
        go.Histogram(x=y_pred, name="Predicted RUL", marker_color="red", opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="RUL Distributions",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_prediction_scatter(y_true, y_pred):
    """Create a scatter plot of predicted vs true RUL values."""
    df = pd.DataFrame({
        'True RUL': y_true,
        'Predicted RUL': y_pred,
        'Error': y_pred - y_true
    })
    
    # Create scatter plot
    fig = px.scatter(
        df, x='True RUL', y='Predicted RUL',
        color='Error', color_continuous_scale='RdBu_r',
        title='Predicted vs True RUL'
    )
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max()) 
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], 
            y=[min_val, max_val],
            mode='lines', 
            line=dict(color='black', dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar=dict(
            title='Error',
        )
    )
    
    return fig


def plot_error_distribution(y_true, y_pred):
    """Plot distribution of prediction errors."""
    errors = y_pred - y_true
    
    fig = go.Figure()
    
    # Add error histogram
    fig.add_trace(go.Histogram(
        x=errors,
        name="Prediction Error",
        marker_color="purple",
        opacity=0.7
    ))
    
    # Add vertical line at zero
    fig.add_vline(
        x=0, 
        line_width=2, 
        line_dash="dash", 
        line_color="red"
    )
    
    # Add mean error line
    mean_error = errors.mean()
    fig.add_vline(
        x=mean_error, 
        line_width=2, 
        line_dash="dot", 
        line_color="green",
        annotation_text=f"Mean: {mean_error:.2f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title_text="Prediction Error Distribution",
        xaxis_title="Prediction Error",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig


def plot_feature_importance(explainer_dir):
    """Plot feature importance from model explanations."""
    importance_file = Path(explainer_dir) / 'global_feature_importance.csv'
    
    if not importance_file.exists():
        st.warning("Feature importance data not found.")
        return None
    
    # Load feature importance data
    importance_df = pd.read_csv(importance_file)
    
    # Create bar plot
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Global Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder':'total ascending'}
    )
    
    return fig


def plot_attention_heatmap(X, attention_weights, sequence_idx=0):
    """Plot attention heatmap for a specific sequence."""
    # Extract the specific sequence and its attention weights
    sequence = X[sequence_idx]
    weights = attention_weights[sequence_idx].flatten()
    
    # Create a dataframe for heatmap
    df = pd.DataFrame(sequence, columns=FEATURE_COLUMNS)
    
    # Create time step column names
    time_steps = [f"t-{SEQUENCE_LENGTH-i}" for i in range(SEQUENCE_LENGTH)]
    
    # Transpose for better visualization
    df_heatmap = df.copy()
    df_heatmap.index = time_steps
    
    # Normalize feature values for better visualization
    for column in df_heatmap.columns:
        if df_heatmap[column].max() > df_heatmap[column].min():
            df_heatmap[column] = (df_heatmap[column] - df_heatmap[column].min()) / (df_heatmap[column].max() - df_heatmap[column].min())
    
    # Create heatmap
    fig = px.imshow(
        df_heatmap.T,
        title=f'Feature Values Across Time Steps (Sequence {sequence_idx})',
        labels=dict(x="Time Step", y="Feature", color="Normalized Value"),
        color_continuous_scale='Viridis'
    )
    
    # Create attention weight plot
    attention_df = pd.DataFrame({
        'Time Step': time_steps,
        'Attention Weight': weights
    })
    
    attention_fig = px.bar(
        attention_df, 
        x='Time Step', 
        y='Attention Weight',
        title=f'Attention Weights (Sequence {sequence_idx})',
        color='Attention Weight',
        color_continuous_scale='Reds'
    )
    
    return fig, attention_fig


def display_rul_alerts(engine_ids, predictions):
    """Display RUL alerts for engines near failure."""
    df = pd.DataFrame({
        'Engine ID': engine_ids,
        'Predicted RUL': predictions.flatten()
    })
    
    # Add status column
    df['Status'] = 'Normal'
    df.loc[df['Predicted RUL'] <= RUL_THRESHOLD_WARNING, 'Status'] = 'Warning'
    df.loc[df['Predicted RUL'] <= RUL_THRESHOLD_CRITICAL, 'Status'] = 'Critical'
    
    # Sort by RUL
    df = df.sort_values('Predicted RUL')
    
    # Create status columns for styling
    status_colors = {
        'Normal': 'green',
        'Warning': 'orange',
        'Critical': 'red'
    }
    
    # Display alerts in columns by status
    cols = st.columns(3)
    
    # Display critical alerts
    critical_df = df[df['Status'] == 'Critical']
    with cols[0]:
        st.subheader("🚨 Critical Alerts")
        if len(critical_df) > 0:
            for _, row in critical_df.iterrows():
                st.error(f"Engine {int(row['Engine ID'])}: RUL = {row['Predicted RUL']:.1f} cycles")
        else:
            st.write("No critical alerts")
    
    # Display warning alerts
    warning_df = df[df['Status'] == 'Warning']
    with cols[1]:
        st.subheader("⚠️ Warning Alerts")
        if len(warning_df) > 0:
            for _, row in warning_df.iterrows():
                st.warning(f"Engine {int(row['Engine ID'])}: RUL = {row['Predicted RUL']:.1f} cycles")
        else:
            st.write("No warning alerts")
    
    # Display healthy engines
    normal_df = df[df['Status'] == 'Normal']
    with cols[2]:
        st.subheader("✅ Healthy Engines")
        if len(normal_df) > 0:
            st.write(f"{len(normal_df)} engines with RUL > {RUL_THRESHOLD_WARNING} cycles")
            if st.checkbox("Show Details"):
                for _, row in normal_df.iterrows():
                    st.info(f"Engine {int(row['Engine ID'])}: RUL = {row['Predicted RUL']:.1f} cycles")
        else:
            st.write("No engines in normal condition")
    
    return df


def main():
    """Main dashboard application."""
    st.title("🛠️ NASA Turbofan Predictive Maintenance")
    st.markdown("""
    This dashboard visualizes results from LSTM-based predictive maintenance models
    for the NASA Turbofan Engine Degradation Simulation dataset.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "Overview", 
        "Model Performance", 
        "Explainability", 
        "RUL Monitoring",
        "Model Comparison",
        "What-If Analysis",
        "Uncertainty Quantification"
    ])
    
    # Get available models
    model_dirs = get_available_models()
    if not model_dirs:
        st.error("No trained models found. Please train a model first.")
        return
    
    # Model selection
    model_options = {str(d): d for d in model_dirs}
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        format_func=lambda x: x.split('/')[-1]
    )
    selected_model_dir = model_options[selected_model_name]
    model_path = selected_model_dir / 'final_model.h5'
    
    # Load data and model
    data = load_data()
    if data is None:
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Make predictions
    y_pred, metrics = make_predictions(model, data)
    
    # Load attention weights if available
    explainer_dir = selected_model_dir / 'explainability'
    attention_file = explainer_dir / 'attention_weights.npy'
    if attention_file.exists():
        attention_weights = np.load(attention_file)
    else:
        attention_weights = None

    # Original pages - Overview, Model Performance, Explainability, RUL Monitoring
    if page in ["Overview", "Model Performance", "Explainability", "RUL Monitoring"]:
        # Overview page
        if page == "Overview":
            st.header("📊 Dataset Overview")
            
            # Dataset stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Sequences", len(data['X_train']))
            with col2:
                st.metric("Validation Sequences", len(data['X_val']))
            with col3:
                st.metric("Test Engines", len(data['engine_ids']))
            
            st.subheader("RUL Distribution")
            
            # RUL distribution plot
            rul_dist_fig = plot_rul_distribution(data['y_test'], y_pred.flatten())
            st.plotly_chart(rul_dist_fig, use_container_width=True)
            
            # Feature information
            st.subheader("Feature Information")
            feature_desc = {
                'setting_1': 'Altitude (0-42K ft)',
                'setting_2': 'Mach number (0-0.84)',
                'setting_3': 'TRA (Throttle Resolver Angle)',
                's_2': 'Total temperature at LPC outlet',
                's_3': 'Total temperature at HPC outlet',
                's_4': 'Total temperature at LPT outlet',
                's_7': 'Total pressure at HPC outlet',
                's_8': 'Physical fan speed',
                's_9': 'Physical core speed',
                's_11': 'Static pressure at HPC outlet',
                's_12': 'Ratio of fuel flow to Ps30',
                's_13': 'Corrected fan speed',
                's_14': 'Corrected core speed',
                's_15': 'Bypass ratio',
                's_17': 'Bleed enthalpy',
                's_20': 'Burner fuel-air ratio',
                's_21': 'HPT coolant bleed'
            }
            
            # Create feature description table
            feature_df = pd.DataFrame({
                'Feature': list(feature_desc.keys()),
                'Description': list(feature_desc.values())
            })
            st.dataframe(feature_df, hide_index=True)
        
        # Model Performance page
        elif page == "Model Performance":
            st.header("📈 Model Performance")
            
            # Metrics
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame({metric: [value] for metric, value in metrics.items()})
            st.dataframe(metrics_df.T.rename(columns={0: 'Value'}), hide_index=False)
            
            # Predictions scatter plot
            st.subheader("Prediction Accuracy")
            scatter_fig = plot_prediction_scatter(data['y_test'], y_pred.flatten())
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Error distribution
            st.subheader("Error Analysis")
            error_fig = plot_error_distribution(data['y_test'], y_pred.flatten())
            st.plotly_chart(error_fig, use_container_width=True)
            
            # Engine-specific analysis
            st.subheader("Engine-specific Analysis")
            engine_selector = st.selectbox(
                "Select Engine ID for Detailed Analysis",
                np.unique(data['engine_ids'])
            )
            
            # Filter data for selected engine
            engine_idx = np.where(data['engine_ids'] == engine_selector)[0][0]
            true_rul = data['y_test'][engine_idx]
            pred_rul = y_pred[engine_idx][0]
            
            # Display engine-specific metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True RUL", f"{true_rul:.1f} cycles")
            with col2:
                st.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
            with col3:
                error = pred_rul - true_rul
                st.metric("Prediction Error", f"{error:.1f} cycles", delta=f"{-abs(error):.1f}" if abs(error) > 0 else "0")
        
        # Explainability page
        elif page == "Explainability":
            st.header("🔍 Model Explainability")
            
            # Feature importance
            st.subheader("Global Feature Importance")
            importance_fig = plot_feature_importance(explainer_dir)
            if importance_fig:
                st.plotly_chart(importance_fig, use_container_width=True)
            
            # Attention visualization
            st.subheader("Attention Mechanism Visualization")
            
            if attention_weights is not None:
                # Select sequence for visualization
                sequence_idx = st.slider("Select Sequence Index", 0, len(data['X_test'])-1, 0)
                
                # Get engine ID for this sequence
                engine_id = data['engine_ids'][sequence_idx]
                st.write(f"Engine ID: {engine_id}")
                
                # Plot attention heatmap and weights
                heatmap_fig, attention_fig = plot_attention_heatmap(
                    data['X_test'], attention_weights, sequence_idx)
                
                # Display plots
                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.plotly_chart(attention_fig, use_container_width=True)
                
                st.info("""
                The attention heatmap shows how the model focuses on different features and time steps
                when making predictions. Higher attention weights indicate more influence on the prediction.
                """)
            else:
                st.warning("Attention weights not available for this model.")
        
        # RUL Monitoring page
        elif page == "RUL Monitoring":
            st.header("⚠️ RUL Monitoring Dashboard")
            
            st.markdown(f"""
            The dashboard monitors the Remaining Useful Life (RUL) of engines and provides alerts based on thresholds:
            - **Critical Alert** 🚨: RUL ≤ {RUL_THRESHOLD_CRITICAL} cycles (immediate maintenance required)
            - **Warning Alert** ⚠️: RUL ≤ {RUL_THRESHOLD_WARNING} cycles (plan maintenance soon)
            - **Normal** ✅: RUL > {RUL_THRESHOLD_WARNING} cycles (healthy condition)
            """)
            
            # Display alerts
            alert_df = display_rul_alerts(data['engine_ids'], y_pred)
            
            # Engine status summary chart
            st.subheader("Engine Status Summary")
            status_counts = alert_df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            
            # Add missing statuses with zero count
            for status in ['Critical', 'Warning', 'Normal']:
                if status not in status_counts['Status'].values:
                    status_counts = pd.concat([
                        status_counts, 
                        pd.DataFrame({'Status': [status], 'Count': [0]})
                    ])
            
            # Set color map
            color_map = {'Critical': 'red', 'Warning': 'orange', 'Normal': 'green'}
            status_counts['Color'] = status_counts['Status'].map(color_map)
            
            # Create pie chart
            fig = px.pie(
                status_counts, 
                values='Count', 
                names='Status',
                color='Status',
                color_discrete_map=color_map,
                title='Engine Status Distribution'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RUL Distribution
            st.subheader("RUL Distribution Across All Engines")
            
            # Create histogram
            hist_fig = px.histogram(
                alert_df,
                x='Predicted RUL',
                color='Status',
                color_discrete_map=color_map,
                title='RUL Distribution by Status',
                nbins=20
            )
            
            # Add threshold lines
            hist_fig.add_vline(
                x=RUL_THRESHOLD_WARNING, 
                line_width=2, 
                line_dash="dash", 
                line_color="orange",
                annotation_text=f"Warning: {RUL_THRESHOLD_WARNING} cycles",
                annotation_position="top right"
            )
            
            hist_fig.add_vline(
                x=RUL_THRESHOLD_CRITICAL, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Critical: {RUL_THRESHOLD_CRITICAL} cycles",
                annotation_position="top right"
            )
            
            st.plotly_chart(hist_fig, use_container_width=True)

    # Model Comparison page
    elif page == "Model Comparison":
        st.header("🔄 Model Comparison")
        
        st.markdown("""
        Compare the performance of multiple models side by side to identify the best performing model.
        """)
        
        # Select models to compare
        selected_models_for_comparison = st.multiselect(
            "Select Models to Compare",
            list(model_options.keys()),
            format_func=lambda x: x.split('/')[-1]
        )
        
        if not selected_models_for_comparison:
            st.warning("Please select at least one model to compare.")
            return
        
        # Load selected models
        models_to_compare = {}
        for model_name in selected_models_for_comparison:
            model_dir = model_options[model_name]
            models_to_compare[model_dir.name] = load_model(model_dir / 'final_model.h5')
        
        # Compare models
        comparison_results = compare_models(models_to_compare, data)
        
        # Display comparison metrics
        st.subheader("Performance Metrics Comparison")
        
        # Create comparison dataframe
        metrics_comparison = {}
        for model_name, result in comparison_results.items():
            metrics_comparison[model_name] = result['metrics']
        
        # Convert to dataframe
        metrics_df = pd.DataFrame(metrics_comparison)
        st.dataframe(metrics_df, hide_index=False)
        
        # Plot comparative metrics
        st.subheader("Comparative Performance")
        
        # Select metric to visualize
        selected_metric = st.selectbox(
            "Select Metric to Visualize",
            ['RMSE', 'MAE', 'R2', 'MAPE', 'PHM_Score'],
            index=0
        )
        
        # Create comparison bar chart
        comparison_df = pd.DataFrame({
            'Model': list(metrics_comparison.keys()),
            selected_metric: [metrics[selected_metric] for metrics in metrics_comparison.values()]
        })
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=selected_metric,
            color='Model',
            title=f'Model Comparison by {selected_metric}'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Compare predictions
        st.subheader("Prediction Comparison")
        
        # Select engine to compare
        engine_selector = st.selectbox(
            "Select Engine ID for Prediction Comparison",
            np.unique(data['engine_ids'])
        )
        
        # Filter data for selected engine
        engine_idx = np.where(data['engine_ids'] == engine_selector)[0][0]
        true_rul = data['y_test'][engine_idx]
        
        # Compare predictions for this engine
        model_predictions = {}
        for model_name, result in comparison_results.items():
            model_predictions[model_name] = result['y_pred'][engine_idx][0]
        
        # Display comparison
        prediction_df = pd.DataFrame({
            'Model': list(model_predictions.keys()),
            'Predicted RUL': list(model_predictions.values()),
            'True RUL': true_rul,
            'Error': [pred - true_rul for pred in model_predictions.values()]
        })
        
        st.dataframe(prediction_df, hide_index=True)
        
        # Create comparison chart
        fig = px.bar(
            prediction_df,
            x='Model',
            y='Predicted RUL',
            color='Error',
            color_continuous_scale='RdBu_r',
            title=f'RUL Predictions for Engine {engine_selector}'
        )
        
        # Add line for true RUL
        fig.add_hline(
            y=true_rul,
            line_width=2,
            line_dash="dash",
            line_color="black",
            annotation_text=f"True RUL: {true_rul:.1f}",
            annotation_position="bottom right"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # What-If Analysis page
    elif page == "What-If Analysis":
        st.header("🔮 What-If Analysis")
        
        st.markdown("""
        Explore how changes in sensor readings or operational settings affect RUL predictions.
        Adjust the values of different features to see their impact on the predicted RUL.
        """)
        
        # Select engine for what-if analysis
        engine_selector = st.selectbox(
            "Select Engine ID for What-If Analysis",
            np.unique(data['engine_ids'])
        )
        
        # Get data for selected engine
        engine_idx = np.where(data['engine_ids'] == engine_selector)[0][0]
        X_sample = data['X_test'][engine_idx]
        true_rul = data['y_test'][engine_idx]
        pred_rul = y_pred[engine_idx][0]
        
        # Display current values and prediction
        st.subheader(f"Engine {engine_selector} - Current State")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("True RUL", f"{true_rul:.1f} cycles")
        with col2:
            st.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
        with col3:
            error = pred_rul - true_rul
            st.metric("Prediction Error", f"{error:.1f} cycles", delta=f"{-abs(error):.1f}" if abs(error) > 0 else "0")
        
        # Define feature ranges for what-if analysis
        # These could be determined automatically from the dataset
        feature_ranges = {}
        X_train_flat = data['X_train'].reshape(-1, data['X_train'].shape[2])
        
        for i, feature in enumerate(FEATURE_COLUMNS):
            feature_min = X_train_flat[:, i].min()
            feature_max = X_train_flat[:, i].max()
            feature_ranges[feature] = (feature_min, feature_max)
        
        # Allow user to select features for what-if analysis
        selected_features = st.multiselect(
            "Select Features to Analyze",
            FEATURE_COLUMNS,
            default=FEATURE_COLUMNS[:3]  # Start with first three features
        )
        
        if not selected_features:
            st.warning("Please select at least one feature to analyze.")
            return
        
        # Filter feature ranges
        selected_feature_ranges = {f: feature_ranges[f] for f in selected_features}
        
        # Perform what-if analysis
        baseline_pred, results = what_if_analysis(model, X_sample, selected_feature_ranges)
        
        # Display results
        st.subheader("What-If Analysis Results")
        
        for feature in selected_features:
            st.write(f"### Impact of {feature}")
            
            # Plot impact of this feature
            if feature in results:
                fig = plot_what_if_results(
                    feature,
                    results[feature]['values'],
                    results[feature]['predictions'],
                    baseline_pred
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"No what-if analysis available for {feature}.")
    
    # Uncertainty Quantification page
    elif page == "Uncertainty Quantification":
        st.header("🎯 Uncertainty Quantification")
        
        st.markdown("""
        Visualize the uncertainty in RUL predictions. This helps in understanding the confidence 
        level of the model's predictions and making more informed maintenance decisions.
        """)
        
        # Generate confidence intervals
        with st.spinner("Generating confidence intervals..."):
            n_samples = st.slider("Number of Monte Carlo Samples", 10, 200, 100)
            noise_level = st.slider("Noise Level", 0.01, 0.20, 0.05)
            
            mean_pred, lower_bound, upper_bound = generate_confidence_intervals(
                model, data['X_test'], n_samples, noise_level)
        
        # Display aggregate uncertainty metrics
        st.subheader("Uncertainty Metrics")
        col1, col2, col3 = st.columns(3)
        
        # Calculate average confidence interval width
        avg_interval_width = np.mean(upper_bound - lower_bound)
        
        # Calculate percentage of true values within confidence interval
        true_within_interval = np.mean((data['y_test'].flatten() >= lower_bound) & 
                                        (data['y_test'].flatten() <= upper_bound)) * 100
        
        with col1:
            st.metric("Avg. Confidence Interval", f"{avg_interval_width:.2f} cycles")
        with col2:
            st.metric("True Values in Interval", f"{true_within_interval:.1f}%")
        with col3:
            st.metric("Uncertainty Level", "Medium" if avg_interval_width < 30 else "High")
        
        # Plot predictions with uncertainty
        st.subheader("RUL Predictions with Uncertainty Bands")
        uncertainty_fig = plot_predictions_with_uncertainty(
            data['y_test'], mean_pred, lower_bound, upper_bound)
        st.plotly_chart(uncertainty_fig, use_container_width=True)
        
        # Engine-specific uncertainty
        st.subheader("Engine-specific Uncertainty")
        
        # Select engine for detailed uncertainty analysis
        engine_selector = st.selectbox(
            "Select Engine ID",
            np.unique(data['engine_ids']),
            key='uncertainty_engine_selector'
        )
        
        # Get data for selected engine
        engine_idx = np.where(data['engine_ids'] == engine_selector)[0][0]
        true_rul = data['y_test'][engine_idx]
        pred_rul = mean_pred[engine_idx]
        lower = lower_bound[engine_idx]
        upper = upper_bound[engine_idx]
        
        # Display uncertainty for selected engine
        st.write(f"### Engine {engine_selector} Uncertainty Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("True RUL", f"{true_rul:.1f} cycles")
        with col2:
            st.metric("Predicted RUL", f"{pred_rul:.1f} cycles")
        with col3:
            st.metric("Confidence Interval", f"{lower:.1f} to {upper:.1f} cycles")
        
        # Calculate probability of failure within different time frames
        if st.checkbox("Show Failure Probability Analysis"):
            st.write("### Failure Probability Analysis")
            st.write("Probability of failure within different time frames:")
            
            # Assume normal distribution for simplicity
            mean = pred_rul
            std = (upper - lower) / 3.92  # 95% confidence interval
            
            time_frames = [10, 20, 30, 50, 100]
            probabilities = []
            
            for t in time_frames:
                # P(RUL <= t)
                prob = stats.norm.cdf(t, loc=mean, scale=std)
                probabilities.append(prob * 100)
            
            prob_df = pd.DataFrame({
                'Time Frame (cycles)': time_frames,
                'Failure Probability (%)': probabilities
            })
            
            st.dataframe(prob_df, hide_index=True)
            
            # Plot probability curve
            time_range = np.linspace(0, 150, 100)
            failure_probs = stats.norm.cdf(time_range, loc=mean, scale=std) * 100
            
            prob_curve_df = pd.DataFrame({
                'RUL (cycles)': time_range,
                'Failure Probability (%)': failure_probs
            })
            
            prob_fig = px.line(
                prob_curve_df,
                x='RUL (cycles)',
                y='Failure Probability (%)',
                title=f'Cumulative Failure Probability for Engine {engine_selector}'
            )
            
            st.plotly_chart(prob_fig, use_container_width=True)


if __name__ == "__main__":
    main()
