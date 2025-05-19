"""
LSTM model with attention mechanism for RUL prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Attention, LayerNormalization, Concatenate
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
import os
import logging
from typing import Tuple, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from src.config import (
    MODELS_DIR, SEQUENCE_LENGTH, BATCH_SIZE, 
    LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttentionBlock(tf.keras.layers.Layer):
    """
    Attention mechanism layer for LSTM.
    
    This layer computes attention weights for each timestep in the sequence,
    allowing the model to focus on the most relevant parts of the input sequence.
    """
    
    def __init__(self, units, **kwargs):
        """
        Initialize the attention block.
        
        Args:
            units: Number of units in the attention layer
        """
        super(AttentionBlock, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units, name='attention_W1')
        self.W2 = Dense(units, name='attention_W2')
        self.V = Dense(1, name='attention_V')
    
    def call(self, features, hidden):
        """
        Compute attention weights.
        
        Args:
            features: Sequence of features (output from LSTM)
            hidden: Hidden state
            
        Returns:
            Context vector and attention weights
        """
        # hidden shape = [batch_size, units]
        # hidden_with_time_axis shape = [batch_size, 1, units]
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape = [batch_size, max_length, 1]
        # we get 1 at the last axis because we are applying score to self.V
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        ))
        
        # attention_weights shape = [batch_size, max_length, 1]
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape = [batch_size, hidden_size]
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights

    def get_config(self):
        """Returns the config of the layer."""
        config = super(AttentionBlock, self).get_config()
        config.update({
            'units': self.units
        })
        return config


class RULPredictor:
    """
    LSTM-based model for Remaining Useful Life (RUL) prediction.
    
    This class provides functionality to build, train, and evaluate an LSTM model
    with attention mechanism for RUL prediction.
    """
    
    def __init__(
        self, 
        sequence_length: int = SEQUENCE_LENGTH,
        n_features: int = len(FEATURE_COLUMNS),
        lstm_units: int = 64,
        attention_units: int = 32,
        dropout_rate: float = 0.3,
        learning_rate: float = LEARNING_RATE
    ):
        """
        Initialize the RUL predictor.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: Number of units in LSTM layers
            attention_units: Number of units in attention layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model: Optional[Model] = None # model will now output [predictions, attention_weights]
        self.history: Optional[Dict] = None
        # self.attention_model will no longer be separate if main model outputs attention
        
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_dir = MODELS_DIR / f"lstm_rul_{self.timestamp}"
        self.model_dir.mkdir(exist_ok=True, parents=True)

    def build_model(self) -> Model:
        inputs = Input(shape=(self.sequence_length, self.n_features), name="input_layer")
        
        lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True), name="bi_lstm_1")(inputs)
        lstm1 = BatchNormalization(name="bn_1")(lstm1)
        lstm1 = Dropout(self.dropout_rate, name="dropout_1")(lstm1)
        
        lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=True), name="bi_lstm_2")(lstm1)
        lstm2 = BatchNormalization(name="bn_2")(lstm2)
        lstm2 = Dropout(self.dropout_rate, name="dropout_2")(lstm2)
        
        lstm3, forward_h, forward_c, backward_h, backward_c = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, return_state=True), name="bi_lstm_3_stateful"
        )(lstm2)
        
        state_h = Concatenate(name="concat_hidden_states")([forward_h, backward_h])
        
        attention = AttentionBlock(self.attention_units, name="attention_block")
        context_vector, attention_weights_raw = attention(lstm3, state_h) # Raw attention weights
        
        # Ensure attention_weights has a clear name for later retrieval if needed by explainer
        attention_weights = tf.keras.layers.Activation('linear', name='attention_output')(attention_weights_raw)

        x = Dense(32, activation='relu', name="dense_1")(context_vector)
        x = Dropout(self.dropout_rate, name="dropout_3")(x)
        rul_output = Dense(1, activation='linear', name="rul_output")(x)
        
        # Model now has two outputs: RUL prediction and attention weights
        model = Model(inputs=inputs, outputs=[rul_output, attention_weights], name="RUL_Model_With_Attention")
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={'rul_output': 'mean_squared_error', 'attention_output': None}, # No loss for attention weights
            loss_weights={'rul_output': 1.0, 'attention_output': 0.0}, # Ensure only RUL loss affects training
            metrics={'rul_output': ['mae', 'mse']} # Metrics only for RUL output
        )
        
        model.summary(line_length=120)
        self.model = model
        return model
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, # This is y_rul_train
        X_val: np.ndarray,
        y_val: np.ndarray,   # This is y_rul_val
        batch_size: int = BATCH_SIZE,
        epochs: int = NUM_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE
    ) -> Dict:
        if self.model is None:
            self.build_model()
        
        # Since model has two outputs, y_train and y_val need to match this structure.
        # We don't have "true" attention weights to compare against, so use dummy targets for attention.
        # The loss for attention_output is None and weight is 0.0, so these dummy targets won't affect training.
        dummy_attention_train = np.zeros((X_train.shape[0], X_train.shape[1], 1)) # Shape matches attention_weights
        dummy_attention_val = np.zeros((X_val.shape[0], X_val.shape[1], 1))

        y_train_dict = {'rul_output': y_train, 'attention_output': dummy_attention_train}
        y_val_dict = {'rul_output': y_val, 'attention_output': dummy_attention_val}

        callbacks = [
            EarlyStopping(
                monitor='val_rul_output_loss', # Monitor loss of the RUL output
                patience=patience,
                verbose=1,
                restore_best_weights=True,
                mode='min'  # Explicitly set mode to min for loss
            ),
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.h5'), # Ensure path is string for TF
                monitor='val_rul_output_loss', # Monitor loss of the RUL output
                save_best_only=True,
                verbose=1,
                mode='min'  # Explicitly set mode to min for loss
            ),
            ReduceLROnPlateau(
                monitor='val_rul_output_loss', # Monitor loss of the RUL output
                factor=0.5,
                patience=patience // 2,
                verbose=1,
                min_lr=1e-6,
                mode='min'  # Explicitly set mode to min for loss
            ),
            TensorBoard(
                log_dir=str(self.model_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        logger.info("Starting model training...")
        history = self.model.fit(
            X_train, y_train_dict,
            validation_data=(X_val, y_val_dict),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        try:
            self.model.save(str(self.model_dir / 'final_model.h5'))
            logger.info(f"Model saved to {self.model_dir}")
            with open(self.model_dir / 'model_architecture.json', 'w') as f:
                f.write(self.model.to_json())
        except IOError as e:
            logger.error(f"Error saving model or architecture: {e}")
            
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict: # y_test is y_rul_test
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        
        logger.info("Evaluating model on test data...")
        
        # Prepare y_test_dict for evaluation, similar to training
        dummy_attention_test = np.zeros((X_test.shape[0], X_test.shape[1], 1))
        y_test_dict = {'rul_output': y_test, 'attention_output': dummy_attention_test}

        results = self.model.evaluate(X_test, y_test_dict, verbose=1)
        
        metrics = {}
        # Results array contains [total_loss, rul_output_loss, rul_output_mae, rul_output_mse, (attention_loss if it had one)]
        # We need to map them correctly based on model.metrics_names
        # model.metrics_names would be like: ['loss', 'rul_output_loss', 'attention_output_loss', 'rul_output_mae', 'rul_output_mse']
        # (attention_output_loss might be present even if weight is 0)
        
        # More robust mapping:
        metric_names = self.model.metrics_names
        for name, value in zip(metric_names, results):
            if name == 'loss': # Overall weighted loss
                 metrics['total_loss'] = value
            elif name.startswith('rul_output_'): # Metrics specific to RUL
                metrics[name.replace('rul_output_', '')] = value # e.g. loss, mae, mse
        
        # Make predictions to calculate RMSE separately if not directly available or to ensure consistency
        # model.predict gives [rul_predictions, attention_weights_predictions]
        rul_predictions, _ = self.model.predict(X_test)
        rul_predictions = rul_predictions.flatten()
        
        rmse = np.sqrt(np.mean((y_test.flatten() - rul_predictions) ** 2))
        metrics['rmse'] = rmse
        
        for name, value in metrics.items():
            logger.info(f"{name.upper()}: {value:.4f}")
            
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray: # Returns only RUL predictions
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        rul_predictions, _ = self.model.predict(X)
        return rul_predictions.flatten()
    
    def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been built or loaded yet.")
        # Model already outputs [rul_predictions, attention_weights]
        rul_predictions, attention_weights = self.model.predict(X)
        return rul_predictions.flatten(), attention_weights

    def load_model(self, model_path: Union[str, Path]) -> None:
        model_path_str = str(model_path) # Ensure it's a string
        logger.info(f"Loading model from {model_path_str}")
        try:
            self.model = tf.keras.models.load_model(
                model_path_str,
                custom_objects={'AttentionBlock': AttentionBlock}
                # No compile=False needed here, as model was saved with optimizer state
                # and compile info should be fine with named outputs.
            )
            # self.attention_model is no longer separate
            logger.info("Model loaded successfully")
        except Exception as e: # Catch a broader range of TF loading errors
            logger.error(f"Failed to load model from {model_path_str}: {e}", exc_info=True)
            self.model = None # Ensure model is None if loading failed
            raise # Re-raise the exception so caller knows


if __name__ == "__main__":
    # Simple test to verify the model builds correctly
    predictor = RULPredictor()
    predictor.build_model()
    
    # Create some dummy data
    X_dummy = np.random.rand(100, SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    y_rul_dummy = np.random.rand(100) # RUL targets
    
    logger.info("Testing model training...")
    predictor.train(
        X_dummy, y_rul_dummy,
        X_dummy[:20], y_rul_dummy[:20], # Dummy validation data
        batch_size=32,
        epochs=2,
        patience=1
    )
    
    logger.info("Testing model prediction...")
    rul_preds = predictor.predict(X_dummy[:5])
    print("RUL Predictions:", rul_preds)
    
    logger.info("Testing model prediction with attention...")
    rul_preds_att, att_weights = predictor.predict_with_attention(X_dummy[:5])
    print("RUL Predictions (from predict_with_attention):", rul_preds_att)
    print("Attention weights shape:", att_weights.shape)

    logger.info("Testing model evaluation...")
    eval_metrics = predictor.evaluate(X_dummy[:10], y_rul_dummy[:10])
    print("Evaluation metrics:", eval_metrics)

    # Test loading the saved model
    if predictor.model is not None:
        saved_model_path = predictor.model_dir / 'final_model.h5'
        if saved_model_path.exists():
            logger.info(f"Attempting to load saved model from: {saved_model_path}")
            loaded_predictor = RULPredictor( # Create new instance to avoid state interference
                sequence_length=predictor.sequence_length,
                n_features=predictor.n_features,
                lstm_units=predictor.lstm_units,
                attention_units=predictor.attention_units
            )
            try:
                loaded_predictor.load_model(saved_model_path)
                logger.info("Successfully loaded and tested the saved model.")
                rul_preds_loaded = loaded_predictor.predict(X_dummy[:5])
                print("Predictions from loaded model:", rul_preds_loaded)
                assert np.allclose(rul_preds, rul_preds_loaded[:len(rul_preds)]), "Mismatch in predictions from original and loaded model"

            except Exception as e:
                logger.error(f"Error during loading or testing loaded model: {e}", exc_info=True)
        else:
            logger.warning(f"Saved model file not found at {saved_model_path}, skipping load test.")
