"""
LSTM (Long Short-Term Memory) model for time series price prediction.
Uses TensorFlow/Keras to predict future price movements.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    tf = None
    Sequential = None
    load_model = None
    LSTM = None
    Dense = None
    Dropout = None
    Input = None
    EarlyStopping = None

from config.settings import settings


class LSTMModel:
    """
    LSTM neural network for price sequence prediction.
    Predicts whether price will go up or down in the next N periods.
    """

    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.model: Optional[Sequential] = None
        self.model_path = os.path.join(settings.ml_model_path, 'lstm_model.h5')
        self.scaler_path = os.path.join(settings.ml_model_path, 'lstm_scaler.pkl')
        self.is_trained = False

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            # Predict if price goes up (1) or down (0) after sequence
            y.append(1 if data[i, 0] > data[i - 1, 0] else 0)

        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture."""
        if not _TF_AVAILABLE:
            raise ImportError("TensorFlow is not available. Install tensorflow to use LSTM.")
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare OHLCV data for LSTM."""
        from sklearn.preprocessing import MinMaxScaler

        # Use close price, volume, and a few indicators
        data_cols = ['Close', 'Volume', 'RSI_14', 'ATR_14', 'BB_Width']
        available = [c for c in data_cols if c in df.columns]
        data = df[available].dropna().values

        # Normalize
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Save scaler
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        X, y = self._create_sequences(scaled_data)
        return X, y

    def train(self, df: pd.DataFrame, epochs: int = 50, validation_split: float = 0.2):
        """Train the LSTM model."""
        if not _TF_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping LSTM training.")
            return
        X, y = self.prepare_data(df)
        if len(X) < 100:
            logger.warning(f"Not enough sequences for LSTM: {len(X)}")
            return

        # Build model
        input_shape = (X.shape[1], X.shape[2])
        self.model = self._build_model(input_shape)

        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        self.is_trained = True
        final_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        logger.info(f"✅ LSTM trained - Train Acc: {final_acc:.2%}, Val Acc: {val_acc:.2%}")

        self.save()

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict next price movement.
        Returns: {'direction': 'up'/'down', 'probability': float, 'confidence': float}
        """
        if not _TF_AVAILABLE:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}
        if not self.is_trained or self.model is None:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        from sklearn.preprocessing import MinMaxScaler

        # Load scaler
        try:
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        # Prepare latest sequence
        data_cols = ['Close', 'Volume', 'RSI_14', 'ATR_14', 'BB_Width']
        available = [c for c in data_cols if c in df.columns]

        if len(df) < self.sequence_length:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        latest_data = df[available].tail(self.sequence_length).values
        if len(latest_data) < self.sequence_length:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        # Scale
        scaled = scaler.transform(latest_data)
        X_pred = scaled.reshape(1, self.sequence_length, len(available))

        # Predict
        proba = float(self.model.predict(X_pred, verbose=0)[0][0])
        confidence = abs(proba - 0.5) * 2

        direction = 'up' if proba > 0.55 else ('down' if proba < 0.45 else 'neutral')

        return {
            'direction': direction,
            'probability': float(proba),
            'confidence': float(confidence),
        }

    def save(self):
        """Save model to disk."""
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"LSTM model saved to {self.model_path}")

    def load(self) -> bool:
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"LSTM model not found at {self.model_path}")
            return False
        try:
            self.model = load_model(self.model_path)
            self.is_trained = True
            logger.info(f"✅ LSTM model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False