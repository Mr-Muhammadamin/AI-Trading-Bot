"""
XGBoost model for price direction prediction.
Trained on technical indicators to predict next candle direction.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

from config.settings import settings


class XGBoostModel:
    """
    XGBoost classifier that predicts market direction (up/down).
    Uses technical indicators as features.
    """

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.feature_columns: list = []
        self.is_trained = False
        self.model_path = os.path.join(settings.ml_model_path, 'xgboost_model.pkl')

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns from indicator dataframe."""
        feature_cols = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI_20',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'ATR_14', 'ATR_Percent',
            'OBV', 'Volume_Ratio',
            'ADX', 'ADX_Pos', 'ADX_Neg',
            'MFI_14', 'ROC_10',
            'Trend_MA', 'Trend_MACD',
        ]
        # Only keep columns that exist in the dataframe
        available = [c for c in feature_cols if c in df.columns]
        return df[available].copy()

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for training.
        Label: 1 if next close > current close, else 0.
        """
        df = df.copy()
        features = self._get_features(df)

        # Create labels: predict if next candle will be bullish
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

        # Remove rows with NaN
        valid = features.dropna().index
        features = features.loc[valid]
        labels = df.loc[valid, 'Target']

        # Align
        common_idx = features.index.intersection(labels.index)
        features = features.loc[common_idx]
        labels = labels.loc[common_idx]

        self.feature_columns = features.columns.tolist()
        return features.values, labels.values

    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        """Train the XGBoost model on historical data."""
        X, y = self.prepare_training_data(df)
        if len(X) < 100:
            logger.warning(f"Not enough data for training: {len(X)} samples")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=20,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        self.is_trained = True
        logger.info(f"✅ XGBoost trained - Acc: {accuracy:.2%}, Prec: {precision:.2%}, "
                     f"Recall: {recall:.2%}, F1: {f1:.2%}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = sorted(zip(self.feature_columns, self.model.feature_importances_),
                                 key=lambda x: x[1], reverse=True)
            logger.info("Top 5 features:")
            for name, imp in importances[:5]:
                logger.info(f"  {name}: {imp:.3f}")

        self.save()

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Predict market direction for the latest data point.
        Returns: {'direction': 'up'/'down', 'probability': float, 'confidence': float}
        """
        if not self.is_trained or self.model is None:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        features = self._get_features(df)
        if features.empty:
            return {'direction': 'neutral', 'probability': 0.5, 'confidence': 0}

        latest = features.iloc[-1:].fillna(0)
        proba = self.model.predict_proba(latest)[0]

        # proba[0] = probability of down, proba[1] = probability of up
        up_prob = proba[1]
        confidence = abs(up_prob - 0.5) * 2  # 0 to 1 scale

        direction = 'up' if up_prob > 0.55 else ('down' if up_prob < 0.45 else 'neutral')

        return {
            'direction': direction,
            'probability': float(up_prob),
            'confidence': float(confidence),
        }

    def save(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
            }, f)
        logger.info(f"XGBoost model saved to {self.model_path}")

    def load(self) -> bool:
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"XGBoost model not found at {self.model_path}")
            return False
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
            logger.info(f"✅ XGBoost model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False