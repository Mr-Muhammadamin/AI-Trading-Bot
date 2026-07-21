"""
Ensemble Decision System.
Combines signals from Technical Analysis, XGBoost, LSTM, and RL Agent
into a single trading decision with confidence scoring.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.technical_analysis import TechnicalAnalysis
from src.ml.xgboost_model import XGBoostModel
from src.ml.lstm_model import LSTMModel
from src.rl.rl_agent import RLAgent
from config.settings import settings


class EnsembleDecision:
    """
    Combines multiple AI models and technical analysis into one decision.
    Uses weighted voting system with dynamic confidence thresholds.
    """

    # Weights for each model (can be adjusted based on performance)
    MODEL_WEIGHTS = {
        'technical': 1.0,
        'xgboost': 0.0,
        'lstm': 0.0,
        'rl_agent': 0.0,
    }

    # Minimum confidence threshold to execute a trade
    MIN_CONFIDENCE = 0.01

    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.xgboost = XGBoostModel()
        self.lstm = LSTMModel()
        self.rl_agent = RLAgent()

        # Load pre-trained models if available
        self.xgboost.load()
        self.lstm.load()
        self.rl_agent.load()

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Full market analysis using all models.
        Returns a comprehensive decision with confidence.
        """
        if df.empty or len(df) < 50:
            return self._neutral_decision("Insufficient data")

        # Compute technical indicators
        df_ta = self.ta.compute_all(df)

        # Get signals from all models
        signals = {}

        # 1. Technical Analysis
        ta_signal = self.ta.get_signal(df_ta)
        signals['technical'] = ta_signal
        logger.debug(f"TA Signal: {ta_signal['signal']} (strength: {ta_signal['strength']})")

        # 2. XGBoost
        if settings.use_xgboost:
            xgb_pred = self.xgboost.predict(df_ta)
            signals['xgboost'] = xgb_pred
            logger.debug(f"XGBoost: {xgb_pred['direction']} (conf: {xgb_pred['confidence']:.2f})")

        # 3. LSTM
        if settings.use_lstm:
            lstm_pred = self.lstm.predict(df_ta)
            signals['lstm'] = lstm_pred
            logger.debug(f"LSTM: {lstm_pred['direction']} (conf: {lstm_pred['confidence']:.2f})")

        # 4. RL Agent
        if settings.use_rl_agent:
            rl_obs = self._get_rl_observation(df_ta)
            rl_pred = self.rl_agent.predict_with_confidence(rl_obs)
            signals['rl_agent'] = rl_pred
            logger.debug(f"RL Agent: {rl_pred['action_name']} (conf: {rl_pred['confidence']:.2f})")

        # Combine signals
        decision = self._combine_signals(signals, df_ta)

        return decision

    def _combine_signals(self, signals: Dict, df: pd.DataFrame) -> Dict:
        """
        Combine all signals using weighted voting.
        """
        buy_score = 0.0
        sell_score = 0.0
        total_weight = 0.0
        reasons = []
        model_details = {}

        # Technical Analysis
        if 'technical' in signals:
            weight = self.MODEL_WEIGHTS['technical']
            sig = signals['technical']
            if sig['signal'] == 'buy':
                buy_score += weight * (sig['strength'] / 100)
            elif sig['signal'] == 'sell':
                sell_score += weight * (sig['strength'] / 100)
            total_weight += weight
            reasons.append(f"TA: {sig['signal']} ({sig['reason']})")
            model_details['technical'] = sig

        # XGBoost
        if 'xgboost' in signals:
            weight = self.MODEL_WEIGHTS['xgboost']
            sig = signals['xgboost']
            if sig['direction'] == 'up':
                buy_score += weight * sig['confidence']
            elif sig['direction'] == 'down':
                sell_score += weight * sig['confidence']
            total_weight += weight
            reasons.append(f"XGB: {sig['direction']} ({sig['probability']:.1%})")
            model_details['xgboost'] = sig

        # LSTM
        if 'lstm' in signals:
            weight = self.MODEL_WEIGHTS['lstm']
            sig = signals['lstm']
            if sig['direction'] == 'up':
                buy_score += weight * sig['confidence']
            elif sig['direction'] == 'down':
                sell_score += weight * sig['confidence']
            total_weight += weight
            reasons.append(f"LSTM: {sig['direction']} ({sig['probability']:.1%})")
            model_details['lstm'] = sig

        # RL Agent
        if 'rl_agent' in signals:
            weight = self.MODEL_WEIGHTS['rl_agent']
            sig = signals['rl_agent']
            if sig['action_name'] == 'buy':
                buy_score += weight * sig['confidence']
            elif sig['action_name'] == 'sell':
                sell_score += weight * sig['confidence']
            total_weight += weight
            reasons.append(f"RL: {sig['action_name']} ({sig['confidence']:.1%})")
            model_details['rl_agent'] = sig

        # Normalize scores
        if total_weight > 0:
            buy_score /= total_weight
            sell_score /= total_weight

        # Determine final signal
        net_score = buy_score - sell_score
        confidence = max(buy_score, sell_score)

        if net_score > 0.005 and confidence >= self.MIN_CONFIDENCE:
            signal = 'buy'
            strength = min(buy_score * 100, 100)
        elif net_score < -0.005 and confidence >= self.MIN_CONFIDENCE:
            signal = 'sell'
            strength = min(sell_score * 100, 100)
        else:
            signal = 'neutral'
            strength = max(buy_score, sell_score) * 100

        # Get current price info
        current_price = df.iloc[-1]['Close']
        atr = df.iloc[-1].get('ATR_14', current_price * 0.01)

        return {
            'signal': signal,
            'strength': float(strength),
            'confidence': float(confidence),
            'buy_score': float(buy_score),
            'sell_score': float(sell_score),
            'net_score': float(net_score),
            'current_price': float(current_price),
            'atr': float(atr),
            'reasons': reasons[:5],
            'model_details': model_details,
            'timestamp': pd.Timestamp.now(),
        }

    def _get_rl_observation(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare observation for RL agent from dataframe."""
        indicator_cols = [
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
        available = [c for c in indicator_cols if c in df.columns]
        indicators = df.iloc[-1][available].values
        indicators = np.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)

        # Position info (no position = zeros)
        pos_info = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return np.concatenate([indicators, pos_info]).astype(np.float32)

    def _neutral_decision(self, reason: str) -> Dict:
        """Return a neutral decision."""
        return {
            'signal': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'net_score': 0.0,
            'current_price': 0.0,
            'atr': 0.0,
            'reasons': [reason],
            'model_details': {},
            'timestamp': pd.Timestamp.now(),
        }

    def get_model_performance_summary(self) -> str:
        """Get a summary of which models are active and loaded."""
        lines = ["🤖 AI Models Status:"]
        lines.append(f"  • Technical Analysis: ✅ Active")
        lines.append(f"  • XGBoost: {'✅ Loaded' if self.xgboost.is_trained else '⏳ Not trained'}")
        lines.append(f"  • LSTM: {'✅ Loaded' if self.lstm.is_trained else '⏳ Not trained'}")
        lines.append(f"  • RL Agent: {'✅ Loaded' if self.rl_agent.is_trained else '⏳ Not trained'}")
        return "\n".join(lines)