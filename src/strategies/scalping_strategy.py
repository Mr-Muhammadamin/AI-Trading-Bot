"""
Scalping Strategy - Tezda kichik foyda olish uchun mo'ljallangan strategiya.
Kichik timeframe (M1, M5) da ishlaydi, 5-10 pip SL/TP bilan.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict
from loguru import logger


class ScalpingStrategy:
    """
    Scalping strategy for quick profits with small SL/TP.
    Uses M1/M5 timeframe, tight stop losses, quick entries.
    """

    def __init__(self):
        self.min_confidence = 0.03  # Scalpingda past confidence bilan ham ochish
        self.max_positions = 5  # Scalpingda ko'proq pozitsiya
        self.sl_pips = 10  # 10 pip stop loss
        self.tp_pips = 15  # 15 pip take profit

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Quick scalping analysis - faqat M1/M5 timeframe uchun.
        """
        if df.empty or len(df) < 20:
            return self._neutral("Insufficient data")

        df_ta = self._compute_indicators(df)
        signal = self._get_signal(df_ta)

        return signal

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute scalping-specific indicators (fast)."""
        df = df.copy()

        # Quick EMAs
        df['EMA_5'] = ta.ema(df['Close'], length=5)
        df['EMA_10'] = ta.ema(df['Close'], length=10)
        df['EMA_20'] = ta.ema(df['Close'], length=20)

        # RSI (faster period)
        df['RSI_7'] = ta.rsi(df['Close'], length=7)

        # Stochastic (fast)
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=5, d=3)
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_5_3_3']
            df['Stoch_D'] = stoch['STOCHd_5_3_3']

        # MACD (faster)
        macd = ta.macd(df['Close'], fast=6, slow=13, signal=5)
        if macd is not None:
            df['MACD'] = macd['MACD_6_13_5']
            df['MACD_Signal'] = macd['MACDs_6_13_5']
            df['MACD_Hist'] = macd['MACDh_6_13_5']

        # ATR for volatility
        df['ATR_5'] = ta.atr(df['High'], df['Low'], df['Close'], length=5)

        # Bollinger Bands (narrow)
        bb = ta.bbands(df['Close'], length=10, std=1.5)
        if bb is not None:
            bb_cols = list(bb.columns)
            bb_upper_col = [c for c in bb_cols if 'BBU' in c][0]
            bb_low_col = [c for c in bb_cols if 'BBL' in c][0]
            bb_mid_col = [c for c in bb_cols if 'BBM' in c][0]
            df['BB_Upper'] = bb[bb_upper_col]
            df['BB_Lower'] = bb[bb_low_col]
            df['BB_Middle'] = bb[bb_mid_col]
            df['BB_Width'] = (bb[bb_upper_col] - bb[bb_low_col]) / bb[bb_mid_col]

        # Volume spike detection
        df['Volume_MA'] = ta.sma(df['Volume'], length=5)
        df['Volume_Spike'] = df['Volume'] / df['Volume_MA'].replace(0, 1)

        return df

    def _get_signal(self, df: pd.DataFrame) -> Dict:
        """Generate scalping signal."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        reasons = []

        # === QUICK TREND ===
        # EMA cross (5/10)
        if latest['EMA_5'] > latest['EMA_10']:
            score += 5
            reasons.append("EMA5>10 bullish")
        else:
            score -= 5
            reasons.append("EMA5<10 bearish")

        # Price vs EMA_20
        if latest['Close'] > latest['EMA_20']:
            score += 3
        else:
            score -= 3

        # === MOMENTUM ===
        # RSI
        if latest['RSI_7'] < 30:
            score += 8
            reasons.append("RSI oversold")
        elif latest['RSI_7'] > 70:
            score -= 8
            reasons.append("RSI overbought")

        # Stochastic cross
        if latest['Stoch_K'] > latest['Stoch_D'] and prev['Stoch_K'] <= prev['Stoch_D']:
            score += 6
            reasons.append("Stoch bullish cross")
        elif latest['Stoch_K'] < latest['Stoch_D'] and prev['Stoch_K'] >= prev['Stoch_D']:
            score -= 6
            reasons.append("Stoch bearish cross")

        # === MACD ===
        if latest['MACD'] > latest['MACD_Signal']:
            score += 4
            reasons.append("MACD bullish")
        else:
            score -= 4
            reasons.append("MACD bearish")

        # === VOLUME ===
        if latest['Volume_Spike'] > 1.5:
            if score > 0:
                score += 4
                reasons.append("High vol bullish")
            else:
                score -= 4
                reasons.append("High vol bearish")

        # === BOLLINGER ===
        if latest['Close'] <= latest['BB_Lower']:
            score += 5
            reasons.append("BB lower bounce")
        elif latest['Close'] >= latest['BB_Upper']:
            score -= 5
            reasons.append("BB upper reject")

        strength = min(abs(score), 100)
        if score > 3:
            signal = 'buy'
        elif score < -3:
            signal = 'sell'
        else:
            signal = 'neutral'

        return {
            'signal': signal,
            'strength': strength,
            'confidence': strength / 100.0,
            'score': score,
            'reason': ' | '.join(reasons[:3]) if reasons else 'No clear signal',
            'current_price': float(df.iloc[-1]['Close']),
            'atr': float(df.iloc[-1].get('ATR_5', df.iloc[-1]['Close'] * 0.001)),
        }

    def _neutral(self, reason: str) -> Dict:
        return {
            'signal': 'neutral',
            'strength': 0,
            'confidence': 0,
            'score': 0,
            'reason': reason,
        }