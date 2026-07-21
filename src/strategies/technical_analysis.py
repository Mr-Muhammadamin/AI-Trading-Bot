"""
Advanced Technical Analysis module.
Computes 50+ indicators for multi-timeframe analysis.
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional
from loguru import logger


class TechnicalAnalysis:
    """
    Computes technical indicators for trading decisions.
    Uses pandas_ta for reliable indicator calculations.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators on the dataframe."""
        if df.empty or len(df) < 50:
            return df

        df = df.copy()

        # === TREND INDICATORS ===
        # Moving Averages
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)

        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
            df['MACD_Hist'] = macd['MACDh_12_26_9']

        # ADX (Trend Strength)
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['ADX_Pos'] = adx['DMP_14']
            df['ADX_Neg'] = adx['DMN_14']

        # Parabolic SAR
        psar = ta.psar(df['High'], df['Low'], df['Close'])
        if psar is not None:
            df['PSAR'] = psar['PSARl_0.02_0.2']
            df['PSAR_Dir'] = psar['PSARs_0.02_0.2']

        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None and isinstance(ichimoku, tuple):
            df['Ichimoku_A'] = ichimoku[0]['ISA_9']
            df['Ichimoku_B'] = ichimoku[0]['ISB_26']
            df['Ichimoku_Base'] = ichimoku[0]['ITS_9']
            df['Ichimoku_SpanA'] = ichimoku[0]['IKS_26']

        # === MOMENTUM INDICATORS ===
        # RSI
        df['RSI_14'] = ta.rsi(df['Close'], length=14)

        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['Stoch_K'] = stoch['STOCHk_14_3_3']
            df['Stoch_D'] = stoch['STOCHd_14_3_3']

        # Williams %R
        df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'])

        # CCI
        df['CCI_20'] = ta.cci(df['High'], df['Low'], df['Close'])

        # ROC (Rate of Change)
        df['ROC_10'] = ta.roc(df['Close'], length=10)

        # Money Flow Index
        df['MFI_14'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])

        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        bb = ta.bbands(df['Close'])
        if bb is not None:
            # Find BB column names dynamically (version compatibility)
            bb_cols = list(bb.columns)
            bb_upper_col = [c for c in bb_cols if 'BBU' in c][0]
            bb_mid_col = [c for c in bb_cols if 'BBM' in c][0]
            bb_low_col = [c for c in bb_cols if 'BBL' in c][0]
            df['BB_Upper'] = bb[bb_upper_col]
            df['BB_Middle'] = bb[bb_mid_col]
            df['BB_Lower'] = bb[bb_low_col]
            df['BB_Width'] = (bb[bb_upper_col] - bb[bb_low_col]) / bb[bb_mid_col]
            df['BB_Percent'] = (df['Close'] - bb[bb_low_col]) / (bb[bb_upper_col] - bb[bb_low_col])

        # ATR (Average True Range)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Percent'] = df['ATR_14'] / df['Close'] * 100

        # Keltner Channels
        kc = ta.kc(df['High'], df['Low'], df['Close'])
        if kc is not None:
            kc_cols = list(kc.columns)
            kc_upper_col = [c for c in kc_cols if 'KCU' in c][0]
            kc_lower_col = [c for c in kc_cols if 'KCL' in c][0]
            df['KC_Upper'] = kc[kc_upper_col]
            df['KC_Lower'] = kc[kc_lower_col]

        # === VOLUME INDICATORS ===
        # OBV (On-Balance Volume)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        # Volume SMA
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

        # === PATTERN RECOGNITION ===
        # Price patterns
        df['Higher_High'] = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))

        # Candle patterns
        df['Doji'] = abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1
        df['Bullish_Engulfing'] = (df['Close'] > df['Open']) & \
                                   (df['Close'].shift(1) < df['Open'].shift(1)) & \
                                   (df['Open'] < df['Close'].shift(1)) & \
                                   (df['Close'] > df['Open'].shift(1))
        df['Bearish_Engulfing'] = (df['Close'] < df['Open']) & \
                                   (df['Close'].shift(1) > df['Open'].shift(1)) & \
                                   (df['Open'] > df['Close'].shift(1)) & \
                                   (df['Close'] < df['Open'].shift(1))

        # === CUSTOM SIGNALS ===
        # Trend direction
        df['Trend_MA'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        df['Trend_MACD'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)

        # Momentum signals
        df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)

        # Volatility regime
        df['High_Volatility'] = (df['ATR_Percent'] > df['ATR_Percent'].rolling(50).mean()).astype(int)

        # Support/Resistance levels (simplified)
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        df['Near_Resistance'] = (df['Close'] >= df['Resistance'] * 0.99).astype(int)
        df['Near_Support'] = (df['Close'] <= df['Support'] * 1.01).astype(int)

        return df

    @staticmethod
    def get_signal(df: pd.DataFrame) -> Dict:
        """
        Generate a trading signal based on technical analysis.
        Returns: {'signal': 'buy'/'sell'/'neutral', 'strength': 0-100, 'reason': str}
        """
        if df.empty or len(df) < 50:
            return {'signal': 'neutral', 'strength': 0, 'reason': 'Insufficient data'}

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        reasons = []

        # === TREND SCORING ===
        # MA Trend
        if latest['SMA_20'] > latest['SMA_50']:
            score += 10
            reasons.append("Bullish MA trend")
        else:
            score -= 10
            reasons.append("Bearish MA trend")

        # MACD
        if latest['MACD'] > latest['MACD_Signal']:
            score += 8
            reasons.append("MACD bullish")
        else:
            score -= 8
            reasons.append("MACD bearish")

        # ADX (strong trend = more confidence)
        if latest['ADX'] > 25:
            if latest['ADX_Pos'] > latest['ADX_Neg']:
                score += 5
            else:
                score -= 5

        # === MOMENTUM SCORING ===
        # RSI
        if latest['RSI_14'] < 30:
            score += 12
            reasons.append("RSI oversold")
        elif latest['RSI_14'] > 70:
            score -= 12
            reasons.append("RSI overbought")
        elif 40 < latest['RSI_14'] < 60:
            # Neutral RSI = no momentum signal
            pass

        # Stochastic
        if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
            score += 8
            reasons.append("Stochastic oversold")
        elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
            score -= 8
            reasons.append("Stochastic overbought")

        # CCI
        if latest['CCI_20'] < -100:
            score += 6
        elif latest['CCI_20'] > 100:
            score -= 6

        # === VOLATILITY SCORING ===
        # Bollinger Bands
        if latest['Close'] <= latest['BB_Lower']:
            score += 8
            reasons.append("Price at BB lower")
        elif latest['Close'] >= latest['BB_Upper']:
            score -= 8
            reasons.append("Price at BB upper")

        # === VOLUME CONFIRMATION ===
        if latest['Volume_Ratio'] > 1.5:
            if score > 0:
                score += 5
                reasons.append("High volume bullish")
            elif score < 0:
                score -= 5
                reasons.append("High volume bearish")

        # === SUPPORT/RESISTANCE ===
        if latest['Near_Support'] == 1:
            score += 6
            reasons.append("Near support level")
        if latest['Near_Resistance'] == 1:
            score -= 6
            reasons.append("Near resistance level")

        # === CANDLE PATTERNS ===
        if latest['Bullish_Engulfing']:
            score += 10
            reasons.append("Bullish engulfing")
        if latest['Bearish_Engulfing']:
            score -= 10
            reasons.append("Bearish engulfing")

        # Determine final signal - scale strength for more active trading
        strength = min(abs(score), 100)
        if score > 0:
            signal = 'buy'
        elif score < 0:
            signal = 'sell'
        else:
            signal = 'neutral'

        return {
            'signal': signal,
            'strength': strength,
            'score': score,
            'reason': ' | '.join(reasons[:3]) if reasons else 'No clear signal'
        }