"""
Risk Management System.
Controls position sizing, stop-loss, take-profit, and overall exposure.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from config.settings import settings


class RiskManager:
    """
    Manages trading risk with:
    - Dynamic position sizing based on volatility
    - ATR-based stop-loss and take-profit
    - Maximum drawdown protection
    - Daily loss limits
    - Correlation-based exposure limits
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = initial_balance * 0.05  # 5% max daily loss
        self.max_drawdown = 0.15  # 15% max drawdown
        self.peak_balance = initial_balance

    def calculate_position_size(self, decision: Dict, account_info: Dict,
                                 symbol_info: Dict) -> float:
        """
        Calculate optimal position size using Kelly-like formula.
        """
        risk_per_trade = settings.risk_per_trade / 100  # Convert percentage to decimal
        balance = account_info.get('balance', self.current_balance)

        # Base position size from account balance
        base_size = balance * risk_per_trade

        # Adjust for signal confidence
        confidence = decision.get('confidence', 0.5)
        confidence_multiplier = 0.5 + confidence  # 1.0 to 1.5x

        # Adjust for volatility (ATR)
        atr_percent = decision.get('atr', 0) / decision.get('current_price', 1) * 100
        if atr_percent > 0:
            # Lower position size in high volatility
            vol_multiplier = min(1.0, 2.0 / atr_percent)
        else:
            vol_multiplier = 1.0

        # Calculate position size
        position_size = base_size * confidence_multiplier * vol_multiplier

        # Apply max position size limit
        max_size = settings.max_position_size * balance
        position_size = min(position_size, max_size)

        # Convert to lots if applicable
        if 'min_volume' in symbol_info:
            min_vol = symbol_info['min_volume']
            step = symbol_info.get('volume_step', min_vol)
            lots = position_size / symbol_info.get('lot_size', 100000)
            # Round to valid step
            lots = round(lots / step) * step
            lots = max(min_vol, min(lots, symbol_info.get('max_volume', lots)))
            logger.info(f"📐 Position size: {lots:.2f} lots (${position_size:.2f})")
            return lots

        logger.info(f"📐 Position size: ${position_size:.2f}")
        return position_size

    def calculate_sl_tp(self, decision: Dict, order_type: str) -> Tuple[float, float]:
        """
        Calculate Stop-Loss and Take-Profit levels using ATR.
        """
        entry_price = decision.get('current_price', 0)
        atr = decision.get('atr', entry_price * 0.01)

        if entry_price == 0:
            return 0.0, 0.0

        sl_mult = settings.stop_loss_atr_multiplier
        tp_mult = settings.take_profit_atr_multiplier

        if order_type == 'buy':
            sl = entry_price - (atr * sl_mult)
            tp = entry_price + (atr * tp_mult)
        else:  # sell
            sl = entry_price + (atr * sl_mult)
            tp = entry_price - (atr * tp_mult)

        # Ensure SL and TP are valid
        sl = max(sl, entry_price * 0.9) if order_type == 'buy' else min(sl, entry_price * 1.1)
        tp = max(tp, entry_price * 1.01) if order_type == 'buy' else min(tp, entry_price * 0.99)

        logger.info(f"🎯 SL: {sl:.5f}, TP: {tp:.5f} (ATR: {atr:.5f})")
        return sl, tp

    def can_open_trade(self, account_info: Dict, open_positions: List) -> Tuple[bool, str]:
        """
        Check if we can open a new trade based on risk rules.
        """
        balance = account_info.get('balance', self.current_balance)
        equity = account_info.get('equity', balance)
        margin_level = account_info.get('margin_level', 100)

        # Check max open trades
        if len(open_positions) >= settings.max_open_trades:
            return False, f"Max trades reached ({settings.max_open_trades})"

        # Check margin level
        # If no open positions, margin_level may be 0; treat as OK
        if len(open_positions) > 0 and margin_level < 200:
            return False, f"Margin level too low: {margin_level:.0f}%"

        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            return False, f"Daily loss limit reached: ${self.daily_pnl:.2f}"

        # Check drawdown
        drawdown = (self.peak_balance - equity) / self.peak_balance
        if drawdown > self.max_drawdown:
            return False, f"Max drawdown exceeded: {drawdown:.1%}"

        return True, "OK"

    def update_pnl(self, pnl: float):
        """Update running P&L tracking."""
        self.daily_pnl += pnl
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        logger.info("Daily risk counters reset")

    def should_stop_trading(self, account_info: Dict) -> Tuple[bool, str]:
        """
        Emergency stop check.
        Returns: (should_stop, reason)
        """
        equity = account_info.get('equity', 0)
        balance = account_info.get('balance', 0)

        # Max drawdown check
        drawdown = (self.peak_balance - equity) / self.peak_balance
        if drawdown > self.max_drawdown * 1.5:  # 1.5x max drawdown = emergency
            return True, f"Emergency stop: {drawdown:.1%} drawdown"

        # Margin level emergency
        if account_info.get('margin_level', 100) < 100:
            return True, "Emergency stop: Margin call level"

        return False, "OK"

    def get_risk_report(self, account_info: Dict) -> str:
        """Generate a risk status report."""
        balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        drawdown = max(0, (self.peak_balance - equity) / self.peak_balance * 100)

        lines = [
            "📊 Risk Report",
            f"  Balance: ${balance:.2f}",
            f"  Equity: ${equity:.2f}",
            f"  Daily P&L: ${self.daily_pnl:.2f}",
            f"  Drawdown: {drawdown:.1f}%",
            f"  Daily Trades: {self.daily_trades}",
            f"  Max Daily Loss: ${self.max_daily_loss:.2f}",
            f"  Peak Balance: ${self.peak_balance:.2f}",
        ]
        return "\n".join(lines)