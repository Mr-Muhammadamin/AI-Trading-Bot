"""
Smart Position Manager - Active position monitoring and management.
Instead of waiting for SL/TP, this module actively monitors open positions
and makes intelligent decisions: trailing SL, early exits on reversals,
partial profit taking, and dynamic adjustments based on market conditions.
"""
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from src.brokers.base_broker import Position, Order
from src.data.data_collector import DataCollector
from config.settings import settings


class SmartPositionManager:
    """
    Actively monitors and manages open positions.
    
    Features:
    - Breakeven SL: When profit reaches threshold, move SL to breakeven
    - Trailing Stop: Dynamic trailing SL that locks profits
    - Partial Close: Take profit on part of position when strong moves occur
    - Reversal Detection: Close early if market reverses against position
    - Volatility-based SL Adjustment: Widen/tighten SL based on ATR
    - Profit Target: Close when profit target is hit (don't wait for TP)
    """

    def __init__(self, broker, data_collector: DataCollector, ensemble=None, telegram=None):
        self.broker = broker
        self.data_collector = data_collector
        self.ensemble = ensemble
        self.telegram = telegram

        # === CONFIGURABLE PARAMETERS ===
        # Breakeven
        self.be_profit_pips = 15          # Pips profit to move SL to breakeven
        self.be_profit_percent = 0.3      # Or 0.3% price move to breakeven

        # Trailing Stop
        self.trail_activation_pips = 25   # Pips profit before trailing activates
        self.trail_distance_pips = 15     # Distance trailing SL follows price
        self.trail_distance_atr_mult = 1.0  # Or use ATR multiplier

        # Partial Profit Taking
        self.partial_profit_pips = 30     # Take partial profit at this level
        self.partial_close_ratio = 0.3    # Close 30% of position
        self.partial_profit2_pips = 50    # Second partial level
        self.partial_close2_ratio = 0.3   # Close another 30%

        # Reversal Detection
        self.reversal_check_enabled = True
        self.reversal_confidence_threshold = 0.15  # Close if opposite signal > this
        self.reversal_profit_needed = 0.001        # Need at least 0.1% profit to close on reversal

        # Profit Taking (early exit when profitable)
        self.profit_target_pips = 40      # Close entirely at this profit
        self.profit_target_percent = 0.8  # 80% of TP distance

        # Volatility Adjustment
        self.sl_adjust_based_on_atr = True
        self.sl_atr_multiplier = 1.5      # Minimum SL distance in ATR

        # Monitoring interval
        self.check_interval_seconds = 5   # How often to check positions

        # Track positions we already managed (avoid duplicate actions)
        self._managed_positions: Dict[int, Dict] = {}

    def check_positions(self, trading_mode: str = 'normal'):
        """
        Main entry point - check and manage all open positions.
        Called periodically from the trading loop.
        """
        try:
            open_positions = self.broker.get_open_positions()
            if not open_positions:
                return

            for position in open_positions:
                self._manage_position(position, trading_mode)

        except Exception as e:
            logger.error(f"[SMART_MGR] Error checking positions: {e}")

    def _manage_position(self, position: Position, trading_mode: str):
        """
        Manage a single open position with all smart features.
        """
        ticket = position.ticket
        symbol = position.symbol
        order_type = position.order_type  # 'buy' or 'sell'
        profit = position.profit
        open_price = position.open_price
        current_price = position.current_price
        current_sl = position.sl
        current_tp = position.tp

        # Get position tracking state
        pos_state = self._managed_positions.get(ticket, {
            'breakeven_set': False,
            'trailing_active': False,
            'highest_price': current_price if order_type == 'buy' else 0,
            'lowest_price': current_price if order_type == 'sell' else float('inf'),
            'max_profit': profit,
            'partial_closed': False,
            'partial2_closed': False,
        })

        # Update tracked extremes
        if order_type == 'buy':
            pos_state['highest_price'] = max(pos_state['highest_price'], current_price)
            if profit > pos_state['max_profit']:
                pos_state['max_profit'] = profit
        else:  # sell
            pos_state['lowest_price'] = min(pos_state['lowest_price'], current_price)
            if profit > pos_state['max_profit']:
                pos_state['max_profit'] = profit

        # Get symbol info for pip calculations
        symbol_info = self.broker.get_symbol_info(symbol)
        point = symbol_info.get('point', 0.00001)
        digits = symbol_info.get('digits', 5)

        # Calculate price move in pips and percent
        pip_value = self._get_pip_value(symbol, point, digits)
        if order_type == 'buy':
            price_move = current_price - open_price
            pips_profit = price_move / pip_value if pip_value > 0 else 0
            profit_percent = (current_price - open_price) / open_price
        else:  # sell
            price_move = open_price - current_price
            pips_profit = price_move / pip_value if pip_value > 0 else 0
            profit_percent = (open_price - current_price) / open_price

        logger.debug(f"[SMART_MGR] {symbol} {order_type} ticket={ticket} "
                     f"profit=${profit:.2f} pips={pips_profit:.1f} "
                     f"SL={current_sl} TP={current_tp}")

        # === 1. BREAKEVEN LOGIC ===
        if not pos_state['breakeven_set'] and profit > 0:
            if pips_profit >= self.be_profit_pips or profit_percent >= self.be_profit_percent:
                self._set_breakeven_sl(position, current_sl, pip_value)
                pos_state['breakeven_set'] = True
                logger.info(f"[SMART_MGR] ✅ {symbol} ticket={ticket}: Breakeven SL set")
                return  # Wait for next cycle

        # === 2. TRAILING STOP LOGIC ===
        if pips_profit >= self.trail_activation_pips:
            # Calculate new trailing SL
            if order_type == 'buy':
                new_sl = current_price - (self.trail_distance_pips * pip_value)
            else:  # sell
                new_sl = current_price + (self.trail_distance_pips * pip_value)

            # Round to valid price
            new_sl = round(new_sl, digits)

            # Only update if new SL is better than current SL
            if order_type == 'buy' and new_sl > current_sl:
                self._modify_position_sl(position, new_sl, f"Trailing SL (profit {pips_profit:.0f}p)")
                pos_state['trailing_active'] = True
                logger.info(f"[SMART_MGR] 🔄 {symbol} ticket={ticket}: Trailing SL -> {new_sl}")
            elif order_type == 'sell' and (current_sl == 0 or new_sl < current_sl):
                if current_sl == 0 or new_sl < current_sl:
                    self._modify_position_sl(position, new_sl, f"Trailing SL (profit {pips_profit:.0f}p)")
                    pos_state['trailing_active'] = True
                    logger.info(f"[SMART_MGR] 🔄 {symbol} ticket={ticket}: Trailing SL -> {new_sl}")

        # === 3. PARTIAL PROFIT TAKING ===
        volume = position.volume

        # First partial level
        if pips_profit >= self.partial_profit_pips and not pos_state['partial_closed'] and volume >= 0.02:
            self._close_partial(position, self.partial_close_ratio, 
                                f"Partial TP1 ({pips_profit:.0f}p)")
            pos_state['partial_closed'] = True
            logger.info(f"[SMART_MGR] ✂️ {symbol} ticket={ticket}: Partial close 1 ({self.partial_close_ratio*100:.0f}%)")
            return

        # Second partial level
        if pips_profit >= self.partial_profit2_pips and not pos_state['partial2_closed'] and volume >= 0.02:
            self._close_partial(position, self.partial_close2_ratio,
                                f"Partial TP2 ({pips_profit:.0f}p)")
            pos_state['partial2_closed'] = True
            logger.info(f"[SMART_MGR] ✂️ {symbol} ticket={ticket}: Partial close 2 ({self.partial_close2_ratio*100:.0f}%)")
            return

        # === 4. PROFIT TARGET - close entirely ===
        if pips_profit >= self.profit_target_pips or profit_percent >= self.profit_target_percent:
            if self._close_position_smart(position, f"Profit target hit ({pips_profit:.0f}p)"):
                logger.info(f"[SMART_MGR] 🎯 {symbol} ticket={ticket}: Profit target closed at ${profit:.2f}")
                return

        # === 5. REVERSAL DETECTION ===
        if self.reversal_check_enabled and self.ensemble and profit > 0:
            # Check if ensemble signal has reversed
            should_close, reason = self._check_reversal(symbol, position, pips_profit, profit_percent)
            if should_close:
                if self._close_position_smart(position, reason):
                    logger.info(f"[SMART_MGR] 🔄 {symbol} ticket={ticket}: Reversal close: {reason} (profit=${profit:.2f})")
                    return

        # === 6. VOLATILITY-BASED SL ADJUSTMENT ===
        if self.sl_adjust_based_on_atr and profit > 0:
            self._adjust_sl_for_volatility(position, trading_mode, pip_value, digits)

        # Save updated state
        self._managed_positions[ticket] = pos_state

    def _set_breakeven_sl(self, position: Position, current_sl: float, pip_value: float):
        """Move stop loss to breakeven (entry price)."""
        symbol_info = self.broker.get_symbol_info(position.symbol)
        digits = symbol_info.get('digits', 5)
        
        # Set SL a few pips above breakeven to avoid slippage closing at loss
        safety_pips = 2 * pip_value
        if position.order_type == 'buy':
            new_sl = round(position.open_price + safety_pips, digits)
        else:
            new_sl = round(position.open_price - safety_pips, digits)

        self._modify_position_sl(position, new_sl, "Breakeven SL")

    def _modify_position_sl(self, position: Position, new_sl: float, reason: str):
        """Modify SL on a position."""
        try:
            success = self.broker.modify_position(position.ticket, sl=new_sl)
            if success:
                if self.telegram:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.telegram.send_message(
                                f"🛡️ *SL Updated*\n"
                                f"{position.symbol} {position.order_type.upper()}\n"
                                f"New SL: {new_sl}\n"
                                f"Reason: {reason}"
                            )
                        )
                    except RuntimeError:
                        pass
        except Exception as e:
            logger.error(f"[SMART_MGR] Failed to modify SL for {position.ticket}: {e}")

    def _close_partial(self, position: Position, ratio: float, reason: str):
        """Close a portion of the position."""
        try:
            partial_volume = round(position.volume * ratio, 2)
            if partial_volume <= 0:
                return

            # Close partial amount
            order_type = 'sell' if position.order_type == 'buy' else 'buy'
            order = Order(
                symbol=position.symbol,
                order_type=order_type,
                volume=partial_volume,
                comment=f"Partial_{reason}",
            )
            ticket = self.broker.place_order(order)
            if ticket:
                logger.info(f"[SMART_MGR] ✅ Partial close: {partial_volume} {position.symbol} - {reason}")
                if self.telegram:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.telegram.send_message(
                                f"✂️ *Partial Close*\n"
                                f"{position.symbol} {position.order_type.upper()}\n"
                                f"Volume: {partial_volume} ({ratio*100:.0f}%)\n"
                                f"Reason: {reason}"
                            )
                        )
                    except RuntimeError:
                        pass
        except Exception as e:
            logger.error(f"[SMART_MGR] Partial close error: {e}")

    def _close_position_smart(self, position: Position, reason: str) -> bool:
        """Close an entire position with notification."""
        try:
            success = self.broker.close_position(position.ticket)
            if success:
                logger.info(f"[SMART_MGR] ✅ Closed {position.symbol} ticket={position.ticket}: {reason} "
                           f"(P&L=${position.profit:.2f})")
                if self.telegram:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.telegram.send_message(
                                f"🔒 *Position Closed*\n"
                                f"{position.symbol} {position.order_type.upper()}\n"
                                f"P&L: ${position.profit:.2f}\n"
                                f"Reason: {reason}"
                            )
                        )
                    except RuntimeError:
                        pass
                # Clean up tracking
                self._managed_positions.pop(position.ticket, None)
                return True
            return False
        except Exception as e:
            logger.error(f"[SMART_MGR] Close error: {e}")
            return False

    def _check_reversal(self, symbol: str, position: Position, 
                        pips_profit: float, profit_percent: float) -> Tuple[bool, str]:
        """
        Check if market shows strong reversal signal against our position.
        Uses AI ensemble analysis when available.
        """
        try:
            # Get latest data
            if self.data_collector:
                df = self.data_collector.get_latest(symbol, 'M5', count=50)
                if df.empty or len(df) < 20:
                    return False, ""

                # Quick technical reversal check
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                # Price action reversal signals
                if position.order_type == 'buy':
                    # For buy positions, look for bearish reversal signals
                    bearish_signals = 0
                    # Bearish engulfing
                    if latest['Close'] < latest['Open'] and prev['Close'] > prev['Open']:
                        if latest['Open'] > prev['Close'] and latest['Close'] < prev['Open']:
                            bearish_signals += 1
                    # Shooting star (small body, long upper wick)
                    candle_range = latest['High'] - latest['Low']
                    if candle_range > 0:
                        upper_wick = latest['High'] - max(latest['Open'], latest['Close'])
                        body = abs(latest['Close'] - latest['Open'])
                        if upper_wick > body * 2 and body < candle_range * 0.3:
                            bearish_signals += 1
                    # RSI overbought and turning down
                    if 'RSI_14' in latest and 'RSI_14' in prev:
                        if latest['RSI_14'] > 70 and latest['RSI_14'] < prev['RSI_14']:
                            bearish_signals += 1
                    # Price broke below EMA_20
                    if 'EMA_20' in latest and latest['Close'] < latest['EMA_20']:
                        bearish_signals += 1

                    if bearish_signals >= 2 and pips_profit > 0:
                        return True, f"Reversal ({bearish_signals} bearish signals)"

                else:  # sell position
                    # For sell positions, look for bullish reversal signals
                    bullish_signals = 0
                    # Bullish engulfing
                    if latest['Close'] > latest['Open'] and prev['Close'] < prev['Open']:
                        if latest['Open'] < prev['Close'] and latest['Close'] > prev['Open']:
                            bullish_signals += 1
                    # Hammer (small body, long lower wick)
                    candle_range = latest['High'] - latest['Low']
                    if candle_range > 0:
                        lower_wick = min(latest['Open'], latest['Close']) - latest['Low']
                        body = abs(latest['Close'] - latest['Open'])
                        if lower_wick > body * 2 and body < candle_range * 0.3:
                            bullish_signals += 1
                    # RSI oversold and turning up
                    if 'RSI_14' in latest and 'RSI_14' in prev:
                        if latest['RSI_14'] < 30 and latest['RSI_14'] > prev['RSI_14']:
                            bullish_signals += 1
                    # Price broke above EMA_20
                    if 'EMA_20' in latest and latest['Close'] > latest['EMA_20']:
                        bullish_signals += 1

                    if bullish_signals >= 2 and pips_profit > 0:
                        return True, f"Reversal ({bullish_signals} bullish signals)"

                # Use ensemble AI if available
                if self.ensemble and profit_percent > self.reversal_profit_needed:
                    decision = self.ensemble.analyze(df)
                    if decision['signal'] != 'neutral':
                        # Check if ensemble signal is opposite to our position
                        if position.order_type == 'buy' and decision['signal'] == 'sell':
                            if decision.get('confidence', 0) > self.reversal_confidence_threshold:
                                return True, f"AI reversal signal (conf: {decision['confidence']:.2f})"
                        elif position.order_type == 'sell' and decision['signal'] == 'buy':
                            if decision.get('confidence', 0) > self.reversal_confidence_threshold:
                                return True, f"AI reversal signal (conf: {decision['confidence']:.2f})"

        except Exception as e:
            logger.debug(f"[SMART_MGR] Reversal check error: {e}")

        return False, ""

    def _adjust_sl_for_volatility(self, position: Position, trading_mode: str,
                                  pip_value: float, digits: int):
        """
        Dynamically adjust SL based on current market volatility (ATR).
        Wider SL in high volatility, tighter in low volatility.
        """
        try:
            if not self.data_collector:
                return

            df = self.data_collector.get_latest(position.symbol, 'M5', count=30)
            if df.empty or 'ATR_14' not in df.columns:
                return

            current_atr = float(df.iloc[-1]['ATR_14'])
            current_sl = position.sl
            current_price = position.current_price

            if current_atr <= 0 or current_sl == 0:
                return

            # Calculate minimum SL distance based on ATR
            min_sl_distance = current_atr * self.sl_atr_multiplier

            if position.order_type == 'buy':
                current_sl_distance = current_price - current_sl
                target_sl = current_price - min_sl_distance
                # Only tighten SL if we're in profit (don't widen it)
                if current_sl_distance > min_sl_distance and current_sl_distance > 0:
                    # Can tighten SL
                    if target_sl > current_sl:
                        target_sl = round(target_sl, digits)
                        self._modify_position_sl(position, target_sl, 
                                                  f"ATR adjust (ATR={current_atr:.5f})")
            else:  # sell
                current_sl_distance = current_sl - current_price
                target_sl = current_price + min_sl_distance
                if current_sl_distance > min_sl_distance and current_sl_distance > 0:
                    if current_sl == 0 or target_sl < current_sl:
                        target_sl = round(target_sl, digits)
                        self._modify_position_sl(position, target_sl,
                                                  f"ATR adjust (ATR={current_atr:.5f})")

        except Exception as e:
            logger.debug(f"[SMART_MGR] Volatility adjustment error: {e}")

    def _get_pip_value(self, symbol: str, point: float, digits: int) -> float:
        """Get pip value based on symbol type."""
        # For forex: 1 pip = 0.0001 or 0.01 for JPY pairs
        # For indices, crypto: use point * 10
        if 'JPY' in symbol or 'XAU' in symbol or 'XAG' in symbol:
            return 0.01  # Standard pip for JPY, Gold, Silver
        elif 'BTC' in symbol or 'ETH' in symbol:
            return point * 10  # Crypto
        else:
            return point * 10  # Standard 5-digit forex

    def cleanup_position(self, ticket: int):
        """Remove position tracking when position is closed externally."""
        self._managed_positions.pop(ticket, None)

    def get_status(self) -> str:
        """Get status of smart position management."""
        tracked = len(self._managed_positions)
        active_trails = sum(1 for s in self._managed_positions.values() if s.get('trailing_active'))
        active_be = sum(1 for s in self._managed_positions.values() if s.get('breakeven_set'))

        lines = [
            "🎯 *Smart Position Manager*",
            f"Tracked Positions: {tracked}",
            f"Breakeven Set: {active_be}",
            f"Trailing Active: {active_trails}",
            "",
            "⚙️ Settings:",
            f"Breakeven: {self.be_profit_pips}p",
            f"Trailing: {self.trail_activation_pips}p activate / {self.trail_distance_pips}p distance",
            f"Partial TP: {self.partial_profit_pips}p ({self.partial_close_ratio*100:.0f}%)",
            f"Profit Target: {self.profit_target_pips}p",
            f"Reversal Check: {'✅' if self.reversal_check_enabled else '❌'}",
        ]
        return "\n".join(lines)