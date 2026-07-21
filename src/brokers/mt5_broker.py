"""
MetaTrader 5 (Exness) Broker Integration.
Real-time trading with Exness via MT5 API.
"""
from typing import Dict, List, Optional
import pandas as pd
import MetaTrader5 as mt5
from loguru import logger

from src.brokers.base_broker import BaseBroker, Order, Position


class MT5Broker(BaseBroker):
    """MT5 broker implementation for Exness."""

    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
    }

    def __init__(self, account: int, password: str, server: str):
        self.account = account
        self.password = password
        self.server = server
        self._connected = False

    def connect(self) -> bool:
        """Connect to MT5 terminal with Exness credentials."""
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        authorized = mt5.login(self.account, self.password, self.server)
        if not authorized:
            logger.error(f"MT5 login failed for account {self.account}: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self._connected = True
        logger.info(f"✅ Connected to Exness MT5 - Account: {self.account}")
        return True

    def disconnect(self) -> bool:
        """Disconnect from MT5."""
        mt5.shutdown()
        self._connected = False
        logger.info("Disconnected from MT5")
        return True

    def is_connected(self) -> bool:
        return self._connected

    def get_account_info(self) -> Dict:
        """Get account balance, equity, margin, etc."""
        if not self._connected:
            return {}
        info = mt5.account_info()
        if info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return {}
        return {
            'balance': info.balance,
            'equity': info.equity,
            'margin': info.margin,
            'margin_free': info.margin_free,
            'margin_level': info.margin_level if info.margin_level else 100,
            'profit': info.profit,
            'currency': info.currency,
            'leverage': info.leverage,
            'name': info.name,
            'server': info.server,
        }

    def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        tf = self.TIMEFRAME_MAP.get(timeframe, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(f"No rates for {symbol} {timeframe}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        df.set_index('time', inplace=True)
        return df

    def place_order(self, order: Order) -> Optional[int]:
        """Place a market order on MT5."""
        symbol_info = mt5.symbol_info(order.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {order.symbol} not found")
            return None

        if not symbol_info.visible:
            mt5.symbol_select(order.symbol, True)

        order_type = mt5.ORDER_TYPE_BUY if order.order_type == 'buy' else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": float(order.volume),
            "type": order_type,
            "price": mt5.symbol_info_tick(order.symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(order.symbol).bid,
            "sl": float(order.sl) if order.sl > 0 else 0.0,
            "tp": float(order.tp) if order.tp > 0 else 0.0,
            "deviation": 10,
            "magic": 123456,
            "comment": order.comment or "AI Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return None

        logger.info(f"✅ Order placed: {order.order_type} {order.volume} {order.symbol} @ ticket {result.order}")
        return result.order

    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket number."""
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.warning(f"Position {ticket} not found")
            return False

        pos = position[0]
        symbol = pos.symbol
        volume = pos.volume
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "AI Bot Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close position failed: {result.retcode}")
            return False

        logger.info(f"✅ Position {ticket} closed")
        return True

    def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        """Modify SL/TP for an open position."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": float(sl) if sl > 0 else 0.0,
            "tp": float(tp) if tp > 0 else 0.0,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify position failed: {result.retcode}")
            return False
        logger.info(f"✅ Position {ticket} modified: SL={sl}, TP={tp}")
        return True

    def get_open_positions(self, symbol: str = "") -> List[Position]:
        """Get all open positions."""
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append(Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                order_type='buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                volume=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                profit=pos.profit,
                sl=pos.sl,
                tp=pos.tp,
            ))
        return result

    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol specifications."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return {}
        tick = mt5.symbol_info_tick(symbol)
        return {
            'symbol': info.name,
            'spread': info.spread,
            'digits': info.digits,
            'point': info.point,
            'lot_size': info.trade_contract_size,
            'min_volume': info.volume_min,
            'max_volume': info.volume_max,
            'volume_step': info.volume_step,
            'ask': tick.ask if tick else 0,
            'bid': tick.bid if tick else 0,
        }