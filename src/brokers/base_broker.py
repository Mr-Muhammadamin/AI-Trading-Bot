"""
Abstract base broker interface for all exchange/broker integrations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd


class Order:
    """Represents a trading order."""
    def __init__(self, symbol: str, order_type: str, volume: float,
                 price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                 comment: str = ""):
        self.symbol = symbol
        self.order_type = order_type  # 'buy' or 'sell'
        self.volume = volume
        self.price = price
        self.sl = sl
        self.tp = tp
        self.comment = comment


class Position:
    """Represents an open position."""
    def __init__(self, ticket: int, symbol: str, order_type: str,
                 volume: float, open_price: float, current_price: float,
                 profit: float, sl: float = 0.0, tp: float = 0.0):
        self.ticket = ticket
        self.symbol = symbol
        self.order_type = order_type
        self.volume = volume
        self.open_price = open_price
        self.current_price = current_price
        self.profit = profit
        self.sl = sl
        self.tp = tp


class BaseBroker(ABC):
    """Abstract base class for broker integrations."""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the broker/exchange."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the broker/exchange."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> Dict:
        """Get account balance, equity, etc."""
        pass

    @abstractmethod
    def get_rates(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Optional[int]:
        """Place an order. Returns ticket number if successful."""
        pass

    @abstractmethod
    def close_position(self, ticket: int) -> bool:
        """Close an open position by ticket number."""
        pass

    @abstractmethod
    def modify_position(self, ticket: int, sl: float = 0.0, tp: float = 0.0) -> bool:
        """Modify SL/TP for an open position."""
        pass

    @abstractmethod
    def get_open_positions(self, symbol: str = "") -> List[Position]:
        """Get all open positions, optionally filtered by symbol."""
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol specifications (spread, lot size, etc.)."""
        pass