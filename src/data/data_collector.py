"""
Real-time data collector that fetches market data from broker and stores it.
Supports multiple timeframes and symbols simultaneously.
"""
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
from loguru import logger

from src.brokers.base_broker import BaseBroker
from config.settings import settings


class DataCollector:
    """
    Real-time market data collector.
    Fetches OHLCV data for multiple symbols/timeframes and caches it.
    """

    def __init__(self, broker: BaseBroker):
        self.broker = broker
        self._cache: Dict[str, pd.DataFrame] = {}
        self._running = False
        self._threads: List[threading.Thread] = []
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def start(self, interval_seconds: int = 5):
        """Start collecting data in background threads."""
        self._running = True
        logger.info(f"🚀 Data collector started (interval: {interval_seconds}s)")

        for symbol in settings.symbols_list:
            for tf in settings.timeframes_list:
                thread = threading.Thread(
                    target=self._collect_loop,
                    args=(symbol, tf, interval_seconds),
                    daemon=True,
                    name=f"Data-{symbol}-{tf}"
                )
                self._threads.append(thread)
                thread.start()

    def stop(self):
        """Stop all data collection threads."""
        self._running = False
        for t in self._threads:
            t.join(timeout=2)
        logger.info("Data collector stopped")

    def on_new_data(self, callback: Callable):
        """Register a callback for new data events."""
        self._callbacks.append(callback)

    def get_latest(self, symbol: str, timeframe: str, count: int = 50) -> pd.DataFrame:
        """Get the latest cached data for a symbol/timeframe."""
        key = f"{symbol}_{timeframe}"
        with self._lock:
            if key in self._cache:
                return self._cache[key].tail(count)
            return pd.DataFrame()

    def get_all_cached(self) -> Dict[str, pd.DataFrame]:
        """Get all cached data."""
        with self._lock:
            return dict(self._cache)

    def _collect_loop(self, symbol: str, timeframe: str, interval: int):
        """Background loop that fetches data periodically."""
        key = f"{symbol}_{timeframe}"
        logger.info(f"  → Collecting {symbol} {timeframe}")

        while self._running:
            try:
                # Fetch latest candles
                df = self.broker.get_rates(symbol, timeframe, count=100)
                if df is not None and not df.empty:
                    with self._lock:
                        old_len = len(self._cache.get(key, pd.DataFrame()))
                        self._cache[key] = df

                        # Notify callbacks if new data arrived
                        if len(df) > old_len:
                            for cb in self._callbacks:
                                try:
                                    cb(symbol, timeframe, df)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Data collect error {symbol} {timeframe}: {e}")

            time.sleep(interval)

    def get_historical_data(self, symbol: str, timeframe: str,
                            days: int = 30) -> pd.DataFrame:
        """Fetch historical data for backtesting."""
        bars_count = days * 24 * 60  # approximate
        if timeframe == 'H1':
            bars_count = days * 24
        elif timeframe == 'H4':
            bars_count = days * 6
        elif timeframe == 'D1':
            bars_count = days
        elif timeframe == 'M5':
            bars_count = days * 288
        elif timeframe == 'M15':
            bars_count = days * 96

        return self.broker.get_rates(symbol, timeframe, count=bars_count)