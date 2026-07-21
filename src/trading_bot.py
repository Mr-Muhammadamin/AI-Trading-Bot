"""
Main AI Trading Bot - Orchestrates all components.
Real-time trading with Exness MT5, AI-powered decision making,
and Telegram monitoring.
"""
import asyncio
import time
import threading
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from loguru import logger

from src.brokers.mt5_broker import MT5Broker
from src.brokers.base_broker import Order, Position
from src.data.data_collector import DataCollector
from src.strategies.technical_analysis import TechnicalAnalysis
from src.strategies.ensemble import EnsembleDecision
from src.strategies.scalping_strategy import ScalpingStrategy
from src.ml.xgboost_model import XGBoostModel
from src.ml.lstm_model import LSTMModel
from src.rl.rl_agent import RLAgent
from src.risk.risk_manager import RiskManager
from src.risk.smart_position_manager import SmartPositionManager
from src.monitoring.telegram_bot import TelegramMonitor
from config.settings import settings


class AITradingBot:
    """
    Main AI Trading Bot class.
    Integrates all components for fully automated trading.
    """

    def __init__(self):
        self.broker: Optional[MT5Broker] = None
        self.data_collector: Optional[DataCollector] = None
        self.ensemble: Optional[EnsembleDecision] = None
        self.risk_manager: Optional[RiskManager] = None
        self.telegram: Optional[TelegramMonitor] = None
        self.smart_mgr: Optional[SmartPositionManager] = None

        self._running = False
        self._paused = False
        self._trading_mode = 'normal'  # 'normal' or 'scalping'
        self._scalper: Optional[ScalpingStrategy] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'start_time': None,
        }

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("🚀 Initializing AI Trading Bot...")

            # 1. Initialize Broker (Exness MT5)
            self.broker = MT5Broker(
                account=settings.exness_account,
                password=settings.exness_password,
                server=settings.exness_server,
            )

            if not self.broker.connect():
                logger.error("❌ Failed to connect to Exness MT5")
                return False

            account_info = self.broker.get_account_info()
            logger.info(f"✅ Connected - Balance: ${account_info.get('balance', 0):.2f}, "
                         f"Leverage: 1:{account_info.get('leverage', 0)}")

            # 2. Initialize Risk Manager
            self.risk_manager = RiskManager(
                initial_balance=account_info.get('balance', 10000)
            )

            # 3. Initialize AI Models
            self.ensemble = EnsembleDecision()
            logger.info(self.ensemble.get_model_performance_summary())

            # 4. Initialize Data Collector
            self.data_collector = DataCollector(self.broker)

            # 5. Initialize Telegram Monitor
            self.telegram = TelegramMonitor()
            self._setup_telegram_callbacks()

            # 6. Initialize Smart Position Manager (after telegram so we have it)
            self.smart_mgr = SmartPositionManager(
                broker=self.broker,
                data_collector=self.data_collector,
                ensemble=self.ensemble,
                telegram=self.telegram,
            )
            logger.info("🎯 Smart Position Manager initialized")

            logger.info("✅ AI Trading Bot initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    def _setup_telegram_callbacks(self):
        """Setup Telegram bot callbacks."""
        if not self.telegram:
            return

        async def get_balance():
            return self.broker.get_account_info() if self.broker else {}

        async def get_positions():
            positions = self.broker.get_open_positions() if self.broker else []
            return [{
                'symbol': p.symbol,
                'order_type': p.order_type,
                'volume': p.volume,
                'open_price': p.open_price,
                'profit': p.profit,
            } for p in positions]

        async def get_risk():
            if self.broker and self.risk_manager:
                info = self.broker.get_account_info()
                return self.risk_manager.get_risk_report(info)
            return "Risk manager not initialized"

        async def pause():
            self._paused = True
            logger.info("Trading paused via Telegram")

        async def resume():
            self._paused = False
            logger.info("Trading resumed via Telegram")

        async def train():
            return await self._train_models_async()

        async def set_mode(mode: str):
            self._trading_mode = mode
            if mode == 'scalping':
                self._scalper = ScalpingStrategy()
                logger.info("⚡ Switching to SCALPING mode")
            else:
                self._scalper = None
                logger.info("📊 Switching to NORMAL mode")

        async def set_sl(pips: int):
            logger.info(f"📏 SL set to {pips} pips")
            # Will be applied to next scalping trades

        async def set_max_trades(n: int):
            settings.max_open_trades = n
            logger.info(f"📋 Max trades set to {n}")

        self.telegram.set_balance_callback(get_balance)
        self.telegram.set_positions_callback(get_positions)
        self.telegram.set_risk_callback(get_risk)
        self.telegram.set_pause_callback(pause)
        self.telegram.set_resume_callback(resume)
        self.telegram.set_train_callback(train)
        self.telegram.set_mode_callback(set_mode)
        self.telegram.set_setsl_callback(set_sl)
        self.telegram.set_setmax_callback(set_max_trades)

        # Smart manager status callback
        async def get_smart_status():
            if self.smart_mgr:
                return self.smart_mgr.get_status()
            return "Smart Position Manager not initialized"

        self.telegram.set_smart_callback(get_smart_status)

    async def _train_models_async(self) -> str:
        """Train AI models asynchronously."""
        results = []

        # Get historical data for training
        for symbol in settings.symbols_list[:1]:  # Train on first symbol only
            for tf in ['H1']:  # Use H1 for training
                df = self.data_collector.get_historical_data(symbol, tf, days=60)
                if df.empty:
                    continue

                df_ta = TechnicalAnalysis.compute_all(df)

                # Train XGBoost
                if settings.use_xgboost and self.ensemble:
                    try:
                        self.ensemble.xgboost.train(df_ta)
                        results.append(f"✅ XGBoost trained on {symbol} {tf}")
                    except Exception as e:
                        results.append(f"❌ XGBoost failed: {e}")

                # Train LSTM
                if settings.use_lstm and self.ensemble:
                    try:
                        self.ensemble.lstm.train(df_ta)
                        results.append(f"✅ LSTM trained on {symbol} {tf}")
                    except Exception as e:
                        results.append(f"❌ LSTM failed: {e}")

                # Train RL Agent
                if settings.use_rl_agent and self.ensemble:
                    try:
                        self.ensemble.rl_agent.train(df_ta)
                        results.append(f"✅ RL Agent trained on {symbol} {tf}")
                    except Exception as e:
                        results.append(f"❌ RL Agent failed: {e}")

        return "\n".join(results) if results else "No models trained"

    def start(self):
        """Start the trading bot."""
        if not self.initialize():
            return

        self._running = True
        self._performance_stats['start_time'] = datetime.now()

        # Start data collector
        self.data_collector.start()

        # Register data callback
        self.data_collector.on_new_data(self._on_new_data)

        # Start Telegram bot
        if self.telegram:
            asyncio.create_task(self.telegram.start())

        # Start main trading loop in background
        self._loop_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._loop_thread.start()

        logger.info("✅ AI Trading Bot is now RUNNING")

    def stop(self):
        """Stop the trading bot."""
        self._running = False
        self._paused = False

        if self.data_collector:
            self.data_collector.stop()

        if self.telegram:
            asyncio.create_task(self.telegram.stop())

        if self.broker:
            self.broker.disconnect()

        logger.info("AI Trading Bot stopped")

    def _trading_loop(self):
        """Main trading loop - runs on a background thread."""
        logger.info("🔄 Trading loop started")

        # Wait for initial data collection
        time.sleep(10)

        while self._running:
            try:
                if self._paused:
                    time.sleep(1)
                    continue

                # Process each symbol based on mode
                for symbol in settings.symbols_list:
                    if self._trading_mode == 'scalping':
                        self._process_symbol_scalping(symbol)
                    else:
                        self._process_symbol(symbol)

                # Check and manage open positions intelligently
                if self.smart_mgr:
                    self.smart_mgr.check_positions(self._trading_mode)

                # Sleep before next iteration
                if self._trading_mode == 'scalping':
                    time.sleep(10)  # Faster checks for scalping
                else:
                    time.sleep(60)  # Normal: check every 60 seconds

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(5)

    def _process_symbol_scalping(self, symbol: str):
        """Scalping: fast trades on M5 with tight SL/TP."""
        try:
            # Use M5 for scalping
            df = self.data_collector.get_latest(symbol, 'M5', count=50)

            if df.empty or len(df) < 20:
                logger.debug(f"[SCALP] {symbol}: No data ({len(df)} rows)")
                return

            # Use scalping strategy
            if not self._scalper:
                self._scalper = ScalpingStrategy()

            decision = self._scalper.analyze(df)

            if decision['signal'] == 'neutral':
                logger.debug(f"[SCALP] {symbol}: neutral")
                return

            logger.info(f"[SCALP] {symbol}: {decision['signal']} (conf: {decision.get('confidence', 0):.2f})")

            # Check positions
            account_info = self.broker.get_account_info()
            open_positions = self.broker.get_open_positions(symbol)

            can_trade, reason = self.risk_manager.can_open_trade(account_info, open_positions)
            if not can_trade:
                return

            symbol_info = self.broker.get_symbol_info(symbol)
            current_price = float(df.iloc[-1]['Close'])
            atr = float(df.iloc[-1].get('ATR_5', current_price * 0.001))

            # Tight scalping SL/TP
            sl_pips = atr * 0.5  # Tight SL
            tp_pips = atr * 1.0  # Tight TP

            order_type = 'buy' if decision['signal'] == 'buy' else 'sell'
            position_size = symbol_info.get('min_volume', 0.01)

            if order_type == 'buy':
                sl = current_price - sl_pips
                tp = current_price + tp_pips
            else:
                sl = current_price + sl_pips
                tp = current_price - tp_pips

            order = Order(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                sl=sl,
                tp=tp,
                comment=f"SCALP_{order_type.upper()}_{decision['confidence']:.0%}",
            )

            ticket = self.broker.place_order(order)

            if ticket:
                self.risk_manager.daily_trades += 1
                logger.info(f"[SCALP] ✅ {order_type.upper()} {symbol} @ {current_price:.2f} | SL: {sl:.2f} TP: {tp:.2f}")
                if self.telegram:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.telegram.send_message(
                                f"⚡ SCALP {order_type.upper()} {symbol}\n"
                                f"Entry: ${current_price:.2f}\n"
                                f"SL: ${sl:.2f} | TP: ${tp:.2f}"
                            )
                        )
                    except RuntimeError:
                        pass
            else:
                logger.error(f"[SCALP] ❌ Order failed on {symbol}")

        except Exception as e:
            logger.error(f"[SCALP] Error: {e}")

    def _process_symbol(self, symbol: str):
        """Process a single symbol: analyze and trade."""
        try:
            # 1. Get latest data for the primary timeframe (H1)
            df = self.data_collector.get_latest(symbol, 'H1', count=100)

            if df.empty or len(df) < 50:
                logger.debug(f"{symbol}: No data or insufficient data ({len(df)} rows)")
                return

            # Debug: check for NaN indicators
            df_ta = self.ensemble.ta.compute_all(df)
            nan_count = df_ta[['Close', 'SMA_20', 'SMA_50', 'MACD', 'RSI_14', 'ATR_14']].isna().sum().sum()
            if nan_count > 0:
                logger.debug(f"{symbol}: NaN indicators detected ({nan_count})")

            # 2. Run AI analysis
            decision = self.ensemble.analyze(df)
            logger.info(f"{symbol}: {decision['signal']} (confidence: {decision.get('confidence', 0):.2f}, "
                       f"score: {decision.get('net_score', 0):.3f})")

            if decision['signal'] == 'neutral':
                return

            # 3. Get account info and open positions
            account_info = self.broker.get_account_info()
            open_positions = self.broker.get_open_positions(symbol)

            # 4. Check risk management
            can_trade, risk_reason = self.risk_manager.can_open_trade(
                account_info, open_positions
            )

            if not can_trade:
                logger.info(f"{symbol}: Risk check failed - {risk_reason}")
                return

            # 5. Get symbol info for position sizing
            symbol_info = self.broker.get_symbol_info(symbol)

            # 6. Calculate position size and SL/TP
            order_type = 'buy' if decision['signal'] == 'buy' else 'sell'

            # Check if we have an opposite position - close it first
            for pos in open_positions:
                if pos.symbol == symbol and pos.order_type != order_type:
                    logger.info(f"Closing opposite position {pos.ticket} on {symbol}")
                    self.broker.close_position(pos.ticket)
                    self.risk_manager.update_pnl(pos.profit)
                    self._update_stats(pos.profit, is_win=pos.profit > 0)

                    if self.telegram:
                        import asyncio
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(
                                self.telegram.send_message(
                                    f"🔒 Closed opposite position on {symbol}: ${pos.profit:.2f}"
                                )
                            )
                        except RuntimeError:
                            logger.debug("No running event loop for Telegram message")

            position_size = self.risk_manager.calculate_position_size(
                decision, account_info, symbol_info
            )
            sl, tp = self.risk_manager.calculate_sl_tp(decision, order_type)

            # 7. Place the order
            order = Order(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                sl=sl,
                tp=tp,
                comment=f"AI_{order_type.upper()}_{decision['confidence']:.0%}",
            )

            ticket = self.broker.place_order(order)

            if ticket:
                self.risk_manager.daily_trades += 1
                logger.info(f"✅ Trade opened: {order_type.upper()} {symbol} @ ticket {ticket}")

                if self.telegram:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(
                            self.telegram.send_trade_alert(
                                decision, 'open',
                                f"Ticket: {ticket} | Size: {position_size} | SL: {sl:.2f} | TP: {tp:.2f}"
                            )
                        )
                    except RuntimeError:
                        logger.debug("No running event loop for Telegram alert")
            else:
                logger.error(f"❌ Failed to place order on {symbol}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    def _on_new_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Callback when new market data arrives."""
        # When new data comes in, let smart manager check positions if any are open
        if self.smart_mgr and not self._paused:
            try:
                self.smart_mgr.check_positions(self._trading_mode)
            except Exception as e:
                logger.debug(f"[SMART_MGR] Check on new data error: {e}")

    def _update_stats(self, pnl: float, is_win: bool):
        """Update performance statistics."""
        self._performance_stats['total_trades'] += 1
        self._performance_stats['total_pnl'] += pnl
        if is_win:
            self._performance_stats['winning_trades'] += 1
        else:
            self._performance_stats['losing_trades'] += 1

    def get_status(self) -> str:
        """Get bot status as a formatted string."""
        stats = self._performance_stats
        total = stats['total_trades']
        win_rate = (stats['winning_trades'] / total * 100) if total > 0 else 0

        lines = [
            "🤖 *AI Trading Bot Status*\n",
            f"Status: {'🟢 Running' if self._running else '🔴 Stopped'}",
            f"Paused: {'Yes' if self._paused else 'No'}",
            f"Uptime: {datetime.now() - stats['start_time'] if stats['start_time'] else 'N/A'}\n",
            f"📊 *Performance:*",
            f"Total Trades: {total}",
            f"Win Rate: {win_rate:.1f}%",
            f"Total P&L: ${stats['total_pnl']:.2f}\n",
        ]

        if self.broker:
            info = self.broker.get_account_info()
            lines.extend([
                f"💰 *Account:*",
                f"Balance: ${info.get('balance', 0):.2f}",
                f"Equity: ${info.get('equity', 0):.2f}",
            ])

        return "\n".join(lines)