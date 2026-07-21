"""
Telegram Bot for real-time monitoring and control of the AI Trading Bot.
Sends trade alerts, performance reports, and accepts commands.
"""
import asyncio
from typing import Dict, Optional, Callable
from datetime import datetime
from loguru import logger

from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config.settings import settings


class TelegramMonitor:
    """
    Telegram bot for monitoring and controlling the trading bot.
    Features:
    - Real-time trade alerts
    - Performance reports
    - Remote commands (status, pause, resume)
    - Error notifications
    """

    def __init__(self):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.application: Optional[Application] = None
        self._command_handlers: Dict[str, Callable] = {}
        self._running = False

    async def start(self):
        """Start the Telegram bot."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not configured. Skipping...")
            return

        try:
            self.application = Application.builder().token(self.token).build()

            # Register commands
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            self.application.add_handler(CommandHandler("positions", self._cmd_positions))
            self.application.add_handler(CommandHandler("balance", self._cmd_balance))
            self.application.add_handler(CommandHandler("risk", self._cmd_risk))
            self.application.add_handler(CommandHandler("pause", self._cmd_pause))
            self.application.add_handler(CommandHandler("resume", self._cmd_resume))
            self.application.add_handler(CommandHandler("train", self._cmd_train))
            self.application.add_handler(CommandHandler("mode", self._cmd_mode))
            self.application.add_handler(CommandHandler("setsl", self._cmd_setsl))
            self.application.add_handler(CommandHandler("setmax", self._cmd_setmax))
            self.application.add_handler(CommandHandler("smart", self._cmd_smart))
            
            # Log all updates
            async def log_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
                if update.message:
                    logger.info(f"Telegram update received: {update.message.text} from {update.message.chat_id}")
                return None
            
            self.application.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("Bot started!")))

            # Start polling
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self._running = True

            # Send startup message
            await self.send_message("🤖 *AI Trading Bot Started*\n"
                                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    f"Symbols: {', '.join(settings.symbols)}\n"
                                    f"Type /help for commands")

            logger.info("✅ Telegram bot started")
        except Exception as e:
            logger.error(f"Failed to start Telegram bot: {e}")

    async def stop(self):
        """Stop the Telegram bot."""
        if self.application and self._running:
            await self.application.stop()
            await self.application.shutdown()
            self._running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, message: str, parse_mode: str = 'Markdown'):
        """Send a message to the configured chat."""
        if not self.token or not self.chat_id or not self._running:
            return
        try:
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def send_trade_alert(self, decision: Dict, action: str, result: str):
        """
        Send a trade alert.
        action: 'open' or 'close'
        result: details about the trade
        """
        emoji = "🟢" if action == 'open' else "🔴"
        signal_emoji = "📈" if decision.get('signal') == 'buy' else "📉"

        message = (
            f"{emoji} *Trade {action.upper()}*\n"
            f"{signal_emoji} Signal: {decision.get('signal', 'N/A').upper()}\n"
            f"Confidence: {decision.get('confidence', 0):.1%}\n"
            f"Price: ${decision.get('current_price', 0):.2f}\n"
            f"Reason: {decision.get('reasons', ['N/A'])[0]}\n"
            f"Result: {result}\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        await self.send_message(message)

    async def send_error(self, error_msg: str):
        """Send an error notification."""
        await self.send_message(f"❌ *Error*\n{error_msg}")

    async def send_performance_report(self, stats: Dict):
        """Send a performance report."""
        message = (
            "📊 *Performance Report*\n"
            f"Total Trades: {stats.get('total_trades', 0)}\n"
            f"Win Rate: {stats.get('win_rate', 0):.1%}\n"
            f"Total P&L: ${stats.get('total_pnl', 0):.2f}\n"
            f"Balance: ${stats.get('balance', 0):.2f}\n"
            f"Equity: ${stats.get('equity', 0):.2f}\n"
            f"Open Positions: {stats.get('open_positions', 0)}"
        )
        await self.send_message(message)

    def register_command(self, command: str, handler: Callable):
        """Register a custom command handler."""
        self._command_handlers[command] = handler

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send bot status."""
        status = "🟢 *Running*" if self._running else "🔴 *Stopped*"
        
        # Get live data if callbacks available
        balance_text = ""
        positions_text = ""
        if self._balance_callback:
            try:
                info = await self._balance_callback()
                balance_text = f"Balance: ${info.get('balance', 0):.2f}\nEquity: ${info.get('equity', 0):.2f}\n"
            except Exception as e:
                balance_text = f"Balance: Error ({e})\n"
        
        if self._positions_callback:
            try:
                positions = await self._positions_callback()
                positions_text = f"Open Positions: {len(positions)}\n"
            except Exception:
                positions_text = "Open Positions: Error\n"
        
        await update.message.reply_text(
            f"🤖 *AI Trading Bot Status*\n\n"
            f"Status: {status}\n"
            f"Symbols: {', '.join(settings.symbols)}\n"
            f"Timeframes: {', '.join(settings.timeframes)}\n"
            f"Max Trades: {settings.max_open_trades}\n"
            f"Risk/Trade: {settings.risk_per_trade}%\n\n"
            f"💰 *Account:*\n{balance_text}\n"
            f"📋 {positions_text}",
            parse_mode='Markdown'
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message."""
        help_text = (
            "🤖 *AI Trading Bot Commands*\n\n"
            "/status - Bot status\n"
            "/balance - Account balance\n"
            "/positions - Open positions\n"
            "/risk - Risk report\n"
            "/pause - Pause trading\n"
            "/resume - Resume trading\n"
            "/train - Train AI models\n"
            "/smart - Smart Position Manager status\n"
            "/mode - Trading mode (scalping/normal)\n"
            "/help - Show this help"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send balance info."""
        if self._balance_callback:
            info = await self._balance_callback()
            await update.message.reply_text(
                f"💰 *Account Balance*\n\n"
                f"Balance: ${info.get('balance', 0):.2f}\n"
                f"Equity: ${info.get('equity', 0):.2f}\n"
                f"Margin: ${info.get('margin', 0):.2f}\n"
                f"Free Margin: ${info.get('margin_free', 0):.2f}\n"
                f"Leverage: 1:{info.get('leverage', 0)}",
                parse_mode='Markdown'
            )

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show open positions."""
        if self._positions_callback:
            positions = await self._positions_callback()
            if not positions:
                await update.message.reply_text("No open positions")
                return
            msg = "📋 *Open Positions*\n\n"
            for p in positions:
                emoji = "🟢" if p.get('order_type') == 'buy' else "🔴"
                msg += (
                    f"{emoji} {p.get('symbol')} {p.get('order_type').upper()}\n"
                    f"  Volume: {p.get('volume')} | Profit: ${p.get('profit', 0):.2f}\n"
                    f"  Entry: ${p.get('open_price', 0):.2f}\n\n"
                )
            await update.message.reply_text(msg, parse_mode='Markdown')

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show risk report."""
        if self._risk_callback:
            report = await self._risk_callback()
            await update.message.reply_text(report, parse_mode='Markdown')

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading."""
        if self._pause_callback:
            await self._pause_callback()
            await update.message.reply_text("⏸️ Trading paused")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading."""
        if self._resume_callback:
            await self._resume_callback()
            await update.message.reply_text("▶️ Trading resumed")

    async def _cmd_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train AI models."""
        await update.message.reply_text("🔄 Training AI models... This may take a while.")
        if self._train_callback:
            result = await self._train_callback()
            await update.message.reply_text(f"✅ Training complete:\n{result}")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch trading mode: /mode scalping or /mode normal"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /mode scalping or /mode normal\nCurrent mode: check /status")
            return
        mode = args[0].lower()
        callback = getattr(self, '_mode_callback', None)
        if mode in ('scalping', 'scalp', 's'):
            if callback:
                await callback('scalping')
            await update.message.reply_text("⚡ *Scalping Mode Activated*\nFast trades with tight SL/TP on M1/M5", parse_mode='Markdown')
        elif mode in ('normal', 'standard', 'n'):
            if callback:
                await callback('normal')
            await update.message.reply_text("📊 *Normal Mode Activated*\nStandard trades on H1/H4/D1", parse_mode='Markdown')
        else:
            await update.message.reply_text("Unknown mode. Use: /mode scalping or /mode normal")

    async def _cmd_setsl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set SL pip distance: /setsl 10 (for scalping)"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /setsl <pips>\nExample: /setsl 10")
            return
        try:
            pips = int(args[0])
            callback = getattr(self, '_setsl_callback', None)
            if callback:
                await callback(pips)
            await update.message.reply_text(f"✅ SL set to {pips} pips")
        except ValueError:
            await update.message.reply_text("Invalid number. Use: /setsl <pips>")

    async def _cmd_smart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show Smart Position Manager status."""
        if self._smart_callback:
            try:
                status = await self._smart_callback()
                await update.message.reply_text(status, parse_mode='Markdown')
            except Exception as e:
                await update.message.reply_text(f"❌ Smart Manager error: {e}")
        else:
            await update.message.reply_text("Smart Position Manager not initialized")

    async def _cmd_setmax(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set max open trades: /setmax 5"""
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /setmax <number>\nExample: /setmax 5")
            return
        try:
            n = int(args[0])
            if n < 1 or n > 20:
                await update.message.reply_text("Max trades must be between 1 and 20")
                return
            callback = getattr(self, '_setmax_callback', None)
            if callback:
                await callback(n)
            await update.message.reply_text(f"✅ Max trades set to {n}")
        except ValueError:
            await update.message.reply_text("Invalid number. Use: /setmax <number>")

    # Callback setters
    def set_balance_callback(self, callback):
        self._balance_callback = callback

    def set_positions_callback(self, callback):
        self._positions_callback = callback

    def set_risk_callback(self, callback):
        self._risk_callback = callback

    def set_pause_callback(self, callback):
        self._pause_callback = callback

    def set_resume_callback(self, callback):
        self._resume_callback = callback

    def set_train_callback(self, callback):
        self._train_callback = callback

    def set_mode_callback(self, callback):
        self._mode_callback = callback

    def set_setsl_callback(self, callback):
        self._setsl_callback = callback

    def set_setmax_callback(self, callback):
        self._setmax_callback = callback

    def set_smart_callback(self, callback):
        self._smart_callback = callback
