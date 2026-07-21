"""
AI Trading Bot - Main Entry Point
Real-time AI-powered trading bot for Exness MT5 with Telegram monitoring.
"""
import asyncio
import sys
import signal
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.trading_bot import AITradingBot
from config.settings import settings


# Configure logging
logger.remove()
logger.add(
    settings.log_file,
    rotation="10 MB",
    retention="30 days",
    level=settings.log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
)
logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{message}</cyan>",
)


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("\n🛑 Shutting down AI Trading Bot...")
    if bot:
        bot.stop()
    sys.exit(0)


# Global bot instance
bot: AITradingBot = None


async def main():
    """Main async entry point."""
    global bot

    logger.info("=" * 50)
    logger.info("🤖 AI TRADING BOT v1.0")
    logger.info("=" * 50)
    logger.info(f"Symbols: {settings.symbols}")
    logger.info(f"Timeframes: {settings.timeframes}")
    logger.info(f"Max Trades: {settings.max_open_trades}")
    logger.info(f"Risk/Trade: {settings.risk_per_trade}%")
    logger.info(f"AI Models: XGBoost={settings.use_xgboost}, "
                f"LSTM={settings.use_lstm}, RL={settings.use_rl_agent}")
    logger.info("=" * 50)

    # Create and start bot
    bot = AITradingBot()
    bot.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        if bot:
            bot.stop()


if __name__ == "__main__":
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)