"""
Global configuration settings loaded from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional

# Load .env file
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(env_path)


class Settings(BaseSettings):
    # === BROKER ===
    # Exness MT5
    exness_account: int = 0
    exness_password: str = ''
    exness_server: str = 'Exness-MT5Trial'

    # Binance (CCXT)
    binance_api_key: str = ''
    binance_secret_key: str = ''
    binance_testnet: bool = True

    # === AI / ML ===
    ml_model_path: str = './data/models'
    use_xgboost: bool = True
    use_lstm: bool = True
    use_rl_agent: bool = True
    use_ensemble: bool = True

    # === TRADING ===
    symbols: str = 'BTCUSD,XAUUSD,ETHUSD'
    timeframes: str = 'M5,M15,H1,H4,D1'
    max_position_size: float = 0.1
    max_open_trades: int = 3
    risk_per_trade: float = 2.0
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 4.0

    # === TELEGRAM ===
    telegram_bot_token: str = ''
    telegram_chat_id: str = ''

    # === DATABASE ===
    database_url: str = 'sqlite:///./data/trading.db'

    # === LOGGING ===
    log_level: str = 'INFO'
    log_file: str = './logs/trading_bot.log'

    @field_validator('symbols', mode='before')
    @classmethod
    def parse_symbols(cls, v):
        if isinstance(v, str):
            return v
        return ','.join(v)

    @field_validator('timeframes', mode='before')
    @classmethod
    def parse_timeframes(cls, v):
        if isinstance(v, str):
            return v
        return ','.join(v)

    @property
    def symbols_list(self) -> List[str]:
        return [s.strip() for s in self.symbols.split(',') if s.strip()]

    @property
    def timeframes_list(self) -> List[str]:
        return [t.strip() for t in self.timeframes.split(',') if t.strip()]

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False


settings = Settings()