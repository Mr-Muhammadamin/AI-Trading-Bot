"""
Reinforcement Learning Trading Environment.
OpenAI Gym-style environment for training RL agents.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from gymnasium import Env, spaces
from loguru import logger


class TradingEnvironment(Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    State: Technical indicators + position info
    Actions: Buy, Sell, Hold
    Reward: Profit-based with risk penalty
    """

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                 max_position_size: float = 0.1, max_trades: int = 3):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.max_trades = max_trades

        # State: indicators + position info
        self.indicator_cols = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI_14', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI_20',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width', 'BB_Percent',
            'ATR_14', 'ATR_Percent',
            'OBV', 'Volume_Ratio',
            'ADX', 'ADX_Pos', 'ADX_Neg',
            'MFI_14', 'ROC_10',
            'Trend_MA', 'Trend_MACD',
        ]
        # Filter to available columns
        self.available_indicators = [c for c in self.indicator_cols if c in df.columns]
        self.n_indicators = len(self.available_indicators)

        # Position info: [has_position, position_type(0/1), entry_price, position_size, unrealized_pnl]
        self.n_position_features = 5

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_indicators + self.n_position_features,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 50  # Start after enough data for indicators
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position = None  # {'type': 'buy'/'sell', 'entry_price': float, 'size': float}
        self.trades_history = []
        self.total_trades = 0
        self.total_profit = 0.0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        action: 0=Hold, 1=Buy, 2=Sell
        """
        done = False
        truncated = False
        reward = 0.0

        # Get current and next price
        current_price = self.df.iloc[self.current_step]['Close']
        next_price = self.df.iloc[self.current_step + 1]['Close'] if self.current_step + 1 < len(self.df) else current_price

        # Execute action
        if action == 1:  # Buy
            if self.position is None and self.total_trades < self.max_trades:
                self.position = {
                    'type': 'buy',
                    'entry_price': current_price,
                    'size': self.max_position_size,
                }
                self.total_trades += 1
                reward = -0.1  # Small cost for opening position

        elif action == 2:  # Sell
            if self.position is None and self.total_trades < self.max_trades:
                self.position = {
                    'type': 'sell',
                    'entry_price': current_price,
                    'size': self.max_position_size,
                }
                self.total_trades += 1
                reward = -0.1

        # Update position P&L
        if self.position is not None:
            if self.position['type'] == 'buy':
                pnl = (next_price - self.position['entry_price']) / self.position['entry_price'] * 100
            else:  # sell
                pnl = (self.position['entry_price'] - next_price) / self.position['entry_price'] * 100

            # Reward based on P&L
            reward += pnl * 10  # Scale reward

            # Close position if action is opposite
            if (action == 2 and self.position['type'] == 'buy') or \
               (action == 1 and self.position['type'] == 'sell'):
                self.total_profit += pnl
                self.trades_history.append({
                    'type': self.position['type'],
                    'entry': self.position['entry_price'],
                    'exit': current_price,
                    'pnl': pnl,
                })
                self.position = None

                # Bonus for profitable trade
                if pnl > 0:
                    reward += pnl * 20
                else:
                    reward += pnl * 10  # Penalty for losing trade

        # Update equity
        self.equity = self.balance + (self.total_profit / 100 * self.initial_balance)

        # Move to next step
        self.current_step += 1

        # Check if done
        if self.current_step >= len(self.df) - 1:
            done = True
            # Close any open position
            if self.position is not None:
                if self.position['type'] == 'buy':
                    final_pnl = (current_price - self.position['entry_price']) / self.position['entry_price'] * 100
                else:
                    final_pnl = (self.position['entry_price'] - current_price) / self.position['entry_price'] * 100
                self.total_profit += final_pnl
                self.position = None

        # Penalty for large drawdown
        if self.equity < self.initial_balance * 0.8:
            reward -= 50
            done = True

        return self._get_observation(), reward, done, truncated, {
            'equity': self.equity,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
            'position': self.position,
        }

    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state)."""
        # Technical indicators
        indicators = self.df.iloc[self.current_step][self.available_indicators].values
        indicators = np.nan_to_num(indicators, nan=0.0, posinf=0.0, neginf=0.0)

        # Position info
        if self.position is not None:
            pos_info = np.array([
                1.0,
                1.0 if self.position['type'] == 'buy' else 0.0,
                self.position['entry_price'] / self.df.iloc[self.current_step]['Close'],
                self.position['size'],
                (self.df.iloc[self.current_step]['Close'] - self.position['entry_price']) / self.position['entry_price'],
            ], dtype=np.float32)
        else:
            pos_info = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return np.concatenate([indicators, pos_info]).astype(np.float32)

    def render(self, mode: str = 'human'):
        """Render the environment."""
        print(f"Step: {self.current_step}, Equity: ${self.equity:.2f}, "
              f"Trades: {self.total_trades}, Profit: {self.total_profit:.2f}%")