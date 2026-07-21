"""
Reinforcement Learning Agent using Stable-Baselines3 (PPO).
Trained on the TradingEnvironment to make optimal trading decisions.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, Optional
from loguru import logger

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    PPO = None
    DummyVecEnv = None
    EvalCallback = None

from src.rl.trading_env import TradingEnvironment
from config.settings import settings


class RLAgent:
    """
    Reinforcement Learning trading agent.
    Uses PPO (Proximal Policy Optimization) algorithm.
    """

    def __init__(self):
        self.model: Optional[PPO] = None
        self.is_trained = False
        self.model_path = os.path.join(settings.ml_model_path, 'ppo_trading_agent')

    def _make_env(self, df: pd.DataFrame) -> DummyVecEnv:
        """Create a vectorized trading environment."""
        def _init():
            return TradingEnvironment(
                df=df,
                initial_balance=10000.0,
                max_position_size=settings.max_position_size,
                max_trades=settings.max_open_trades,
            )
        return DummyVecEnv([_init])

    def train(self, df: pd.DataFrame, total_timesteps: int = 50000):
        """
        Train the RL agent on historical data.
        """
        if not _SB3_AVAILABLE:
            logger.warning("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
            return
        env = self._make_env(df)

        # Create eval env
        split_idx = int(len(df) * 0.8)
        eval_df = df.iloc[split_idx:].reset_index(drop=True)
        eval_env = self._make_env(eval_df)

        # Initialize model
        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            tensorboard_log=os.path.join(settings.ml_model_path, 'tensorboard'),
        )

        # Eval callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(settings.ml_model_path, 'best_model'),
            log_path=os.path.join(settings.ml_model_path, 'logs'),
            eval_freq=5000,
            deterministic=True,
            render=False,
        )

        # Train
        logger.info(f"🚀 Training RL Agent for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        self.is_trained = True
        logger.info("✅ RL Agent training complete")

        # Test
        self._test(eval_df)

        self.save()

    def _test(self, df: pd.DataFrame):
        """Test the agent on a dataframe and log results."""
        env = self._make_env(df)
        obs = env.reset()
        total_reward = 0
        steps = 0

        while steps < len(df) - 50:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            if done:
                break

        logger.info(f"📊 RL Test - Total Reward: {total_reward:.2f}, "
                     f"Steps: {steps}")

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict the best action for the current state.
        Returns: 0=Hold, 1=Buy, 2=Sell
        """
        if not _SB3_AVAILABLE:
            return 0
        if not self.is_trained or self.model is None:
            return 0  # Hold if not trained

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def predict_with_confidence(self, observation: np.ndarray) -> Dict:
        """
        Predict action with confidence score.
        """
        if not _SB3_AVAILABLE:
            return {'action': 0, 'action_name': 'hold', 'confidence': 0.0}
        if not self.is_trained or self.model is None:
            return {'action': 0, 'action_name': 'hold', 'confidence': 0.0}

        action, states = self.model.predict(observation, deterministic=False)

        # Get action probabilities
        try:
            probs = self.model.policy.get_distribution(observation).distribution.probs
            confidence = float(probs[0][action].max().cpu().numpy())
        except:
            confidence = 0.5

        action_names = {0: 'hold', 1: 'buy', 2: 'sell'}
        return {
            'action': int(action),
            'action_name': action_names[int(action)],
            'confidence': float(confidence),
        }

    def save(self):
        """Save model to disk."""
        if self.model is None:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"RL Agent saved to {self.model_path}")

    def load(self) -> bool:
        """Load model from disk."""
        model_file = f"{self.model_path}.zip"
        if not os.path.exists(model_file):
            logger.warning(f"RL Agent model not found at {model_file}")
            return False
        try:
            # Create a temporary env to load the model
            dummy_df = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
            env = self._make_env(dummy_df)
            self.model = PPO.load(self.model_path, env=env)
            self.is_trained = True
            logger.info(f"✅ RL Agent loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load RL Agent: {e}")
            return False