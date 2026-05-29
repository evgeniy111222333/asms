"""RL trading environment and execution optimizer."""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from acms.ml.config import MLConfig


class TradingEnvironment:
    """RL Trading Environment for execution optimization.

    Implements a gymnasium-compatible environment where:
    - Observation: market data features + inventory + PnL + indicators
    - Action: buy/sell/hold with continuous size parameter
    - Reward: risk-adjusted return (Sharpe-like)
    """

    def __init__(self, candles_data: dict, initial_inventory: float = 100.0,
                 max_steps: int = 100, transaction_cost_bps: float = 10.0,
                 risk_free_rate: float = 0.0):
        self.candles_data = candles_data
        self.initial_inventory = initial_inventory
        self.max_steps = max_steps
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate

        self.current_step = 0
        self.inventory = initial_inventory
        self.avg_price = 0.0
        self.total_cost = 0.0
        self.done = False

        self.closes = candles_data.get("close", np.array([]))
        self.volumes = candles_data.get("volume", np.array([]))
        self.highs = candles_data.get("high", np.array([]))
        self.lows = candles_data.get("low", np.array([]))

    def _create_gym_env(self):
        """Create the gymnasium environment with full observation/action spaces."""
        try:
            import gymnasium as gym
            from gymnasium import spaces

            closes = self.closes
            volumes = self.volumes
            highs = self.highs
            lows = self.lows
            initial_inventory = self.initial_inventory
            max_steps = self.max_steps
            transaction_cost_bps = self.transaction_cost_bps
            risk_free_rate = self.risk_free_rate

            class ACMSTradingEnv(gym.Env):
                """ACMS Trading Environment following gymnasium interface.

                Observation space (7 dimensions):
                - Normalized price (relative to first close)
                - Normalized volume
                - Inventory ratio (remaining/initial)
                - Step progress ratio
                - Running PnL (normalized)
                - ATR-like volatility proxy
                - Price position in high-low range

                Action space (3 dimensions, continuous):
                - action[0]: direction (-1=buy, 0=hold, +1=sell)
                - action[1]: order size fraction [0,1]
                - action[2]: limit price offset [-0.5%, +0.5%]
                """

                metadata = {"render_modes": ["human"]}

                def __init__(self):
                    super().__init__()
                    self.closes = closes
                    self.volumes = volumes
                    self.highs = highs
                    self.lows = lows
                    self.initial_inventory = initial_inventory
                    self.max_steps = max_steps
                    self.transaction_cost_bps = transaction_cost_bps
                    self.risk_free_rate = risk_free_rate
                    self.avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else 1.0

                    # Action: [direction, size_fraction, price_offset]
                    self.action_space = spaces.Box(
                        low=np.array([-1.0, 0.0, -1.0]),
                        high=np.array([1.0, 1.0, 1.0]),
                        dtype=np.float32,
                    )

                    # Observation: 7 features
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32,
                    )

                    self.reset()

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    self.current_step = 0
                    self.inventory = self.initial_inventory
                    self.total_cost = 0.0
                    self.avg_price = 0.0
                    self.realized_pnl = 0.0
                    self.pnl_history: List[float] = []
                    self.position = 0.0
                    self.position_avg_price = 0.0
                    return self._get_obs(), {}

                def step(self, action):
                    direction = action[0]
                    size_fraction = action[1]
                    price_offset = action[2] * 0.005

                    idx = min(self.current_step, len(self.closes) - 1)
                    price = self.closes[idx]
                    fill_price = price * (1 + price_offset)

                    # Determine order quantity
                    if direction > 0.3:  # Buy
                        quantity = size_fraction * self.inventory * 0.1
                        cost = fill_price * quantity * (1 + self.transaction_cost_bps / 10000)
                        self.total_cost += cost
                        self.inventory -= cost / fill_price
                        # Update position tracking
                        old_pos_value = self.position * self.position_avg_price
                        self.position += quantity
                        if self.position > 0:
                            self.position_avg_price = (old_pos_value + fill_price * quantity) / self.position
                    elif direction < -0.3:  # Sell
                        quantity = min(size_fraction * abs(self.position) * 0.1, abs(self.position))
                        if quantity > 0 and self.position > 0:
                            proceeds = fill_price * quantity * (1 - self.transaction_cost_bps / 10000)
                            self.realized_pnl += (fill_price - self.position_avg_price) * quantity
                            self.position -= quantity
                            self.inventory += proceeds / fill_price

                    self.current_step += 1

                    # Compute unrealized PnL
                    current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
                    unrealized_pnl = self.position * (current_price - self.position_avg_price)
                    total_pnl = self.realized_pnl + unrealized_pnl
                    self.pnl_history.append(total_pnl)

                    # Reward: risk-adjusted return
                    if len(self.pnl_history) > 1:
                        returns = np.diff(self.pnl_history[-min(20, len(self.pnl_history)):])
                        reward = np.mean(returns) - 0.5 * np.var(returns)
                    else:
                        reward = 0.0

                    # Penalty for excessive inventory
                    if self.inventory < self.initial_inventory * 0.1:
                        reward -= 1.0

                    terminated = self.current_step >= self.max_steps or self.inventory <= 0.01
                    truncated = False

                    return self._get_obs(), float(reward), terminated, truncated, {}

                def _get_obs(self):
                    idx = min(self.current_step, len(self.closes) - 1)
                    price_norm = self.closes[idx] / self.closes[0] - 1
                    volume_norm = self.volumes[idx] / (self.avg_volume + 1e-10)
                    inventory_ratio = self.inventory / self.initial_inventory
                    step_ratio = self.current_step / self.max_steps

                    # Running PnL normalized
                    pnl_norm = 0.0
                    if self.initial_inventory > 0:
                        current_price = self.closes[idx]
                        unrealized = self.position * (current_price - self.position_avg_price)
                        pnl_norm = (self.realized_pnl + unrealized) / self.initial_inventory

                    # ATR-like volatility proxy
                    lookback = min(14, idx)
                    if lookback > 1:
                        atr = float(np.mean(self.highs[idx-lookback:idx] - self.lows[idx-lookback:idx]))
                        atr_norm = atr / (self.closes[0] + 1e-10)
                    else:
                        atr_norm = 0.0

                    # Price position in range
                    range_val = self.highs[idx] - self.lows[idx]
                    if range_val > 0:
                        range_pos = (self.closes[idx] - self.lows[idx]) / range_val
                    else:
                        range_pos = 0.5

                    return np.array([
                        price_norm, volume_norm, inventory_ratio, step_ratio,
                        pnl_norm, atr_norm, range_pos,
                    ], dtype=np.float32)

            return ACMSTradingEnv()
        except ImportError:
            raise ImportError("gymnasium is required for RL environment")

    def make_env(self):
        """Create and return the gymnasium environment."""
        return self._create_gym_env()


# Also include RLExecutionOptimizer
class RLExecutionOptimizer:
    """Reinforcement learning for optimal execution using Stable-Baselines3.

    Provides a complete pipeline for training RL agents on the
    trading environment, with support for PPO, A2C, and DQN algorithms.
    """

    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm
        self.model = None
        self.env = None

    def create_environment(self, candles_data: dict, initial_inventory: float = 100.0,
                           max_steps: int = 100, transaction_cost_bps: float = 10.0):
        """Create a trading environment for RL.

        Args:
            candles_data: Dict with 'close', 'volume', 'high', 'low' arrays.
            initial_inventory: Starting inventory size.
            max_steps: Maximum number of steps per episode.
            transaction_cost_bps: Transaction cost in basis points.

        Returns:
            gymnasium.Env instance.
        """
        env_factory = TradingEnvironment(
            candles_data, initial_inventory, max_steps, transaction_cost_bps
        )
        self.env = env_factory.make_env()
        return self.env

    def train(self, total_timesteps: int = 100000, **kwargs) -> Dict:
        """Train RL agent.

        Args:
            total_timesteps: Total training timesteps.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            Dict with training info.
        """
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable-baselines3 is required")

        if self.env is None:
            raise RuntimeError("Environment not created. Call create_environment() first.")

        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        algo_class = algo_map.get(self.algorithm, PPO)

        learning_rate = kwargs.get("learning_rate", 3e-4)
        n_steps = kwargs.get("n_steps", 2048)
        batch_size = kwargs.get("batch_size", 64)
        n_epochs = kwargs.get("n_epochs", 10)

        self.model = algo_class(
            "MlpPolicy", self.env,
            learning_rate=learning_rate,
            n_steps=n_steps if self.algorithm == "PPO" else None,
            batch_size=batch_size,
            n_epochs=n_epochs if self.algorithm == "PPO" else None,
            verbose=0,
        )
        self.model.learn(total_timesteps=total_timesteps)
        return {"algorithm": self.algorithm, "total_timesteps": total_timesteps}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict optimal action for given observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Selected action array.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        """Save trained RL model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load trained RL model from disk."""
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable-baselines3 is required")
        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        algo_class = algo_map.get(self.algorithm, PPO)
        self.model = algo_class.load(path)


# ============================================================================
# Model Persistence Utilities
# ============================================================================



__all__ = ["TradingEnvironment", "RLExecutionOptimizer"]
