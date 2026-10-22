import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from collections import deque

class GoldTradingEnv(gym.Env):
    """
    Custom Environment for XAUUSD (Gold) Trading using Gymnasium.
    Uses a Dense Reward (Step-by-Step PnL) structure and 2 Action Spaces (0=SELL, 1=BUY).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, window_size: int = 10):
        super(GoldTradingEnv, self).__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance
        self.window_size = window_size

        # Drop non-numeric and datetime columns to get clean features
        # 'time' is usually the index or a column containing datetime string
        self.features_df = self.df.select_dtypes(include=[np.number])
        if 'time' in self.features_df.columns:
            self.features_df = self.features_df.drop(columns=['time'])

        # Action Space: 0 = SELL, 1 = BUY
        self.action_space = spaces.Discrete(2)

        # Observation Space: Box with shape of features (except datetime)
        num_features = self.features_df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * num_features,),
            dtype=np.float32
        )

        # Buffer
        self.window_data = deque(maxlen=window_size)

        # Initialize tracking variables
        self.current_step = window_size
        self.current_position = 0  # 0: flat, 1: BUY, -1: SELL
        self.balance = self.initial_balance

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Handle seeding if provided
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.current_position = 0  # 0: flat (no position), 1: BUY, -1: SELL
        self.balance = self.initial_balance

        self.window_data.clear()
        for i in range(self.current_step - self.window_size, self.current_step):
            self.window_data.append(self.features_df.iloc[i].to_numpy(dtype=np.float32))

        return np.array(self.window_data).flatten(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Get current close price and previous close price
        current_price = float(self.df.iloc[self.current_step]['close'])
        previous_price = float(self.df.iloc[self.current_step - 1]['close'])

        # Calculate price difference
        price_diff = current_price - previous_price

        # Map action to target position: 1 = BUY (1), 0 = SELL (-1)
        target_position = 1 if action == 1 else -1

        # Calculate step reward based on position BEFORE taking action
        step_reward = self.current_position * price_diff

        # Apply Spread/Commission Penalty
        penalty = 0.0
        if self.current_position != target_position:
            penalty = 0.1

        # Final Reward
        reward = step_reward - penalty

        current_row = self.df.iloc[self.current_step]
        bos = float(current_row.get('bos_status', 0))
        mss = float(current_row.get('mss_status', 0))
        sweep = float(current_row.get('liquidity_sweep', 0))
        fvg = float(current_row.get('fvg_status', 0))
        smc_score = bos + mss + sweep + fvg
        if target_position == 1 and smc_score > 0:
            reward += 0.05 * min(smc_score, 3)
        elif target_position == -1 and smc_score < 0:
            reward += 0.05 * min(abs(smc_score), 3)
        elif smc_score != 0:
            reward -= 0.05 * min(abs(smc_score), 3)

        # Update position
        self.current_position = target_position

        # Update balance (for logging / visualization)
        self.balance += reward

        # Ambil fitur pada self.current_step saat ini
        current_features = self.features_df.iloc[self.current_step].to_numpy(dtype=np.float32)
        # Masukkan (append) ke self.window_data
        self.window_data.append(current_features)

        # Advance to the next step
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # 1. Reached the end of the dataset
        if self.current_step >= len(self.features_df):
            terminated = True
        
        # 2. Bankrupt (balance drops to 0 or below)
        elif self.balance <= 0:
            terminated = True

        # Ratakan menjadi obs = np.array(self.window_data).flatten()
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        return np.array(self.window_data).flatten()

    def _get_info(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'position': self.current_position,
            'balance': self.balance
        }

if __name__ == '__main__':
    import os
    csv_path = 'data/xauusd_m15_rl_train.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} tidak ditemukan!")
        print("Generating dummy data for testing...")
        dates = pd.date_range(start="2026-01-01", periods=100, freq="15min")
        df = pd.DataFrame(index=dates)
        df["close"] = 2300.0 + np.random.normal(0, 5, 100).cumsum()
        df["open"] = df["close"].shift(1).fillna(2300.0)
        df["high"] = df[["open", "close"]].max(axis=1) + 2.0
        df["low"] = df[["open", "close"]].min(axis=1) - 2.0
        df["tick_volume"] = 1000
        df["spread"] = 160
        df["real_volume"] = 10000
        df["volume"] = 1000
        df["vwap"] = df["close"].rolling(10).mean().fillna(2300.0)
        df["vwap_dist_pct"] = 0.0
        df["ema_50"] = df["close"].rolling(50).mean().fillna(2300.0)
        df["ema_50_dist_pct"] = 0.0
        df["atr_14"] = 5.0
        df["is_squeeze"] = 0
        df["fvg_status"] = 0
    else:
        df = pd.read_csv(csv_path)
        
    print(f"Memuat dataset dari {csv_path} dengan bentuk: {df.shape}")
    env = GoldTradingEnv(df)
    obs, info = env.reset()
    print("Reset env. Obs shape:", obs.shape)
    print("Initial Info:", info)
    
    print("\n" + "="*80)
    print(f"{'Step':<5} | {'Action':<5} | {'Prev Pos':<8} | {'New Pos':<8} | {'Price Diff':<10} | {'Reward':<8} | {'Balance':<10} | {'Terminated':<10}")
    print("="*80)
    
    action_names = {0: "SELL", 1: "BUY"}
    for step_num in range(1, 11):
        action = env.action_space.sample()
        prev_pos = env.current_position
        
        current_price = float(env.df.iloc[env.current_step]['close'])
        previous_price = float(env.df.iloc[env.current_step - 1]['close'])
        price_diff = current_price - previous_price
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        pos_str = lambda p: "BUY" if p == 1 else ("SELL" if p == -1 else "FLAT")
        
        print(f"{step_num:03d}   | {action_names[action]:<5} | {pos_str(prev_pos):<8} | {pos_str(info['position']):<8} | {price_diff:<10.3f} | {reward:<8.3f} | {info['balance']:<10.2f} | {str(terminated):<10}")
        if terminated or truncated:
            print("="*80)
            print("Episode Selesai!")
            break
    print("="*80)
