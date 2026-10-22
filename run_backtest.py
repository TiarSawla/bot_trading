import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.rl_environment import GoldTradingEnv

def backtest(model_path, csv_path, initial_balance=1000.0):
    print(f"\n--- Backtesting {model_path} dengan balance ${initial_balance} ---")
    
    # Load model & data
    if not os.path.exists(model_path + ".zip"):
        print(f"Model {model_path} tidak ditemukan!")
        return
    model = PPO.load(model_path)
    df = pd.read_csv(csv_path)
    
    # Env
    env = GoldTradingEnv(df, initial_balance=initial_balance)
    obs, info = env.reset()
    
    done = False
    rewards = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        done = terminated or truncated
        
    print(f"Hasil Akhir:")
    print(f"Balance: ${info['balance']:.2f}")
    print(f"Total Reward: {sum(rewards):.2f}")
    print(f"Return: {((info['balance'] - initial_balance) / initial_balance) * 100:.2f}%")

if __name__ == "__main__":
    # Test berbagai skenario
    budgets = [100.0, 1000.0, 10000.0]
    for b in budgets:
        backtest("models/ppo_gold_m1_smc", "data/xauusd_m1_rl_train.csv", b)
        backtest("models/ppo_gold_m5_smc", "data/xauusd_m5_rl_train.csv", b)
