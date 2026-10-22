# Installation Guide for Dependencies:
# pip install stable-baselines3[extra] pandas gymnasium

import os
import sys
import argparse
import pandas as pd

# Menambahkan root directory ke sys.path agar module 'src' dapat ditemukan saat dijalankan langsung
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from src.rl_environment import GoldTradingEnv

def train():
    parser = argparse.ArgumentParser(description="Train PPO Gold Trading Agent")
    parser.add_argument("--csv", type=str, default="data/xauusd_m15_rl_train.csv", help="Path dataset CSV")
    parser.add_argument("--output", type=str, default="models/ppo_gold_scalper_v4_smc", help="Model output path without .zip")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Training timesteps")
    parser.add_argument("--window-size", type=int, default=10, help="Observation window size")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="PPO learning rate")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {csv_path}. Jalankan rl_data_miner.py terlebih dahulu.")
        
    print(f"Memuat dataset dari {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset berhasil dimuat. Total bar/baris data: {len(df)}")
    
    print("Menginisialisasi GoldTradingEnv...")
    env = GoldTradingEnv(df, window_size=args.window_size)
    print(f"Observation shape: {env.observation_space.shape}")
    
    print("Inisialisasi model PPO dengan MlpPolicy...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print(f"Memulai pelatihan model PPO untuk {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps)
    
    models_dir = os.path.dirname(args.output) or "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Menyimpan model yang telah dilatih ke {args.output}...")
    model.save(args.output)
    
    print(f"Sukses: Model RL PPO berhasil dilatih dan disimpan ke '{args.output}.zip'!")

if __name__ == "__main__":
    train()
