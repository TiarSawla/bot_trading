import os
import sys
import argparse
import pandas as pd
from stable_baselines3 import PPO
from rl_environment_agresif import GoldAggressiveGridEnv

def train():
    parser = argparse.ArgumentParser(description="Train PPO Aggressive Grid Agent")
    parser.add_argument("--csv", type=str, default="data/xauusd_mtf_agresif.csv", help="Path dataset CSV")
    parser.add_argument("--output", type=str, default="models/ppo_gold_agresif_mtf_v1", help="Output model path")
    parser.add_argument("--timesteps", type=int, default=3000000, help="Training timesteps")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.csv}")

    print(f"[INFO] Memuat dataset {args.csv}...")
    df = pd.read_csv(args.csv)
    
    print("[INFO] Inisialisasi Environment...")
    env = GoldAggressiveGridEnv(
        df=df,
        initial_balance=10000.0,
        layer_distance=2.0,
        max_layers=5,
        lot_step=0.01,
        tp_from_first_entry=1.0,
        basket_sl_from_last_entry=4.0
    )

    print("[INFO] Inisialisasi PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log="./tensorboard_logs/AGR_MTF_V1/"
    )

    print(f"[INFO] Training for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"[SUCCESS] Model saved to {args.output}.zip")

if __name__ == "__main__":
    train()
