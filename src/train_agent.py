# Installation Guide for Dependencies:
# pip install stable-baselines3[extra] pandas gymnasium

import os
import sys
import pandas as pd

# Menambahkan root directory ke sys.path agar module 'src' dapat ditemukan saat dijalankan langsung
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO
from src.rl_environment import GoldTradingEnv

def train():
    # 1. Muat dataset historis
    csv_path = "data/xauusd_m15_rl_train.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di path: {csv_path}. Jalankan rl_data_miner.py terlebih dahulu.")
        
    print(f"Memuat dataset dari {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset berhasil dimuat. Total bar/baris data: {len(df)}")
    
    # 2. Inisialisasi Environment RL kustom
    print("Menginisialisasi GoldTradingEnv...")
    env = GoldTradingEnv(df)
    
    # 3. Buat model PPO dengan kebijakan MlpPolicy (tabular/flat feature observations)
    print("Inisialisasi model PPO dengan MlpPolicy...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # 4. Latih model RL (1,000,000 timesteps)
    total_timesteps = 1000000
    print(f"Memulai pelatihan model PPO untuk {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # 5. Buat direktori penyimpanan jika belum ada dan simpan model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, "ppo_gold_scalper_v3")
    print(f"Menyimpan model yang telah dilatih ke {model_save_path}...")
    model.save(model_save_path)
    
    print(f"Sukses: Model RL PPO berhasil dilatih dan disimpan ke '{model_save_path}.zip'!")

if __name__ == "__main__":
    train()
