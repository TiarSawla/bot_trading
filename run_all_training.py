import os
import subprocess

# Daftar konfigurasi training
configs = [
    {"tf": "M1", "bars": 5000, "timesteps": 500000, "model": "models/ppo_gold_m1_smc"},
    {"tf": "M5", "bars": 5000, "timesteps": 500000, "model": "models/ppo_gold_m5_smc"}
]

python_exe = r"C:\Tiar\Trade\AI-Assistant\trade\env\Scripts\python.exe"

for cfg in configs:
    csv_file = f"data/xauusd_{cfg['tf'].lower()}_rl_train.csv"
    print(f"--- Memproses {cfg['tf']} ---")
    
    # 1. Generate data
    print(f"Generating data for {cfg['tf']}...")
    subprocess.run([python_exe, "src/rl_data_miner.py", "--symbol", "XAUUSD", "--timeframe", cfg['tf'], "--bars", str(cfg['bars']), "--output", csv_file], check=True)
    
    # 2. Train model
    print(f"Training model for {cfg['tf']}...")
    subprocess.run([python_exe, "src/train_agent.py", "--csv", csv_file, "--output", cfg['model'], "--timesteps", str(cfg['timesteps'])], check=True)

print("Semua proses selesai!")
