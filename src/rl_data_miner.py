import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# MT5 Timeframe mapping
TIMEFRAME_MAP = {
    "M1": 1,          # mt5.TIMEFRAME_M1
    "M5": 5,          # mt5.TIMEFRAME_M5
    "M15": 15,        # mt5.TIMEFRAME_M15
    "M30": 16,        # mt5.TIMEFRAME_M30
    "H1": 16385,      # mt5.TIMEFRAME_H1
    "H4": 16388,      # mt5.TIMEFRAME_H4
    "D1": 16408,      # mt5.TIMEFRAME_D1
}

# MT5 library import
try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 library is not installed. Please install it using: pip install MetaTrader5")
    sys.exit(1)

# pandas_ta library import (handles both standard pandas_ta and pandas_ta_classic fork)
try:
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_classic as ta
    except ImportError:
        print("[ERROR] pandas_ta library is not installed. Please install it using: pip install pandas-ta-classic")
        sys.exit(1)


def generate_mock_data(num_bars=10000):
    """
    Generates realistic mock market data for testing feature engineering
    without requiring a running MetaTrader 5 terminal.
    """
    print("[INFO] Generating synthetic/mock market data for testing...")
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=num_bars, freq="15min")
    
    # Generate prices using a random walk simulating Gold (XAUUSD)
    np.random.seed(42)
    close_prices = [2300.0]
    for _ in range(num_bars - 1):
        # typical 15-minute price fluctuation for gold
        change = np.random.normal(0.0, 1.2)
        close_prices.append(max(100.0, close_prices[-1] + change))
        
    df = pd.DataFrame(index=dates)
    df["close"] = close_prices
    df["open"] = df["close"].shift(1).fillna(df["close"]) + np.random.normal(0, 0.4, num_bars)
    df["high"] = df[["open", "close"]].max(axis=1) + np.abs(np.random.normal(0, 0.8, num_bars))
    df["low"] = df[["open", "close"]].min(axis=1) - np.abs(np.random.normal(0, 0.8, num_bars))
    df["tick_volume"] = np.random.randint(100, 3000, size=num_bars)
    df["spread"] = np.random.randint(12, 28, size=num_bars)
    df["real_volume"] = df["tick_volume"] * 10
    
    # Ensure there are no NaNs in base columns
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.index.name = "time"
    return df


def fetch_mt5_data(symbol, timeframe, num_bars):
    """
    Connects to MetaTrader 5 and fetches the requested historical data.
    """
    print(f"[INFO] Initializing MetaTrader 5...")
    if not mt5.initialize():
        print(f"[ERROR] mt5.initialize() failed. Error code: {mt5.last_error()}")
        return None

    print(f"[INFO] MT5 initialized successfully. Version: {mt5.version()}")
    
    # Check if the symbol is available
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # Check standard symbol variations
        alternatives = [symbol, symbol + "m", symbol + "b", "GOLD", "XAUUSD"]
        found_symbol = None
        for alt in alternatives:
            info = mt5.symbol_info(alt)
            if info is not None:
                found_symbol = alt
                break
        
        if found_symbol is None:
            print(f"[ERROR] Symbol '{symbol}' (and variations) not found in MT5 Market Watch.")
            mt5.shutdown()
            return None
        else:
            print(f"[INFO] Original symbol '{symbol}' not found. Using available variant: '{found_symbol}'")
            symbol = found_symbol

    # Make sure the symbol is selected in Market Watch
    if not mt5.symbol_select(symbol, True):
        print(f"[WARNING] Could not select symbol '{symbol}' in Market Watch.")

    # Copy rates from MT5
    print(f"[INFO] Copying {num_bars} rates of {symbol} for timeframe {timeframe}...")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    
    mt5.shutdown()
    print("[INFO] MetaTrader 5 connection closed.")

    if rates is None or len(rates) == 0:
        print(f"[ERROR] Failed to fetch rates from MT5. Error: {mt5.last_error()}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def compute_features(df):
    """
    Performs Feature Engineering on the OHLCV DataFrame using pandas_ta.
    """
    print("[INFO] Computing technical indicator features...")
    
    # Ensure volume exists for VWAP calculation
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 1.0

    # 1. VWAP daily
    # df.ta.vwap returns a Series. Assign to vwap.
    try:
        vwap_series = df.ta.vwap(anchor="D")
        if vwap_series is None or vwap_series.isna().all():
            raise ValueError("pandas_ta vwap returned None or NaN")
        df["vwap"] = vwap_series
    except Exception as e:
        print(f"[WARNING] pandas_ta VWAP failed ({e}). Calculating custom daily VWAP fallback...")
        # Custom daily VWAP calculation fallback
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = tp * df["volume"]
        df_tp = pd.DataFrame({"pv": pv, "volume": df["volume"]}, index=df.index)
        df_tp["date"] = df_tp.index.date
        cum_pv = df_tp.groupby("date")["pv"].cumsum()
        cum_v = df_tp.groupby("date")["volume"].cumsum()
        df["vwap"] = cum_pv / cum_v.replace(0, 1)

    df["vwap_dist_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    # 2. EMA 50
    df["ema_50"] = df.ta.ema(length=50)
    df["ema_50_dist_pct"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100

    # 3. ATR 14
    df["atr_14"] = df.ta.atr(length=14)

    # 4. Volatility Squeeze (Bollinger Bands vs Keltner Channels)
    # BBands (20, 2)
    bb = df.ta.bbands(length=20, std=2)
    # Keltner Channels (20, 1.5)
    kc = df.ta.kc(length=20, scalar=1.5)

    if bb is not None and kc is not None:
        # Dynamically locate columns to support version variants
        bbu_col = [c for c in bb.columns if "BBU" in c][0]
        bbl_col = [c for c in bb.columns if "BBL" in c][0]
        kcu_col = [c for c in kc.columns if "KCU" in c][0]
        kcl_col = [c for c in kc.columns if "KCL" in c][0]
        
        # is_squeeze = 1 if Bollinger Bands are inside Keltner Channels
        df["is_squeeze"] = ((bb[bbu_col] < kc[kcu_col]) & (bb[bbl_col] > kc[kcl_col])).astype(int)
    else:
        print("[WARNING] BBands or KC computation failed. Setting is_squeeze to 0.")
        df["is_squeeze"] = 0

    # 5. Fair Value Gap (FVG Status)
    # 1: Bullish FVG (Current low > High of 2 bars ago)
    # -1: Bearish FVG (Current high < Low of 2 bars ago)
    # 0: No FVG / Neutral
    df["fvg_status"] = 0
    df.loc[df["low"] > df["high"].shift(2), "fvg_status"] = 1
    df.loc[df["high"] < df["low"].shift(2), "fvg_status"] = -1
    
    df = add_smc_features(df)
    return df

def add_smc_features(df):
    atr = df["atr_14"].copy() if "atr_14" in df.columns else (df["high"] - df["low"]).rolling(14).mean()
    atr = atr.replace(0, 0.01).fillna(0.01)
    lb = 3
    df["swing_high"] = 0
    df["swing_low"] = 0
    n = len(df)
    for i in range(lb, n - lb):
        if df["high"].iloc[i] == max(df["high"].iloc[i - lb : i + lb + 1]):
            df.at[df.index[i], "swing_high"] = 1
        if df["low"].iloc[i] == min(df["low"].iloc[i - lb : i + lb + 1]):
            df.at[df.index[i], "swing_low"] = 1
    df["bos_status"] = 0
    sh_idx = df.index[df["swing_high"] == 1].tolist()
    sl_idx = df.index[df["swing_low"] == 1].tolist()
    for i in range(2, n):
        for idx in reversed(sh_idx):
            if idx < df.index[i] and df["high"].iloc[i] > df.loc[idx, "high"]:
                df.at[df.index[i], "bos_status"] = 1
                break
        if df["bos_status"].iloc[i] == 0:
            for idx in reversed(sl_idx):
                if idx < df.index[i] and df["low"].iloc[i] < df.loc[idx, "low"]:
                    df.at[df.index[i], "bos_status"] = -1
                    break
    df["mss_status"] = 0
    for i in range(5, n):
        h1, h2, h3 = df["high"].iloc[i - 2], df["high"].iloc[i - 1], df["high"].iloc[i]
        l2, l3 = df["low"].iloc[i - 1], df["low"].iloc[i]
        if h3 > h2 and l3 > l2 and h2 > h1:
            df.at[df.index[i], "mss_status"] = 1
        elif h3 < h2 and l3 < l2 and h2 < h1:
            df.at[df.index[i], "mss_status"] = -1
    df["fvg_strength"] = 0.0
    for i in range(2, n):
        atr_i = atr.iloc[i]
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            df.at[df.index[i], "fvg_strength"] = min((df["low"].iloc[i] - df["high"].iloc[i - 2]) / atr_i, 3.0)
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            df.at[df.index[i], "fvg_strength"] = min((df["low"].iloc[i - 2] - df["high"].iloc[i]) / atr_i, 3.0)
    df["liquidity_distance"] = 3.0
    for i in range(1, n):
        price = df["close"].iloc[i]
        atr_i = atr.iloc[i]
        d_high = [((df.loc[idx, "high"] - price) / atr_i) for idx in reversed(sh_idx) if idx <= df.index[i] and df.loc[idx, "high"] > price]
        d_low = [((price - df.loc[idx, "low"]) / atr_i) for idx in reversed(sl_idx) if idx <= df.index[i] and df.loc[idx, "low"] < price]
        cands = []
        if d_high:
            cands.append(d_high[0])
        if d_low:
            cands.append(d_low[0])
        if cands:
            df.at[df.index[i], "liquidity_distance"] = min(min(cands), 5.0)
    df["liquidity_sweep"] = 0
    for i in range(2, n):
        if df["low"].iloc[i] < df["low"].iloc[i - 1] and df["close"].iloc[i] > df["close"].iloc[i - 1]:
            df.at[df.index[i], "liquidity_sweep"] = 1
        elif df["high"].iloc[i] > df["high"].iloc[i - 1] and df["close"].iloc[i] < df["close"].iloc[i - 1]:
            df.at[df.index[i], "liquidity_sweep"] = -1
    return df

def main():
    parser = argparse.ArgumentParser(description="RL Historical Data Miner from MT5")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Symbol to mine (e.g. XAUUSD, XAUUSDm)")
    parser.add_argument("--timeframe", type=str, default="M15", choices=TIMEFRAME_MAP.keys(), help="Timeframe (default: M15)")
    parser.add_argument("--bars", type=int, default=10000, help="Number of bars to retrieve (default: 10000)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output CSV (default: data/<symbol>_<timeframe>_rl_train.csv)")
    parser.add_argument("--mock", action="store_true", help="Force mock data generation without connecting to MT5")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/{args.symbol.lower()}_{args.timeframe.lower()}_rl_train.csv"

    df = None

    if args.mock:
        df = generate_mock_data(args.bars)
    else:
        # Try fetching from MT5
        timeframe_val = TIMEFRAME_MAP[args.timeframe]
        df = fetch_mt5_data(args.symbol, timeframe_val, args.bars)
        if df is None:
            print("[WARNING] Could not fetch data from MT5. Falling back to Mock data to verify pipeline...")
            df = generate_mock_data(args.bars)

    # Compute features
    df = compute_features(df)

    # Record row count before dropping NaNs
    initial_rows = len(df)
    
    # Drop rows with NaN values (resulting from indicators like EMA, ATR, etc.)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    print(f"[INFO] Dropped {initial_rows - final_rows} rows containing NaN values.")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Export to CSV
    df.to_csv(args.output)
    print(f"[SUCCESS] Final dataset saved to '{args.output}'")
    print(f"[SUMMARY] Total processed rows successfully saved: {final_rows} rows.")


if __name__ == "__main__":
    main()
