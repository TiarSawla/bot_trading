import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("[ERROR] MetaTrader5 library is not installed. Please install it using: pip install MetaTrader5")
    sys.exit(1)

try:
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_classic as ta
    except ImportError:
        print("[ERROR] pandas_ta library is not installed. Please install it using: pip install pandas-ta-classic")
        sys.exit(1)

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


def fetch_rates(mt5_ctx, symbol: str, timeframe, bars: int):
    mt5_ctx.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, min(bars, 100000))
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        raise RuntimeError(f"Failed to fetch {symbol} (timeframe={timeframe}, bars={bars}): {err}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def add_smc_features(df: pd.DataFrame):
    n = len(df)
    lb = 3
    roll_high = df['high'].rolling(window=lb * 2 + 1, center=True, min_periods=1).max()
    roll_low = df['low'].rolling(window=lb * 2 + 1, center=True, min_periods=1).min()
    df['swing_high'] = (df['high'] == roll_high).astype(int)
    df['swing_low'] = (df['low'] == roll_low).astype(int)
    edge_mask = pd.Series(False, index=df.index)
    edge_mask.iloc[:lb] = True
    edge_mask.iloc[-lb:] = True
    df.loc[edge_mask, 'swing_high'] = 0
    df.loc[edge_mask, 'swing_low'] = 0

    sh_sofar = df['high'].where(df['swing_high'] == 1).ffill().shift(1)
    sl_sofar = df['low'].where(df['swing_low'] == 1).ffill().shift(1)
    df['bos_status'] = 0
    df.loc[df['high'] > sh_sofar, 'bos_status'] = 1
    df.loc[(df['bos_status'] == 0) & (df['low'] < sl_sofar), 'bos_status'] = -1

    h1, h2, h3 = df['high'].shift(2), df['high'].shift(1), df['high']
    l2, l3 = df['low'].shift(1), df['low']
    bullish_mss = (h3 > h2) & (l3 > l2) & (h2 > h1)
    bearish_mss = (h3 < h2) & (l3 < l2) & (h2 < h1)
    df['mss_status'] = 0
    df.loc[bullish_mss, 'mss_status'] = 1
    df.loc[bearish_mss, 'mss_status'] = -1

    df['liquidity_sweep'] = 0
    sweep_up = (df['low'] < df['low'].shift(1)) & (df['close'] > df['close'].shift(1))
    sweep_dn = (df['high'] > df['high'].shift(1)) & (df['close'] < df['close'].shift(1))
    df.loc[sweep_up, 'liquidity_sweep'] = 1
    df.loc[sweep_dn, 'liquidity_sweep'] = -1
    return df


def add_features(df: pd.DataFrame, prefix: str = ""):
    out = df.copy()
    if "volume" not in out.columns:
        out["volume"] = out["tick_volume"] if "tick_volume" in out.columns else 1.0
    out["ema_9"] = ta.ema(out["close"], length=9)
    out["ema_21"] = ta.ema(out["close"], length=21)
    out["ema_50"] = ta.ema(out["close"], length=50)
    out["ema_200"] = ta.ema(out["close"], length=200)
    out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    adx = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx is not None:
        adx_col = [c for c in adx.columns if c.startswith("ADX")][0]
        out["adx_14"] = adx[adx_col]
    else:
        out["adx_14"] = 0.0
    out["candle_range"] = out["high"] - out["low"]
    body = (out["close"] - out["open"]).abs()
    out["body_ratio"] = body / out["candle_range"].replace(0, np.nan)
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["volume_sma_20"] = out["volume"].rolling(20).mean()
    out["volume_spike"] = (out["volume"] > out["volume_sma_20"] * 2.0).astype(int)
    out["trend_ema"] = np.where(out["ema_9"] > out["ema_21"], 1, np.where(out["ema_9"] < out["ema_21"], -1, 0))
    out["macro_trend"] = np.where(out["close"] > out["ema_200"], 1, np.where(out["close"] < out["ema_200"], -1, 0))
    out["fvg_status"] = 0
    out.loc[out["low"] > out["high"].shift(2), "fvg_status"] = 1
    out.loc[out["high"] < out["low"].shift(2), "fvg_status"] = -1
    out = add_smc_features(out)
    keep = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume", "volume", "ema_9", "ema_21", "ema_50", "ema_200", "atr_14", "adx_14", "candle_range", "body_ratio", "upper_wick", "lower_wick", "volume_spike", "trend_ema", "macro_trend", "fvg_status", "bos_status", "mss_status", "liquidity_sweep"]
    out = out[[c for c in keep if c in out.columns]]
    if prefix:
        out = out.add_prefix(prefix)
    return out


def main():
    parser = argparse.ArgumentParser(description="MTF dataset miner for aggressive grid PPO")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--bars", type=int, default=200000)
    parser.add_argument("--output", type=str, default="data/xauusd_mtf_agresif.csv")
    args = parser.parse_args()

    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    symbol = args.symbol
    if mt5.symbol_info(symbol) is None:
        alternatives = [symbol, symbol + "m", symbol + "b", symbol + "c", "XAUUSD", "GOLD"]
        found = next((s for s in alternatives if mt5.symbol_info(s) is not None), None)
        if found is None:
            mt5.shutdown()
            raise RuntimeError(f"Symbol not found: {symbol}")
        symbol = found
    mt5.symbol_select(symbol, True)

    print(f"[INFO] Fetching M1={args.bars}, M5/M15/H1 context bars...")
    m1 = fetch_rates(mt5, symbol, TIMEFRAME_MAP["M1"], args.bars)
    m5 = fetch_rates(mt5, symbol, TIMEFRAME_MAP["M5"], max(1000, args.bars // 5 + 500))
    m15 = fetch_rates(mt5, symbol, TIMEFRAME_MAP["M15"], max(1000, args.bars // 15 + 500))
    h1 = fetch_rates(mt5, symbol, TIMEFRAME_MAP["H1"], max(1000, args.bars // 60 + 500))
    mt5.shutdown()

    print("[INFO] Computing features...")
    base = add_features(m1, "")
    m5f = add_features(m5, "m5_")
    m15f = add_features(m15, "m15_")
    h1f = add_features(h1, "h1_")

    df = base.sort_index()
    df = pd.merge_asof(df, m5f.sort_index(), left_index=True, right_index=True, direction="backward")
    df = pd.merge_asof(df, m15f.sort_index(), left_index=True, right_index=True, direction="backward")
    df = pd.merge_asof(df, h1f.sort_index(), left_index=True, right_index=True, direction="backward")
    before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output)
    print(f"[SUCCESS] Saved {len(df)} rows to {args.output}. Dropped {before - len(df)} rows.")


if __name__ == "__main__":
    main()
