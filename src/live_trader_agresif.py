import os
import sys
import argparse
import time
import math
import requests
from bs4 import BeautifulSoup
from datetime import datetime, time as dt_time
import logging
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from stable_baselines3 import PPO
from dotenv import load_dotenv

try:
    import pandas_ta as ta
except ImportError:
    try:
        import pandas_ta_classic as ta
    except ImportError:
        print('[ERROR] pandas_ta library is not installed. Please install it using: pip install pandas-ta-classic')
        sys.exit(1)
import schedule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('live_trader.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('LiveTrader')

model = None
SYMBOL = 'XAUUSDb'
LOT_SIZE = 0.01
MAGIC_NUMBER = 202606
DEVIATION = 20
DRY_RUN = False
ACCOUNT_TYPE = 'standard'  # 'standard', 'cent', atau 'premium'

# Load Environment variables early
load_dotenv()

# Aggressive Trading Configuration
AGGRESSIVE_MODE = os.getenv('AGGRESSIVE_MODE', 'false').strip().lower() == 'true'
MAX_ACTIVE_POSITIONS = int(os.getenv('MAX_ACTIVE_POSITIONS', '3' if AGGRESSIVE_MODE else '1'))
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv('HIGH_CONFIDENCE_THRESHOLD', '0.75'))
HIGH_CONFIDENCE_LOT_MULTIPLIER = float(os.getenv('HIGH_CONFIDENCE_LOT_MULTIPLIER', '2.0' if AGGRESSIVE_MODE else '1.0'))
USE_MARKET_ORDERS_FOR_HIGH_CONFIDENCE = os.getenv('USE_MARKET_ORDERS_FOR_HIGH_CONFIDENCE', 'true' if AGGRESSIVE_MODE else 'false').strip().lower() == 'true'

# Hyper-Aggressive MTF Scalper Config
SCALPER_MODE = os.getenv('SCALPER_MODE', 'false').strip().lower() == 'true'
SCALPER_INTERVAL_SECONDS = int(os.getenv('SCALPER_INTERVAL_SECONDS', '1'))
SCALPER_SL_USD = float(os.getenv('SCALPER_SL_USD', '0.0'))  # Default 0.0 (Tanpa SL agar di-handle Layering)
SCALPER_TP_MAX_USD = float(os.getenv('SCALPER_TP_MAX_USD', '10.0'))
SCALPER_MIN_M5_SCORE = int(os.getenv('SCALPER_MIN_M5_SCORE', '2'))
SCALPER_TRAIL_DISTANCE_USD = float(os.getenv('SCALPER_TRAIL_DISTANCE_USD', '0.8'))
SCALPER_PENDING_DISTANCE_USD = float(os.getenv('SCALPER_PENDING_DISTANCE_USD', '0.4'))

# Trading Strategy Configuration (Smart Layering - Hold Until Profit)
MAX_LAYERS = int(os.getenv('MAX_LAYERS', '7'))
BASKET_TP_USD = float(os.getenv('BASKET_TP_USD', '2.00'))
ROLLOVER_START = dt_time(0, 0)
ROLLOVER_END = dt_time(0, 5)

# Scalper & Grid Config
SCALPER_ATR_MULT = float(os.getenv('SCALPER_ATR_MULT', '1.2'))
SCALPER_PENDING_DISTANCE_PIPS = float(os.getenv('SCALPER_PENDING_DISTANCE_PIPS', '3.0'))
PENDING_ADJUST_THRESHOLD_USD = float(os.getenv('PENDING_ADJUST_THRESHOLD_USD', '0.3'))
PENDING_LOCK_ZONE_USD = float(os.getenv('PENDING_LOCK_ZONE_USD', '0.15'))
MIN_MARGIN_LEVEL_FOR_LAYERING = float(os.getenv('MIN_MARGIN_LEVEL_FOR_LAYERING', '200'))
HARD_CUT_MARGIN_LEVEL = float(os.getenv('HARD_CUT_MARGIN_LEVEL', '0'))      # 0 = Matikan Hard Cut Loss (Hold sampai profit)
EMERGENCY_MARGIN_LEVEL = float(os.getenv('EMERGENCY_MARGIN_LEVEL', '0'))    # 0 = Matikan Emergency Cut Loss
USE_INITIAL_SL = os.getenv('USE_INITIAL_SL', 'false').strip().lower() == 'true'
GRID_LAYER_DISTANCE_PIPS = float(os.getenv('GRID_LAYER_DISTANCE_PIPS', '35'))  # Jarak layering 35 pips ($3.50 di XAUUSD)
XAU_PIP_VALUE = float(os.getenv('XAU_PIP_VALUE', '0.10'))
CENT_LAYER_LOT_STEP = float(os.getenv('CENT_LAYER_LOT_STEP', '0.05'))
CENT_FIRST_LAYER_LOT = float(os.getenv('CENT_FIRST_LAYER_LOT', '0.05'))
STANDARD_LAYER_LOT_STEP = float(os.getenv('STANDARD_LAYER_LOT_STEP', '0.01'))
BASKET_SL_PIPS = float(os.getenv('BASKET_SL_PIPS', '0'))                    # 0 = Tanpa SL Basket
TP_FROM_FIRST_ENTRY_PIPS = float(os.getenv('TP_FROM_FIRST_ENTRY_PIPS', '12'))  # Target TP 12 pips ($1.20) dari harga rata-rata (BEP)
NEWS_CACHE_SECONDS = 300
NEWS_CACHE = {'checked_at': 0.0, 'blocked': False}

# Throttle & log-once state untuk manajemen SL/TP basket
_sltp_last_update = {}   # key -> timestamp update terakhir (anti-flicker 5 detik)
_last_sltp_log = {}      # key -> timestamp log terakhir (log sekali per jam)

MODEL_TIMEFRAMES = (
    ('', mt5.TIMEFRAME_M1),
    ('m5_', mt5.TIMEFRAME_M5),
    ('m15_', mt5.TIMEFRAME_M15),
    ('h1_', mt5.TIMEFRAME_H1),
)
MODEL_TF_FEATURES = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'volume', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'atr_14', 'adx_14',
    'candle_range', 'body_ratio', 'upper_wick', 'lower_wick', 'volume_spike',
    'trend_ema', 'macro_trend', 'fvg_status', 'bos_status', 'mss_status',
    'liquidity_sweep',
]
MODEL_MTF_FEATURES = [
    feature if not prefix else f'{prefix}{feature}'
    for prefix, _ in MODEL_TIMEFRAMES
    for feature in MODEL_TF_FEATURES
]
LEGACY_MODEL_FEATURES = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'volume', 'vwap', 'vwap_dist_pct', 'ema_50', 'ema_50_dist_pct', 'atr_14',
    'is_squeeze', 'fvg_status', 'swing_high', 'swing_low', 'bos_status',
    'mss_status', 'fvg_strength', 'liquidity_distance', 'liquidity_sweep',
]


def initialize_mt5():
    logger.info('Memuat konfigurasi lingkungan (.env)...')
    load_dotenv()
    account_str = os.getenv('MT5_ACCOUNT')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    if not all([account_str, password, server]):
        logger.error('Kredensial MT5 tidak lengkap di file .env! Pastikan MT5_ACCOUNT, MT5_PASSWORD, dan MT5_SERVER terisi.')
        sys.exit(1)
    try:
        account = int(account_str)
    except ValueError:
        logger.error(f"MT5_ACCOUNT '{account_str}' di .env bukan integer valid.")
        sys.exit(1)
    logger.info('Menginisialisasi MetaTrader 5...')
    if not mt5.initialize():
        logger.error(f'mt5.initialize() gagal. Kode Error: {mt5.last_error()}')
        sys.exit(1)
    logger.info(f'Mencoba login ke akun MT5 {account} di server {server}...')
    login_success = mt5.login(login=account, password=password, server=server)
    if not login_success:
        logger.error(f'Gagal login ke akun MT5. Kode Error: {mt5.last_error()}')
        mt5.shutdown()
        sys.exit(1)
    logger.info(f'Berhasil login ke server {server}!')
    return True


def get_dynamic_lot(symbol: str) -> float:
    """
    Dispatcher lot dinamis berdasarkan ACCOUNT_TYPE dan Balance:
    1. Akun CENT:
       - Balance >= 5000 USC -> Lot 0.70
       - Balance >= 1500 USC -> Lot 0.35
       - Balance < 1500 USC -> Lot 0.01
    2. Akun Standard, Premium, Zero:
       - Default Lot 0.01
       - Jika Balance >= 50,000,000 IDR (atau ekuivalen USD) -> Lot 0.05 atau lebih (+0.01 lot setiap kelipatan 10,000,000 IDR)
    """
    account_info = mt5.account_info()
    if account_info is None:
        logger.warning('[Lot] Gagal mengambil account_info. Menggunakan lot default 0.01')
        return 0.01

    balance = account_info.balance
    currency = getattr(account_info, 'currency', 'USD')

    symbol_idr = "USDIDR"
    fallback_rate = 16300.0
    tick = mt5.symbol_info_tick(symbol_idr)
    rate = tick.bid if (tick is not None and tick.bid > 0) else fallback_rate

    if ACCOUNT_TYPE == 'cent':
        if balance >= 5000:
            lot = 0.70
        elif balance >= 1500:
            lot = 0.35
        else:
            lot = 0.01
        logger.info(f"[Cent Lot] Balance: {balance:.2f} {currency} -> Lot: {lot}")
        return lot
    else:
        if currency == 'IDR':
            balance_idr = balance
        else:
            balance_idr = balance * rate

        if balance_idr < 50000000.0:
            lot = 0.01
            logger.info(f"[Standard/Premium/Zero Lot] Balance IDR Ekuivalen: Rp{balance_idr:,.2f} < Rp50,000,000 -> Lot: {lot}")
        else:
            extra_lot = ((balance_idr - 50000000.0) // 10000000.0) * 0.01
            lot = min(0.05 + extra_lot, 1.0)
            logger.info(f"[Standard/Premium/Zero Lot] Balance IDR Ekuivalen: Rp{balance_idr:,.2f} >= Rp50,000,000 -> Lot: {lot} (Base 0.05 + Extra {extra_lot:.2f})")

        return round(lot, 2)


def compute_features(df):
    if 'volume' not in df.columns:
        if 'tick_volume' in df.columns:
            df['volume'] = df['tick_volume']
        else:
            df['volume'] = 1.0
    try:
        vwap_series = df.ta.vwap(anchor='D')
        if vwap_series is None or vwap_series.isna().all():
            raise ValueError('pandas_ta vwap returned None or NaN')
        df['vwap'] = vwap_series
    except Exception as e:
        logger.warning(f'pandas_ta VWAP failed ({e}). Calculating custom daily VWAP fallback...')
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        pv = tp * df['volume']
        df_tp = pd.DataFrame({'pv': pv, 'volume': df['volume']}, index=df.index)
        df_tp['date'] = df_tp.index.date
        cum_pv = df_tp.groupby('date')['pv'].cumsum()
        cum_v = df_tp.groupby('date')['volume'].cumsum()
        df['vwap'] = cum_pv / cum_v.replace(0, 1)
    df['vwap_dist_pct'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    df['ema_50'] = df.ta.ema(length=50)
    df['ema_50_dist_pct'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
    df['atr_14'] = df.ta.atr(length=14)
    adx_df = df.ta.adx(length=14)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        logger.warning('ADX 14 calculation failed. Setting ADX_14 to NaN.')
        df['ADX_14'] = np.nan
    bb = df.ta.bbands(length=20, std=2)
    kc = df.ta.kc(length=20, scalar=1.5)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)
    if bb is not None and kc is not None:
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        kcu_col = [c for c in kc.columns if 'KCU' in c][0]
        kcl_col = [c for c in kc.columns if 'KCL' in c][0]
        df['is_squeeze'] = ((df[bbu_col] < kc[kcu_col]) & (df[bbl_col] > kc[kcl_col])).astype(int)
    else:
        logger.warning('BBands atau KC gagal dikalkulasi. Set is_squeeze ke 0.')
        df['is_squeeze'] = 0
    df['fvg_status'] = 0
    df.loc[df['low'] > df['high'].shift(2), 'fvg_status'] = 1
    df.loc[df['high'] < df['low'].shift(2), 'fvg_status'] = -1
    df = add_smc_features(df)

    vol_col = 'real_volume' if ('real_volume' in df.columns and df['real_volume'].astype(float).sum() > 0) else 'tick_volume'
    if vol_col not in df.columns:
        vol_col = 'volume'
    df['volume_sma_20'] = df[vol_col].rolling(window=20).mean()

    return df


def compute_model_features(df):
    """Build feature columns used by GoldAggressiveGridEnv during training."""
    out = df.copy()
    if 'volume' not in out.columns:
        out['volume'] = out['tick_volume'] if 'tick_volume' in out.columns else 1.0

    out['ema_9'] = ta.ema(out['close'], length=9)
    out['ema_21'] = ta.ema(out['close'], length=21)
    out['ema_50'] = ta.ema(out['close'], length=50)
    out['ema_200'] = ta.ema(out['close'], length=200)
    out['atr_14'] = ta.atr(out['high'], out['low'], out['close'], length=14)

    adx = ta.adx(out['high'], out['low'], out['close'], length=14)
    adx_col = next((col for col in adx.columns if col.startswith('ADX')), None) if adx is not None else None
    out['adx_14'] = adx[adx_col] if adx_col else np.nan

    out['candle_range'] = out['high'] - out['low']
    body = (out['close'] - out['open']).abs()
    out['body_ratio'] = body / out['candle_range'].replace(0, np.nan)
    out['upper_wick'] = out['high'] - out[['open', 'close']].max(axis=1)
    out['lower_wick'] = out[['open', 'close']].min(axis=1) - out['low']
    out['volume_sma_20'] = out['volume'].rolling(20).mean()
    out['volume_spike'] = (out['volume'] > out['volume_sma_20'] * 2.0).astype(int)
    out['trend_ema'] = np.where(out['ema_9'] > out['ema_21'], 1, np.where(out['ema_9'] < out['ema_21'], -1, 0))
    out['macro_trend'] = np.where(out['close'] > out['ema_200'], 1, np.where(out['close'] < out['ema_200'], -1, 0))
    out['fvg_status'] = 0
    out.loc[out['low'] > out['high'].shift(2), 'fvg_status'] = 1
    out.loc[out['high'] < out['low'].shift(2), 'fvg_status'] = -1
    out = add_smc_features(out)
    return out


def add_smc_features(df):
    atr = df['atr_14'].copy() if 'atr_14' in df.columns else (df['high'] - df['low']).rolling(14).mean()
    atr = atr.replace(0, 0.01).fillna(0.01)
    lb = 3
    roll_high = df['high'].rolling(window=lb * 2 + 1, center=True, min_periods=1).max()
    roll_low = df['low'].rolling(window=lb * 2 + 1, center=True, min_periods=1).min()
    edge_mask = pd.Series(False, index=df.index)
    edge_mask.iloc[:lb] = True
    edge_mask.iloc[-lb:] = True
    df['swing_high'] = ((df['high'] == roll_high) & ~edge_mask).astype(int)
    df['swing_low'] = ((df['low'] == roll_low) & ~edge_mask).astype(int)

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

    fvg_up = (df['low'] > df['high'].shift(2))
    fvg_dn = (df['high'] < df['low'].shift(2))
    df['fvg_strength'] = 0.0
    df.loc[fvg_up, 'fvg_strength'] = ((df.loc[fvg_up, 'low'] - df['high'].shift(2).loc[fvg_up]) / atr[fvg_up]).clip(upper=3.0)
    df.loc[fvg_dn, 'fvg_strength'] = ((df['low'].shift(2).loc[fvg_dn] - df.loc[fvg_dn, 'high']) / atr[fvg_dn]).clip(upper=3.0)

    df['liquidity_distance'] = 3.0
    sh_prices = df['high'].where(df['swing_high'] == 1).ffill()
    sl_prices = df['low'].where(df['swing_low'] == 1).ffill()
    price = df['close']
    atr_i = atr
    d_high = (sh_prices - price) / atr_i
    d_low = (price - sl_prices) / atr_i
    d_high = d_high.where((sh_prices > price) & (sh_prices.notna()), np.inf)
    d_low = d_low.where((sl_prices < price) & (sl_prices.notna()), np.inf)
    liquidity_dist = np.minimum(d_high, d_low).clip(lower=0, upper=5.0)
    df['liquidity_distance'] = liquidity_dist.fillna(3.0)

    sweep_up = (df['low'] < df['low'].shift(1)) & (df['close'] > df['close'].shift(1))
    sweep_dn = (df['high'] > df['high'].shift(1)) & (df['close'] < df['close'].shift(1))
    df['liquidity_sweep'] = 0
    df.loc[sweep_up, 'liquidity_sweep'] = 1
    df.loc[sweep_dn, 'liquidity_sweep'] = -1
    return df


def get_live_state(symbol):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 200)
    if rates is None or len(rates) == 0:
        logger.error(f'Gagal mengambil rates dari MT5 untuk {symbol}. Error: {mt5.last_error()}')
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = compute_features(df)
    df.dropna(inplace=True)
    features = df
    last_10_rows = features.iloc[-10:]
    if len(last_10_rows) < 10:
        logger.error(f'Jumlah baris fitur setelah dropna ({len(last_10_rows)}) kurang dari 10.')
        return None
    current_price = float(last_10_rows.iloc[-1]['close'])
    atr_value = float(last_10_rows.iloc[-1]['atr_14'])
    last_row = last_10_rows.iloc[-1]
    logger.info(f'--- Fitur Terkalkulasi (Waktu Candle Tutup: {last_row.name}) ---')
    logger.info(f"Open: {last_row['open']:.3f} | High: {last_row['high']:.3f} | Low: {last_row['low']:.3f} | Close: {last_row['close']:.3f}")
    logger.info(f"VWAP: {last_row['vwap']:.3f} | VWAP Dist Pct: {last_row['vwap_dist_pct']:.4f}%")
    logger.info(f"EMA 50: {last_row['ema_50']:.3f} | EMA 50 Dist Pct: {last_row['ema_50_dist_pct']:.4f}%")
    logger.info(f"ATR 14: {last_row['atr_14']:.4f}")
    if 'ADX_14' in last_row:
        logger.info(f"ADX 14: {last_row['ADX_14']:.2f}")
    if 'BBL_20_2.0' in last_row and 'BBU_20_2.0' in last_row:
        logger.info(f"BB 20,2: Lower={last_row['BBL_20_2.0']:.3f} | Mid={last_row['BBM_20_2.0']:.3f} | Upper={last_row['BBU_20_2.0']:.3f}")
    logger.info(f"Is Squeeze: {last_row['is_squeeze']} | FVG Status: {last_row['fvg_status']}")
    logger.info(f"Spread: {last_row['spread']} | Volume: {last_row['volume']}")
    logger.info('---------------------------------------------')
    feature_cols = LEGACY_MODEL_FEATURES
    try:
        features_subset = last_10_rows[feature_cols]
    except KeyError as e:
        logger.error(f'Kolom fitur tidak lengkap: {e}')
        return None
    obs = features_subset.to_numpy(dtype=np.float32).flatten()
    return (obs, current_price, atr_value, df)


def get_model_mtf_observation(symbol):
    frames = {}
    for prefix, timeframe in MODEL_TIMEFRAMES:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 320)
        if rates is None or len(rates) < 220:
            logger.error(f'[ML CONTRACT] Data {prefix or "m1"} tidak cukup untuk fitur model ({len(rates) if rates is not None else 0}/220).')
            return None
        frame = pd.DataFrame(rates)
        frame['time'] = pd.to_datetime(frame['time'], unit='s')
        frame.set_index('time', inplace=True)
        frame = compute_model_features(frame)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 10:
            logger.error(f'[ML CONTRACT] Fitur {prefix or "m1"} kurang dari 10 baris valid.')
            return None
        frame = frame[MODEL_TF_FEATURES].add_prefix(prefix)
        frames[prefix] = frame.sort_index()

    merged = frames['']
    for prefix, _ in MODEL_TIMEFRAMES[1:]:
        merged = pd.merge_asof(
            merged.sort_index(),
            frames[prefix].sort_index(),
            left_index=True,
            right_index=True,
            direction='backward',
        )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_MTF_FEATURES)
    if len(merged) < 10:
        logger.error(f'[ML CONTRACT] MTF merge menghasilkan {len(merged)} baris valid, minimal 10.')
        return None

    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f'[ML CONTRACT] Gagal mengambil posisi untuk state model. Error: {mt5.last_error()}')
        return None
    positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    buy_positions = [pos for pos in positions if pos.type == mt5.POSITION_TYPE_BUY]
    sell_positions = [pos for pos in positions if pos.type == mt5.POSITION_TYPE_SELL]
    side = 1 if buy_positions and not sell_positions else -1 if sell_positions and not buy_positions else 0
    total_volume = sum(float(pos.volume) for pos in positions)
    avg_price = sum(float(pos.volume) * float(pos.price_open) for pos in positions) / total_volume if total_volume else 0.0
    floating = sum(float(pos.profit) + float(pos.swap) for pos in positions)
    account_info = mt5.account_info()
    if account_info is None:
        logger.error('[ML CONTRACT] Gagal mengambil account_info untuk state model.')
        return None
    equity = float(getattr(account_info, 'equity', 0.0) or 0.0)

    tp_price = 0.0
    state_sl = 0.0
    side_positions = buy_positions or sell_positions
    if side_positions and side:
        distance = TP_FROM_FIRST_ENTRY_PIPS * XAU_PIP_VALUE
        tp_price = avg_price + distance if side == 1 else avg_price - distance
        active_sls = [pos.sl for pos in side_positions if pos.sl > 0]
        if active_sls:
            state_sl = max(active_sls) if side == 1 else min(active_sls)

    state = np.array([
        side,
        len(positions),
        avg_price,
        floating,
        equity,
        get_margin_level(),
        tp_price,
        state_sl,
    ], dtype=np.float32)
    market = merged.iloc[-10:][MODEL_MTF_FEATURES].to_numpy(dtype=np.float32).flatten()
    return np.concatenate([market, state])


def get_filling_mode(symbol):
    return get_filling_modes(symbol)[0]


def get_filling_modes(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]

    filling = int(getattr(symbol_info, 'filling_mode', 0) or 0)
    modes = []
    if filling & 1:
        modes.append(mt5.ORDER_FILLING_FOK)
    if filling & 2:
        modes.append(mt5.ORDER_FILLING_IOC)
    for mode in (mt5.ORDER_FILLING_RETURN, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC):
        if mode not in modes:
            modes.append(mode)
    return modes


def get_min_trade_distance(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return None
    stops = int(getattr(symbol_info, 'trade_stops_level', 0) or 0)
    freeze = int(getattr(symbol_info, 'trade_freeze_level', 0) or 0)
    points = max(stops, freeze, 30)
    return (points + 10) * symbol_info.point


def normalize_volume(symbol, volume):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return round(float(volume), 2)
    minimum = float(getattr(symbol_info, 'volume_min', 0.0) or 0.0)
    maximum = float(getattr(symbol_info, 'volume_max', 0.0) or 0.0)
    step = float(getattr(symbol_info, 'volume_step', 0.01) or 0.01)
    if minimum <= 0 or maximum <= 0 or step <= 0:
        return round(float(volume), 2)
    clipped = min(max(float(volume), minimum), maximum)
    normalized = math.floor((clipped + 1e-9) / step) * step
    return round(max(normalized, minimum), 8)


def close_position(position_ticket, position_type, symbol, volume, reason='UNSPECIFIED'):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk close. Tiket: {position_ticket}. Error: {mt5.last_error()}')
        return False
    if position_type == 0:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    elif position_type == 1:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        logger.error(f'Tipe posisi tidak dikenal: {position_type}')
        return False
    close_request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': normalize_volume(symbol, volume),
        'type': order_type,
        'position': position_ticket,
        'price': price,
        'deviation': DEVIATION,
        'magic': MAGIC_NUMBER,
        'comment': f'Close {reason}',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': get_filling_mode(symbol),
    }
    if DRY_RUN:
        logger.info(f'[DRY RUN] [{reason}] CLOSE order untuk tiket: {position_ticket}, volume: {volume} di harga {price}')
        return True

    invalid_fill = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)
    last_result = None
    for filling_mode in get_filling_modes(symbol):
        close_request['type_filling'] = filling_mode
        result = mt5.order_send(close_request)
        last_result = result
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f'[SUCCESS] [{reason}] Berhasil menutup tiket {position_ticket} (Vol: {volume}) di harga {result.price}')
            return True
        if result is None:
            break
        if result.retcode != invalid_fill:
            break

    if last_result is None:
        logger.error(f'[{reason}] Gagal menutup tiket {position_ticket}: hasil order_send None. Error: {mt5.last_error()}')
    else:
        logger.error(f'[{reason}] Gagal menutup tiket {position_ticket}. Retcode: {last_result.retcode}, Deskripsi: {last_result.comment}')
    return False


def open_position(symbol, order_type, volume, sl=None, tp=None):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk open. Error: {mt5.last_error()}')
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol info tidak ditemukan.")
        return False

    stops_level = max(
        int(getattr(symbol_info, 'trade_stops_level', 0) or 0),
        int(getattr(symbol_info, 'trade_freeze_level', 0) or 0),
        30,
    )
    min_distance = (stops_level + 10) * symbol_info.point

    sl_price = sl
    tp_price = tp

    if order_type == mt5.ORDER_TYPE_BUY:
        price = tick.ask
        type_str = 'BUY'
        current_bid = tick.bid
        if sl_price is not None:
            if sl_price > (current_bid - min_distance):
                adjusted_sl = round(current_bid - min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan SL BUY untuk {symbol}: {sl_price} -> {adjusted_sl} (min_distance: {min_distance:.5f})")
                sl_price = adjusted_sl
        if tp_price is not None:
            if tp_price < (current_bid + min_distance):
                adjusted_tp = round(current_bid + min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan TP BUY untuk {symbol}: {tp_price} -> {adjusted_tp} (min_distance: {min_distance:.5f})")
                tp_price = adjusted_tp

    elif order_type == mt5.ORDER_TYPE_SELL:
        price = tick.bid
        type_str = 'SELL'
        current_ask = tick.ask
        if sl_price is not None:
            if sl_price < (current_ask + min_distance):
                adjusted_sl = round(current_ask + min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan SL SELL untuk {symbol}: {sl_price} -> {adjusted_sl} (min_distance: {min_distance:.5f})")
                sl_price = adjusted_sl
        if tp_price is not None:
            if tp_price > (current_ask - min_distance):
                adjusted_tp = round(current_ask - min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan TP SELL untuk {symbol}: {tp_price} -> {adjusted_tp} (min_distance: {min_distance:.5f})")
                tp_price = adjusted_tp
    else:
        logger.error(f'Tipe order tidak didukung: {order_type}')
        return False

    volume = normalize_volume(symbol, volume)
    filling_mode = get_filling_mode(symbol)
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': volume,
        'type': order_type,
        'price': price,
        'deviation': DEVIATION,
        'magic': MAGIC_NUMBER,
        'comment': f'Open {type_str} position via RL Bot',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': filling_mode
    }
    if sl_price is not None:
        request['sl'] = sl_price
    if tp_price is not None:
        request['tp'] = tp_price
    if DRY_RUN:
        logger.info(f'[DRY RUN] OPEN {type_str} order dengan volume: {volume} di harga {price} (SL: {sl_price}, TP: {tp_price})')
        return True

    invalid_fill = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)
    last_result = None
    for filling_mode in get_filling_modes(symbol):
        request['type_filling'] = filling_mode
        result = mt5.order_send(request)
        last_result = result
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f'[SUCCESS] Berhasil membuka posisi {type_str} (Ticket: {result.order}, Vol: {volume}) di harga {result.price}')
            return True
        if result is None:
            break
        if result.retcode not in [invalid_fill, 10014, 10013]:
            break

    if last_result is None:
        logger.error(f'Gagal membuka posisi {type_str}: hasil order_send None. Error: {mt5.last_error()}')
    else:
        logger.error(f'Gagal membuka posisi {type_str}. Retcode: {last_result.retcode}, Deskripsi: {last_result.comment}')
    return False


def evade_round_number(price, is_buy_sl):
    remainder = price % 10
    if remainder < 2 or remainder > 8:
        buffer = 2.5
        if is_buy_sl:
            return price - buffer
        else:
            return price + buffer
    return price


def is_rollover_time():
    now = datetime.now().time()
    return ROLLOVER_START <= now <= ROLLOVER_END


def is_kill_zone_active():
    """Keep entries inside configured local London/New York activity windows."""
    now = datetime.now().time()
    london = dt_time(14, 0) <= now <= dt_time(17, 0)
    new_york = now >= dt_time(19, 0) or now <= dt_time(3, 0)
    return london or new_york


def check_high_impact_news():
    """
    Mengecek berita High Impact USD menggunakan endpoint JSON resmi ForexFactory CDN
    agar tidak terkena blokir SSL ISP (TrustPositif) maupun proteksi anti-bot Cloudflare.
    Set ENABLE_NEWS_FILTER=false di .env jika ingin mematikan filter berita sepenuhnya.
    """
    if os.getenv('ENABLE_NEWS_FILTER', 'true').strip().lower() == 'false':
        return False

    now = time.time()
    if now - NEWS_CACHE['checked_at'] < NEWS_CACHE_SECONDS:
        return NEWS_CACHE['blocked']

    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        events = response.json()

        today_str = datetime.now().strftime('%Y-%m-%d')
        for event in events:
            if event.get('country') == 'USD' and event.get('impact') == 'High':
                event_date = str(event.get('date', ''))
                if today_str in event_date:
                    logger.warning(f"[NEWS FILTER] Terdeteksi berita High Impact USD hari ini ({event.get('title')})! Entry baru ditangguhkan.")
                    NEWS_CACHE.update(checked_at=now, blocked=True)
                    return True

        NEWS_CACHE.update(checked_at=now, blocked=False)
        return False
    except Exception as e:
        # Jika fetch gagal (misal 429 rate-limit), PERTAHANAN state blok terakhir
        # (jangan reset ke non-blocked) agar filter tidak buta saat error.
        logger.warning(f"[NEWS FILTER] Gagal mengambil kalender berita ({e}). Mempertahankan state terakhir ({NEWS_CACHE['blocked']}).")
        NEWS_CACHE['checked_at'] = now
        return NEWS_CACHE['blocked']


def get_basket_avg_price(symbol, magic, pos_type=None):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return None

    total_volume = 0.0
    total_weighted_price = 0.0

    for pos in positions:
        if pos.magic == magic and (pos_type is None or pos.type == pos_type):
            total_volume += pos.volume
            total_weighted_price += pos.volume * pos.price_open

    return total_weighted_price / total_volume if total_volume > 0 else None


def get_next_lot(current_layer: int, symbol: str = None) -> float:
    """
    Skema Lot Smart Layering:
    - Akun Cent: Bertahap mulai dari CENT_FIRST_LAYER_LOT
    - Akun Standard/Premium/Zero: Soft Multiplier agar ketahanan margin kuat tapi BEP cepat mendekati harga market
      * Layer 1 - 2 : 1x base_lot (0.01)
      * Layer 3 - 4 : 2x base_lot (0.02)
      * Layer 5+    : 3x base_lot (0.03)
    """
    if ACCOUNT_TYPE == 'cent':
        if current_layer == 1:
            return round(CENT_FIRST_LAYER_LOT, 2)
        elif current_layer in (2, 3):
            return round(CENT_FIRST_LAYER_LOT + (current_layer - 1) * 0.02, 2)
        elif current_layer == 4:
            return round(CENT_FIRST_LAYER_LOT + 0.04 + CENT_LAYER_LOT_STEP, 2)
        else:
            return round(CENT_FIRST_LAYER_LOT + 0.04 + CENT_LAYER_LOT_STEP + (CENT_LAYER_LOT_STEP * 2), 2)

    base_lot = get_dynamic_lot(symbol or SYMBOL)
    multiplier = 1 if current_layer <= 2 else (2 if current_layer <= 4 else 3)
    return round(base_lot * multiplier, 2)


def get_margin_level() -> float:
    account_info = mt5.account_info()
    if account_info is None:
        logger.warning('[MARGIN] Gagal mengambil account_info. Layering diblokir sampai data tersedia.')
        return 0.0
    margin_level = float(getattr(account_info, 'margin_level', 0.0) or 0.0)
    if margin_level <= 0:
        return 9999.0
    return margin_level


def close_all_bot_positions(symbol: str, magic: int, reason='UNSPECIFIED'):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return
    for pos in positions:
        if pos.magic == magic:
            close_position(pos.ticket, pos.type, symbol, pos.volume, reason=reason)
            time.sleep(0.1)


def emergency_margin_protection(symbol: str, magic: int) -> bool:
    if EMERGENCY_MARGIN_LEVEL <= 0:
        return False
    margin_level = get_margin_level()
    if margin_level < EMERGENCY_MARGIN_LEVEL:
        logger.warning(f"[EMERGENCY MARGIN] Margin Level {margin_level:.2f}% < {EMERGENCY_MARGIN_LEVEL:.2f}%. Close semua posisi bot.")
        close_all_bot_positions(symbol, magic, reason='EMERGENCY_MARGIN')
        return True
    return False


def get_layer_distance(symbol: str, atr_value: float) -> float:
    """Jarak antar layer ditentukan oleh GRID_LAYER_DISTANCE_PIPS"""
    symbol_upper = symbol.upper()
    if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
        return GRID_LAYER_DISTANCE_PIPS * XAU_PIP_VALUE
    return 2.0


def manage_grid(symbol: str, magic: int):
    """
    Logika Smart Layering & Basket Averaging (Hold Until Profit):
    - Jika ada posisi yang sedang floating minus dan bergerak berlawanan sejauh GRID_LAYER_DISTANCE_PIPS,
      buka layer baru searah untuk menarik harga rata-rata (BEP) mendekati harga market.
    - Tidak melakukan Cut Loss selama HARD_CUT_MARGIN_LEVEL = 0.
    - Mengaktifkan profit-lock saat BASKET_TP_USD tercapai; close hanya melalui TP harga atau safety exit.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return

    my_positions = [p for p in positions if p.magic == magic]
    if not my_positions:
        return

    buy_pos = [p for p in my_positions if p.type == mt5.POSITION_TYPE_BUY]
    sell_pos = [p for p in my_positions if p.type == mt5.POSITION_TYPE_SELL]

    # Tetap jalankan evaluasi TP Basket meskipun sedang Rollover/News
    if is_rollover_time() or check_high_impact_news():
        logger.info("[GRID] Filter aktif (Rollover/News). Menahan penambahan layer baru, tetap memantau TP Basket.")
        manage_basket_tp_sl(symbol, magic, buy_pos, sell_pos)
        return

    if emergency_margin_protection(symbol, magic):
        return

    layer_distance = get_layer_distance(symbol, 0.0)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    margin_level = get_margin_level()
    can_add_layer = margin_level >= MIN_MARGIN_LEVEL_FOR_LAYERING

    # --- MANAGE BUY LAYERING ---
    if buy_pos:
        buy_pos.sort(key=lambda x: x.price_open)
        lowest_buy = buy_pos[0]  # Posisi BUY dengan harga paling rendah

        if len(buy_pos) < MAX_LAYERS and can_add_layer:
            next_entry = lowest_buy.price_open - layer_distance
            if tick.ask <= next_entry:
                next_layer = len(buy_pos) + 1
                lot = get_next_lot(next_layer, symbol)
                logger.info(f"[LAYERING BUY] Layer {next_layer}/{MAX_LAYERS}: Membuka BUY averaging di {tick.ask:.3f} | Lot: {lot}")
                opened = open_position(symbol, mt5.ORDER_TYPE_BUY, lot, None, None)
                time.sleep(0.2)
                if opened:
                    refreshed = mt5.positions_get(symbol=symbol)
                    if refreshed:
                        buy_pos = [p for p in refreshed if p.magic == magic and p.type == mt5.POSITION_TYPE_BUY]
                        sell_pos = [p for p in refreshed if p.magic == magic and p.type == mt5.POSITION_TYPE_SELL]

        elif HARD_CUT_MARGIN_LEVEL > 0 and len(buy_pos) >= MAX_LAYERS:
            total_profit = sum(p.profit + p.swap for p in buy_pos)
            if total_profit < 0 and margin_level < HARD_CUT_MARGIN_LEVEL:
                logger.warning(f"[HARD CUT LOSS] Margin Level {margin_level:.2f}% < {HARD_CUT_MARGIN_LEVEL:.2f}%. Close All BUY.")
                close_all_positions(symbol, magic, mt5.POSITION_TYPE_BUY, reason='HARD_CUT_MARGIN')
                return

    # --- MANAGE SELL LAYERING ---
    if sell_pos:
        sell_pos.sort(key=lambda x: x.price_open)
        highest_sell = sell_pos[-1]  # Posisi SELL dengan harga paling tinggi

        if len(sell_pos) < MAX_LAYERS and can_add_layer:
            next_entry = highest_sell.price_open + layer_distance
            if tick.bid >= next_entry:
                next_layer = len(sell_pos) + 1
                lot = get_next_lot(next_layer, symbol)
                logger.info(f"[LAYERING SELL] Layer {next_layer}/{MAX_LAYERS}: Membuka SELL averaging di {tick.bid:.3f} | Lot: {lot}")
                opened = open_position(symbol, mt5.ORDER_TYPE_SELL, lot, None, None)
                time.sleep(0.2)
                if opened:
                    refreshed = mt5.positions_get(symbol=symbol)
                    if refreshed:
                        buy_pos = [p for p in refreshed if p.magic == magic and p.type == mt5.POSITION_TYPE_BUY]
                        sell_pos = [p for p in refreshed if p.magic == magic and p.type == mt5.POSITION_TYPE_SELL]

        elif HARD_CUT_MARGIN_LEVEL > 0 and len(sell_pos) >= MAX_LAYERS:
            total_profit = sum(p.profit + p.swap for p in sell_pos)
            if total_profit < 0 and margin_level < HARD_CUT_MARGIN_LEVEL:
                logger.warning(f"[HARD CUT LOSS] Margin Level {margin_level:.2f}% < {HARD_CUT_MARGIN_LEVEL:.2f}%. Close All SELL.")
                close_all_positions(symbol, magic, mt5.POSITION_TYPE_SELL, reason='HARD_CUT_MARGIN')
                return

    # Kelola TP Basket secara kolektif berdasarkan harga rata-rata (BEP)
    manage_basket_tp_sl(symbol, magic, buy_pos, sell_pos)


def place_dual_pending_orders(symbol: str, magic: int):
    """Pasang BuyStop dan SellStop saat flat (opsional jika ingin mode jaring murni)"""
    orders = mt5.orders_get(symbol=symbol)
    if orders:
        existing = [o for o in orders if o.magic == magic and o.type in [mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP]]
        if existing:
            return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return

    atr_val = get_atr_value(symbol, mt5.TIMEFRAME_M1)
    distance = get_layer_distance(symbol, atr_val)

    lot = get_next_lot(1, symbol)
    digits = symbol_info.digits

    buy_price = round(tick.ask + distance, digits)
    sell_price = round(tick.bid - distance, digits)

    logger.info(f"[GRID INIT] Dual Pending: BUY_STOP {buy_price} | SELL_STOP {sell_price} | Distance: {distance:.3f}")
    place_pending_order(symbol, mt5.ORDER_TYPE_BUY_STOP, lot, buy_price, None, None)
    place_pending_order(symbol, mt5.ORDER_TYPE_SELL_STOP, lot, sell_price, None, None)


def close_all_positions(symbol: str, magic: int, pos_type: int, reason='UNSPECIFIED'):
    """Tutup semua posisi dengan tipe tertentu dan catat alasan close."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return
    for pos in positions:
        if pos.magic == magic and pos.type == pos_type:
            close_position(pos.ticket, pos.type, symbol, pos.volume, reason=reason)
            time.sleep(0.1)


def manage_basket_tp_sl(symbol: str, magic: int, buy_pos: list, sell_pos: list):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    symbol_info = mt5.symbol_info(symbol)
    digits = symbol_info.digits if symbol_info is not None else 3
    basket_sl_distance = BASKET_SL_PIPS * XAU_PIP_VALUE
    min_distance = get_min_trade_distance(symbol) or 0.0

    def update_positions_sltp(positions: list, new_sl=None, new_tp=None, label='BASKET',
                              avg_price=None, require_profit_side=False):
        for pos in positions:
            current_sl = pos.sl
            current_tp = pos.tp
            is_buy = pos.type == mt5.POSITION_TYPE_BUY
            candidate_sl = new_sl
            candidate_tp = new_tp
            if candidate_sl is not None and min_distance > 0:
                if is_buy:
                    candidate_sl = min(candidate_sl, tick.bid - min_distance)
                else:
                    candidate_sl = max(candidate_sl, tick.ask + min_distance)
                candidate_sl = round(candidate_sl, digits)
                if candidate_sl <= 0:
                    candidate_sl = None
            if candidate_tp is not None and min_distance > 0:
                tp_is_valid = candidate_tp > tick.bid + min_distance if is_buy else candidate_tp < tick.ask - min_distance
                if tp_is_valid:
                    candidate_tp = round(candidate_tp, digits)
                else:
                    candidate_tp = None

            # --- Guard P3: SL tidak boleh melewati TP ---
            # BUY:  SL harus di bawah TP.  Jika SL >= TP -> jangan kirim SL (biarkan TP penentu).
            # SELL: SL harus di atas TP.  Jika SL <= TP -> jangan kirim SL (biarkan TP penentu).
            if candidate_sl is not None and candidate_tp is not None:
                if is_buy and candidate_sl >= candidate_tp:
                    candidate_sl = None
                elif (not is_buy) and candidate_sl <= candidate_tp:
                    candidate_sl = None
            if candidate_sl is None and new_sl is not None:
                now_ts = time.time()
                key = ('SL_TP_GUARD', pos.ticket)
                if now_ts - _last_sltp_log.get(key, 0.0) > 3600:
                    _last_sltp_log[key] = now_ts
                    logger.warning(
                        f"[{label}] Tiket {pos.ticket} kandidat SL={new_sl} melewati TP={new_tp}; "
                        f"dibatalkan agar tidak memotong profit."
                    )

            # --- Guard sisi-profit (P1/P2): lock SL tidak boleh jatuh di sisi rugi ---
            # BUY: SL lock harus di atas avg. SELL: SL lock harus di bawah avg.
            # Jika clamp broker mendorong ke sisi rugi, pertahankan SL lama (jangan pemotong loss).
            if require_profit_side and candidate_sl is not None and avg_price is not None:
                profit_side_ok = candidate_sl > avg_price if is_buy else candidate_sl < avg_price
                if not profit_side_ok:
                    candidate_sl = None
                    now_ts = time.time()
                    key = ('LOSS_SIDE_SL', pos.ticket)
                    if now_ts - _last_sltp_log.get(key, 0.0) > 3600:
                        _last_sltp_log[key] = now_ts
                        logger.warning(
                            f"[{label}] Tiket {pos.ticket} SL={new_sl} terdorong ke sisi rugi "
                            f"(avg={avg_price:.2f}); dipertahankan SL lama agar tidak memotong loss."
                        )

            sl_changed = candidate_sl is not None and (current_sl == 0 or (is_buy and candidate_sl > current_sl) or (not is_buy and candidate_sl < current_sl))
            tp_changed = candidate_tp is not None and abs(current_tp - candidate_tp) > (10 ** -digits)
            if not sl_changed and not tp_changed:
                continue
            final_sl = candidate_sl if sl_changed else current_sl
            final_tp = candidate_tp if tp_changed else current_tp
            if DRY_RUN:
                logger.info(f"[DRY RUN] [{label}] Tiket {pos.ticket}: SL {current_sl}->{final_sl} | TP {current_tp}->{final_tp}")
                continue
            request = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "symbol": symbol, "sl": final_sl, "tp": final_tp}
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{label}] Tiket {pos.ticket} updated: SL={final_sl}, TP={final_tp}")
            elif result is None:
                logger.error(f"[{label}] Gagal update tiket {pos.ticket}: hasil None. Error: {mt5.last_error()}")
            else:
                logger.error(f"[{label}] Gagal update tiket {pos.ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}")

    def process_side(positions: list, is_buy: bool):
        if not positions:
            return
        total_profit = sum(p.profit + p.swap for p in positions)
        pos_type = mt5.POSITION_TYPE_BUY if is_buy else mt5.POSITION_TYPE_SELL
        avg_price = get_basket_avg_price(symbol, magic, pos_type)
        if avg_price is None:
            return

        positions.sort(key=lambda x: x.price_open)
        last_entry = positions[0] if is_buy else positions[-1]

        basket_pnl = total_profit

        # 1. Lock profit hanya mengaktifkan SL sisi-profit ketika basket sudah benar-benar profit
        #    (harga live melewati avg + min_distance di sisi profit). PnL >= target saja tidak cukup
        #    karena di akun premium/cent nilai PnL ter-skala sehingga threshold nominal mudah tercapai
        #    padahal harga masih di sisi rugi -> akan memasang SL yang kemudian memotong loss.
        on_profit_side = (
            (tick.bid - avg_price) > min_distance if is_buy
            else (avg_price - tick.ask) > min_distance
        )
        profit_lock_active = len(positions) >= 2 and on_profit_side and basket_pnl >= BASKET_TP_USD

        # 2. Hitung harga TP kolektif dari HARGA RATA-RATA (avg_price / BEP), bukan dari entry pertama!
        tp_distance = TP_FROM_FIRST_ENTRY_PIPS * XAU_PIP_VALUE
        tp_price = round(avg_price + tp_distance, digits) if is_buy else round(avg_price - tp_distance, digits)

        if profit_lock_active:
            logger.info(f"[BASKET PROFIT LOCK] {'BUY' if is_buy else 'SELL'} | PnL: ${basket_pnl:.2f} >= Target: ${BASKET_TP_USD:.2f} | TP: {tp_price} | Lock SL dipertahankan/ditingkatkan")

        # Tutup di market jika harga sudah melewati tp_price dan keranjang sudah positif
        target_reached = tick.bid >= tp_price if is_buy else tick.ask <= tp_price
        if target_reached and basket_pnl > 0:
            logger.info(f"[BASKET TP] Target rata-rata {'BUY' if is_buy else 'SELL'} ({tp_price}) tercapai (PnL: ${basket_pnl:.2f}); menutup seluruh basket di market.")
            close_all_positions(symbol, magic, pos_type, reason='BASKET_TP')
            return

        # 3. Pengaturan SL & Trailing Basket (Tanpa SL saat rugi jika BASKET_SL_PIPS == 0)
        new_sl = None
        if BASKET_SL_PIPS > 0 and len(positions) >= MAX_LAYERS:
            new_sl = round(last_entry.price_open - basket_sl_distance, digits) if is_buy else round(last_entry.price_open + basket_sl_distance, digits)

        # 4. Lock SL (break-even plus): hanya dipasang saat basket berada di sisi profit,
        #    menjaga SL tetap di sisi yang menguntungkan (BELUM memotong kerugian).
        #    Jika kandidat jatuh di sisi rugi, skip (tidak dipaksa) agar tidak jadi jebakan loss.
        if profit_lock_active:
            be_plus = 0.15
            lock_sl = round(avg_price + be_plus, digits) if is_buy else round(avg_price - be_plus, digits)
            if is_buy:
                # SL BUY harus berada di bawah harga live (bid - min_distance)
                if min_distance > 0:
                    lock_sl = min(lock_sl, round(tick.bid - min_distance, digits))
                if lock_sl > avg_price:
                    new_sl = lock_sl
                else:
                    # Kandidat lock jatuh di sisi rugi -> jangan pasang SL yang memotong loss.
                    now_ts = time.time()
                    key = ('LOCK_SLTP', pos_type)
                    last_log = _last_sltp_log.get(key, 0.0)
                    if now_ts - last_log > 3600:
                        _last_sltp_log[key] = now_ts
                        logger.warning(
                            f"[BASKET PROFIT LOCK] {('BUY' if is_buy else 'SELL')} kandidat SL "
                            f"{lock_sl} jatuh di sisi rugi (avg={avg_price:.2f}); skip agar tidak memotong loss."
                        )
            else:
                # SL SELL harus berada di atas harga live (ask + min_distance)
                if min_distance > 0:
                    lock_sl = max(lock_sl, round(tick.ask + min_distance, digits))
                if lock_sl < avg_price:
                    new_sl = lock_sl
                else:
                    now_ts = time.time()
                    key = ('LOCK_SLTP', pos_type)
                    last_log = _last_sltp_log.get(key, 0.0)
                    if now_ts - last_log > 3600:
                        _last_sltp_log[key] = now_ts
                        logger.warning(
                            f"[BASKET PROFIT LOCK] {('BUY' if is_buy else 'SELL')} kandidat SL "
                            f"{lock_sl} jatuh di sisi rugi (avg={avg_price:.2f}); skip agar tidak memotong loss."
                        )

        # 5. Trailing basket: aktif saat profit lock (P1) atau basket cukup profit.
        #    offset = 50% profit per-unit, dibatasi (a) min_distance agar SL valid,
        #    (b) 50% basket_pnl agar tidak mengambang jauh dari harga live
        #    (fokus: loss per reversal kecil, profit tetap dominan).
        #    Aturan "SL tidak boleh melewati TP" dipaksakan di update_positions_sltp.
        if profit_lock_active or basket_pnl >= 3.0:
            profit_per_unit = max(tick.bid - avg_price if is_buy else avg_price - tick.ask, 0.0)
            # Trailing agresif: SL mengunci 50% dari keuntungan per-unit.
            # Jika PnL basket masih kecil, SL tetap ketat (min_distance dari avg),
            # sehingga loss per-reversal kecil; profit tetap dominan.
            trail_offset = 0.5 * profit_per_unit
            trail_offset = max(trail_offset, min_distance)
            if is_buy:
                trail_sl = round(avg_price + trail_offset, digits)
                if trail_sl > avg_price and (new_sl is None or trail_sl > new_sl):
                    new_sl = trail_sl
            else:
                trail_sl = round(avg_price - trail_offset, digits)
                if trail_sl < avg_price and (new_sl is None or trail_sl < new_sl):
                    new_sl = trail_sl
            logger.info(f"[BASKET TRAIL] {'BUY' if is_buy else 'SELL'} | PnL: ${basket_pnl:.2f} | SL-> {new_sl}")

        # 6. Anti-flicker: tulis SL/TP hanya jika ada perubahan nyata, dan throttle
        #    per tiket 5 detik. Throttle di-level harga (bukan per tiket) karena semua
        #    tiket di basket ini memakai harga baru yang sama.
        def _sltp_wanted_differs(pos, sl_candidate, tp_candidate):
            if sl_candidate is None and tp_candidate is None:
                return False
            cur_sl, cur_tp = pos.sl, pos.tp
            if sl_candidate is not None and abs(sl_candidate - cur_sl) > (10 ** -digits):
                return True
            if tp_candidate is not None and abs(tp_candidate - cur_tp) > (10 ** -digits):
                return True
            return False

        need_update = any(_sltp_wanted_differs(p, new_sl, tp_price) for p in positions)
        now_ts = time.time()
        throttle_key = ('SLTP_THROTTLE', pos_type)
        last_update = _sltp_last_update.get(throttle_key, 0.0)
        throttled = need_update and (now_ts - last_update) < 5.0

        if not throttled and need_update:
            _sltp_last_update[throttle_key] = now_ts
            update_positions_sltp(
                positions, new_sl, tp_price,
                'BASKET PROFIT LOCK' if profit_lock_active else 'BASKET SLTP',
                avg_price=avg_price,
                require_profit_side=profit_lock_active,
            )
        elif need_update and profit_lock_active:
            # Log sesekali untuk menginformasikan bahwa update tertunda karena throttle
            if now_ts - _last_sltp_log.get(('TRAIL_THROTTLE', pos_type), 0.0) > 30:
                _last_sltp_log[('TRAIL_THROTTLE', pos_type)] = now_ts
                logger.debug(f"[BASKET TRAIL] Throttled (5s/tiket): akan di-update pada siklus berikutnya")

    process_side(buy_pos, True)
    process_side(sell_pos, False)


def check_macro_trend(symbol):
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, 250)
        if rates is None or len(rates) < 200:
            logger.warning(f"[MTF Filter H1] Gagal mengambil data H1 cukup ({len(rates) if rates is not None else 0}/200). Filter dilewati.")
            return None

        df = pd.DataFrame(rates)
        df['close'] = df['close'].astype(float)
        df['ema_200'] = ta.ema(df['close'], length=200)

        last_close = df['close'].iloc[-1]
        last_ema = df['ema_200'].iloc[-1]

        if pd.isna(last_ema):
            logger.warning("[MTF Filter H1] Kalkulasi EMA 200 H1 menghasilkan NaN. Filter dilewati.")
            return None

        logger.info(f"[MTF Filter H1] H1 Close: {last_close:.3f} | H1 EMA 200: {last_ema:.3f}")
        if last_close > last_ema:
            return 1
        elif last_close < last_ema:
            return 0
    except Exception as e:
        logger.error(f"[MTF Filter H1] Error dalam check_macro_trend: {e}")
        return None


def detect_order_block(symbol, timeframe, lookback=30):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
        if rates is None or len(rates) < 5:
            logger.warning(f"[OB DETECTION] Gagal mengambil rates dari MT5 atau data kurang ({len(rates) if rates is not None else 0}).")
            return None

        df = pd.DataFrame(rates)
        df['body_size'] = (df['close'] - df['open']).abs()

        sub_df = df.iloc[1:]
        if sub_df.empty:
            return None

        imp_idx = sub_df['body_size'].idxmax()
        imp_row = df.loc[imp_idx]

        is_bullish_imp = imp_row['close'] > imp_row['open']
        ob_type = "bullish" if is_bullish_imp else "bearish"

        opposite_idx = None
        if ob_type == "bullish":
            for j in range(imp_idx - 1, -1, -1):
                if df.loc[j, 'close'] < df.loc[j, 'open']:
                    opposite_idx = j
                    break
        else:
            for j in range(imp_idx - 1, -1, -1):
                if df.loc[j, 'close'] > df.loc[j, 'open']:
                    opposite_idx = j
                    break

        if opposite_idx is None:
            opposite_idx = imp_idx - 1

        ob_top = float(df.loc[opposite_idx, 'high'])
        ob_bottom = float(df.loc[opposite_idx, 'low'])

        return ob_top, ob_bottom, ob_type

    except Exception as e:
        logger.error(f"[OB DETECTION] Error saat mendeteksi Order Block: {e}")
        return None


def get_atr_value(symbol, timeframe):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 50)
    if rates is None or len(rates) < 15:
        return 0.5
    df = pd.DataFrame(rates)
    try:
        atr = df.ta.atr(length=14)
        if atr is not None and not atr.isna().all():
            return float(atr.iloc[-1])
    except Exception:
        pass
    hl = df['high'] - df['low']
    return float(hl.rolling(14).mean().iloc[-1])


def get_tf_trend(symbol, timeframe):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 30)
        if rates is None or len(rates) < 22:
            return "NEUTRAL"
        df = pd.DataFrame(rates)
        df['close'] = df['close'].astype(float)

        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)

        last_row = df.iloc[-1]
        ema_9 = last_row['ema_9']
        ema_21 = last_row['ema_21']

        if pd.isna(ema_9) or pd.isna(ema_21):
            return "NEUTRAL"

        if ema_9 > ema_21:
            return "BUY"
        elif ema_9 < ema_21:
            return "SELL"
        return "NEUTRAL"
    except Exception as e:
        logger.error(f"[TF TREND] Error on timeframe {timeframe} for {symbol}: {e}")
        return "NEUTRAL"


def micro_trailing_stop(symbol):
    """
    Step Trailing Stop khusus ketika hanya ada 1 layer posisi yang sedang profit.
    Jika sedang multi-layer (>1 posisi), manajemen TP/SL sepenuhnya dipegang oleh manage_basket_tp_sl.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return

    bot_positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    if len(bot_positions) != 1:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"[MICRO TRAILING] Gagal mengambil tick untuk {symbol}")
        return

    symbol_info = mt5.symbol_info(symbol)
    digits = symbol_info.digits if symbol_info is not None else 3
    min_distance = get_min_trade_distance(symbol) or 0.0

    pos = bot_positions[0]
    ticket = pos.ticket
    open_price = pos.price_open
    current_sl = pos.sl
    current_tp = pos.tp
    target_sl = None

    if pos.type == 0:  # BUY
        current_price = tick.bid
        gain = current_price - open_price

        if gain >= 8.0:
            level = 8.0 + int((gain - 8.0) / 3.0) * 3.0
            lock_profit = level - 3.0
            target_sl = round(open_price + lock_profit, digits)
        elif gain >= 5.0:
            target_sl = round(open_price + 3.0, digits)
        elif gain >= 2.5:
            target_sl = round(open_price + 1.2, digits)
        elif gain >= 1.0:
            target_sl = round(open_price + 0.3, digits)

        if target_sl is not None:
            if min_distance > 0:
                target_sl = min(target_sl, tick.bid - min_distance)
                target_sl = round(target_sl, digits)
            if target_sl <= 0:
                target_sl = None
        if target_sl is not None and target_sl > current_sl:
            if DRY_RUN:
                logger.info(f"[DRY RUN] [STEP TRAIL] BUY Tiket {ticket}: SL {current_sl} -> {target_sl} (Gain: {gain:.2f})")
            else:
                request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": target_sl, "tp": current_tp}
                res = mt5.order_send(request)
                if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[SUCCESS] [STEP TRAIL] BUY Tiket {ticket} SL digeser ke {target_sl}")

    elif pos.type == 1:  # SELL
        current_price = tick.ask
        gain = open_price - current_price

        if gain >= 8.0:
            level = 8.0 + int((gain - 8.0) / 3.0) * 3.0
            lock_profit = level - 3.0
            target_sl = round(open_price - lock_profit, digits)
        elif gain >= 5.0:
            target_sl = round(open_price - 3.0, digits)
        elif gain >= 2.5:
            target_sl = round(open_price - 1.2, digits)
        elif gain >= 1.0:
            target_sl = round(open_price - 0.3, digits)

        if target_sl is not None:
            if min_distance > 0:
                target_sl = max(target_sl, tick.ask + min_distance)
                target_sl = round(target_sl, digits)
        if target_sl is not None and (current_sl == 0.0 or target_sl < current_sl):
            if DRY_RUN:
                logger.info(f"[DRY RUN] [STEP TRAIL] SELL Tiket {ticket}: SL {current_sl} -> {target_sl} (Gain: {gain:.2f})")
            else:
                request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "symbol": symbol, "sl": target_sl, "tp": current_tp}
                res = mt5.order_send(request)
                if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"[SUCCESS] [STEP TRAIL] SELL Tiket {ticket} SL digeser ke {target_sl}")


def place_pending_order(symbol, order_type, volume, price, sl=None, tp=None):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk pending. Error: {mt5.last_error()}')
        return False

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol info tidak ditemukan.")
        return False

    stops_level = max(
        int(getattr(symbol_info, 'trade_stops_level', 0) or 0),
        int(getattr(symbol_info, 'trade_freeze_level', 0) or 0),
        30,
    )
    min_distance = (stops_level + 10) * symbol_info.point

    volume = normalize_volume(symbol, volume)
    entry_price = round(price, symbol_info.digits)
    sl_price = round(sl, symbol_info.digits) if sl is not None else None
    tp_price = round(tp, symbol_info.digits) if tp is not None else None

    current_bid = tick.bid
    current_ask = tick.ask

    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
        type_str = 'BUY_LIMIT'
        if entry_price > (current_ask - min_distance):
            entry_price = round(current_ask - min_distance, symbol_info.digits)
        if sl_price is not None and sl_price > (entry_price - min_distance):
            sl_price = round(entry_price - min_distance, symbol_info.digits)
        if tp_price is not None and tp_price < (entry_price + min_distance):
            tp_price = round(entry_price + min_distance, symbol_info.digits)

    elif order_type == mt5.ORDER_TYPE_SELL_LIMIT:
        type_str = 'SELL_LIMIT'
        if entry_price < (current_bid + min_distance):
            entry_price = round(current_bid + min_distance, symbol_info.digits)
        if sl_price is not None and sl_price < (entry_price + min_distance):
            sl_price = round(entry_price + min_distance, symbol_info.digits)
        if tp_price is not None and tp_price > (entry_price - min_distance):
            tp_price = round(entry_price - min_distance, symbol_info.digits)

    elif order_type == mt5.ORDER_TYPE_BUY_STOP:
        type_str = 'BUY_STOP'
        if entry_price < (current_ask + min_distance):
            entry_price = round(current_ask + min_distance, symbol_info.digits)
        if sl_price is not None and sl_price > (entry_price - min_distance):
            sl_price = round(entry_price - min_distance, symbol_info.digits)
        if tp_price is not None and tp_price < (entry_price + min_distance):
            tp_price = round(entry_price + min_distance, symbol_info.digits)

    elif order_type == mt5.ORDER_TYPE_SELL_STOP:
        type_str = 'SELL_STOP'
        if entry_price > (current_bid - min_distance):
            entry_price = round(current_bid - min_distance, symbol_info.digits)
        if sl_price is not None and sl_price < (entry_price + min_distance):
            sl_price = round(entry_price + min_distance, symbol_info.digits)
        if tp_price is not None and tp_price > (entry_price - min_distance):
            tp_price = round(entry_price - min_distance, symbol_info.digits)

    else:
        logger.error(f'Tipe pending order tidak didukung: {order_type}')
        return False

    if sl_price is not None:
        sl_price = round(evade_round_number(sl_price, is_buy_sl=(order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP])), symbol_info.digits)

    filling_mode = get_filling_mode(symbol)
    request = {
        'action': mt5.TRADE_ACTION_PENDING,
        'symbol': symbol,
        'volume': volume,
        'type': order_type,
        'price': entry_price,
        'deviation': DEVIATION,
        'magic': MAGIC_NUMBER,
        'comment': f'Pending {type_str} via RL Bot',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': filling_mode
    }
    if sl_price is not None:
        request['sl'] = sl_price
    if tp_price is not None:
        request['tp'] = tp_price

    if DRY_RUN:
        logger.info(f'[DRY RUN] Place pending {type_str} order dengan volume: {volume} di harga {entry_price} (SL: {sl_price}, TP: {tp_price})')
        return True

    invalid_fill = getattr(mt5, 'TRADE_RETCODE_INVALID_FILL', 10030)
    placed_code = getattr(mt5, 'TRADE_RETCODE_PLACED', 10008)
    last_result = None
    for filling_mode in get_filling_modes(symbol):
        request['type_filling'] = filling_mode
        result = mt5.order_send(request)
        last_result = result
        if result is not None and result.retcode in [mt5.TRADE_RETCODE_DONE, placed_code]:
            logger.info(f'[SUCCESS] Berhasil menempatkan pending {type_str} (Ticket: {getattr(result, "order", 0)}, Vol: {volume}) di harga {entry_price} (SL: {sl_price}, TP: {tp_price})')
            return True
        if result is None:
            break
        if result.retcode not in [invalid_fill, 10014, 10013]:
            break

    if last_result is None:
        logger.error(f'Gagal menempatkan pending {type_str}: hasil order_send None. Error: {mt5.last_error()}')
    else:
        logger.error(f'Gagal menempatkan pending {type_str}. Retcode: {last_result.retcode}, Deskripsi: {last_result.comment}')
    return False


def cancel_pending_order(order_ticket):
    existing = mt5.orders_get(ticket=order_ticket)
    if existing is None:
        logger.error(f'Gagal membaca pending order {order_ticket} sebelum cancel: {mt5.last_error()}')
        return False
    if len(existing) == 0:
        logger.info(f'[PENDING ORDER] Tiket {order_ticket} sudah tidak aktif; cancel dilewati.')
        return True
    request = {
        'action': mt5.TRADE_ACTION_REMOVE,
        'order': order_ticket,
        'deviation': DEVIATION,
        'magic': MAGIC_NUMBER
    }
    if DRY_RUN:
        logger.info(f'[DRY RUN] CANCEL pending order tiket: {order_ticket}')
        return True
    result = mt5.order_send(request)
    if result is None:
        logger.error(f'Gagal menghapus pending order {order_ticket}: hasil order_send None. Error: {mt5.last_error()}')
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f'Gagal menghapus pending order {order_ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}')
        return False
    logger.info(f'[SUCCESS] Berhasil menghapus pending order {order_ticket}')
    return True


def modify_pending_order(order_ticket, new_price=None, new_sl=None, new_tp=None):
    existing = mt5.orders_get(ticket=order_ticket)
    if existing is None:
        logger.error(f'Gagal membaca pending order {order_ticket} sebelum modifikasi: {mt5.last_error()}')
        return False
    if not existing:
        logger.info(f'[PENDING ORDER] Tiket {order_ticket} sudah tidak aktif; modifikasi dilewati.')
        return True
    if new_price is None and new_sl is None and new_tp is None:
        logger.warning(f'[PENDING ORDER] Tidak ada perubahan untuk tiket {order_ticket}.')
        return False

    order = existing[0]
    symbol = order.symbol
    order_type = order.type
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    if tick is None or symbol_info is None:
        logger.error(f'Gagal membaca tick/info symbol untuk modifikasi order {order_ticket}.')
        return False

    min_distance = get_min_trade_distance(symbol) or 0.0
    entry_price = new_price if new_price is not None else order.price_open
    if order_type == mt5.ORDER_TYPE_BUY_LIMIT:
        entry_price = min(entry_price, tick.ask - min_distance)
    elif order_type == mt5.ORDER_TYPE_SELL_LIMIT:
        entry_price = max(entry_price, tick.bid + min_distance)
    elif order_type == mt5.ORDER_TYPE_BUY_STOP:
        entry_price = max(entry_price, tick.ask + min_distance)
    elif order_type == mt5.ORDER_TYPE_SELL_STOP:
        entry_price = min(entry_price, tick.bid - min_distance)
    entry_price = round(entry_price, symbol_info.digits)
    if new_price is not None:
        new_price = entry_price
    if new_sl is not None:
        if order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
            new_sl = min(new_sl, entry_price - min_distance)
        else:
            new_sl = max(new_sl, entry_price + min_distance)
        new_sl = round(evade_round_number(new_sl, order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]), symbol_info.digits)
    if new_tp is not None:
        if order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
            new_tp = max(new_tp, entry_price + min_distance)
        else:
            new_tp = min(new_tp, entry_price - min_distance)
        new_tp = round(new_tp, symbol_info.digits)

    request = {
        "action": mt5.TRADE_ACTION_MODIFY,
        "order": order_ticket,
        "symbol": symbol,
    }
    if new_price is not None:
        request["price"] = new_price
    if new_sl is not None:
        request["sl"] = new_sl
    if new_tp is not None:
        request["tp"] = new_tp

    if DRY_RUN:
        logger.info(f"[DRY RUN] MODIFY pending order {order_ticket}: Price={new_price}, SL={new_sl}, TP={new_tp}")
        return True

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Gagal memodifikasi order {order_ticket}: {mt5.last_error()}")
        return False
    logger.info(f"[SUCCESS] Berhasil memodifikasi order {order_ticket}")
    return True


def manage_pending_orders_lifecycle(symbol, bias):
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return

    current_time = tick.time

    for order in orders:
        if order.magic != MAGIC_NUMBER:
            continue

        ticket = order.ticket
        order_type = order.type
        sl = order.sl

        should_cancel = False
        reason = ""

        age_seconds = current_time - order.time_setup
        if age_seconds > 900:  # 15 minutes expiry
            should_cancel = True
            reason = f"expired (age: {age_seconds}s)"

        elif order_type in [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]:
            if bias != 'BUY':
                should_cancel = True
                reason = f"bias changed from BUY to {bias}"
            elif tick.bid < sl and sl > 0:
                should_cancel = True
                reason = f"price {tick.bid} broke stop loss {sl}"

        elif order_type in [mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP]:
            if bias != 'SELL':
                should_cancel = True
                reason = f"bias changed from SELL to {bias}"
            elif tick.ask > sl and sl > 0:
                should_cancel = True
                reason = f"price {tick.ask} broke stop loss {sl}"

        if should_cancel:
            logger.info(f"[PENDING ORDER] Mengambil keputusan CANCEL untuk tiket {ticket}: {reason}")
            cancel_pending_order(ticket)


def analyze_ltf_setup(symbol, bias):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error("[LTF] Gagal mengambil tick.")
        return None

    current_bid = tick.bid
    current_ask = tick.ask
    current_price = current_ask if bias == 'BUY' else current_bid

    atr_m1 = get_atr_value(symbol, mt5.TIMEFRAME_M1)

    symbol_upper = symbol.upper()
    if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
        target_tp = TP_FROM_FIRST_ENTRY_PIPS * XAU_PIP_VALUE
        target_sl = 3.0
    else:
        target_sl = 2.0 * atr_m1 if atr_m1 > 0 else 0.003
        target_tp = 2.0 * target_sl

    ob_m5_info = detect_order_block(symbol, mt5.TIMEFRAME_M5, lookback=40)
    ob_m1_info = detect_order_block(symbol, mt5.TIMEFRAME_M1, lookback=40)

    rates_m1_5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 1, 5)
    if rates_m1_5 is not None and len(rates_m1_5) >= 5:
        df_m1_5 = pd.DataFrame(rates_m1_5)
        recent_high_m1 = float(df_m1_5['high'].max())
        recent_low_m1 = float(df_m1_5['low'].min())
    else:
        recent_high_m1 = current_price + 2 * atr_m1
        recent_low_m1 = current_price - 2 * atr_m1

    if bias == 'BUY':
        if ob_m5_info is not None:
            ob_top_m5, ob_bottom_m5, ob_type_m5 = ob_m5_info
            if ob_type_m5 == 'bullish' and current_price > ob_top_m5:
                entry = ob_top_m5
                sl = entry - target_sl
                tp = entry + target_tp
                logger.info(f"[LTF SETUP] BUY_LIMIT M5 OB terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
                return mt5.ORDER_TYPE_BUY_LIMIT, entry, sl, tp

        if ob_m1_info is not None:
            ob_top_m1, ob_bottom_m1, ob_type_m1 = ob_m1_info
            if ob_type_m1 == 'bullish' and current_price > ob_top_m1:
                entry = ob_top_m1
                sl = entry - target_sl
                tp = entry + target_tp
                logger.info(f"[LTF SETUP] BUY_LIMIT M1 OB terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
                return mt5.ORDER_TYPE_BUY_LIMIT, entry, sl, tp

        entry = recent_high_m1 + 0.1 * atr_m1
        sl = entry - target_sl
        tp = entry + target_tp
        logger.info(f"[LTF SETUP] BUY_STOP Breakout M1 terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
        return mt5.ORDER_TYPE_BUY_STOP, entry, sl, tp

    elif bias == 'SELL':
        if ob_m5_info is not None:
            ob_top_m5, ob_bottom_m5, ob_type_m5 = ob_m5_info
            if ob_type_m5 == 'bearish' and current_price < ob_bottom_m5:
                entry = ob_bottom_m5
                sl = entry + target_sl
                tp = entry - target_tp
                logger.info(f"[LTF SETUP] SELL_LIMIT M5 OB terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
                return mt5.ORDER_TYPE_SELL_LIMIT, entry, sl, tp

        if ob_m1_info is not None:
            ob_top_m1, ob_bottom_m1, ob_type_m1 = ob_m1_info
            if ob_type_m1 == 'bearish' and current_price < ob_bottom_m1:
                entry = ob_bottom_m1
                sl = entry + target_sl
                tp = entry - target_tp
                logger.info(f"[LTF SETUP] SELL_LIMIT M1 OB terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
                return mt5.ORDER_TYPE_SELL_LIMIT, entry, sl, tp

        entry = recent_low_m1 - 0.1 * atr_m1
        sl = entry + target_sl
        tp = entry - target_tp
        logger.info(f"[LTF SETUP] SELL_STOP Breakout M1 terdeteksi. Entry: {entry:.3f}, TP: {tp:.3f}")
        return mt5.ORDER_TYPE_SELL_STOP, entry, sl, tp

    return None


def execute_trade(bias, symbol, atr_value, current_price, confidence=0.5):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f'Gagal mengambil posisi dari MT5. Error: {mt5.last_error()}')
        return
    positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    logger.info(f'Status Posisi MT5 saat ini untuk {symbol}: Total={len(positions)}')

    # Jika sudah ada posisi aktif, biarkan manage_grid yang mengurus layering sampai profit
    if len(positions) > 0:
        total_pnl = sum(pos.profit + pos.swap for pos in positions)
        logger.info(f'[HOLD & LAYER MODE] Sudah ada {len(positions)} posisi aktif (Floating PnL: ${total_pnl:.2f}). Fokus pada penyelesaian Basket.')
        return

    if bias not in ('BUY', 'SELL'):
        logger.info('Bias HOLD terdeteksi. Tidak melakukan aksi transaksi baru.')
        return

    base_lot_size = get_next_lot(1, symbol)
    lot_size = base_lot_size

    is_high_confidence = confidence >= HIGH_CONFIDENCE_THRESHOLD
    if is_high_confidence:
        lot_size = round(base_lot_size * HIGH_CONFIDENCE_LOT_MULTIPLIER, 2)
        logger.info(f"[HIGH CONFIDENCE] Confidence {confidence:.4f} >= {HIGH_CONFIDENCE_THRESHOLD}. Menggunakan multiplier {HIGH_CONFIDENCE_LOT_MULTIPLIER}x: Lot {base_lot_size} -> {lot_size}")
    else:
        logger.info(f'Ukuran lot Layer 1 untuk order baru: {lot_size}')

    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        orders = []

    pending_types = (
        [mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP]
        if bias == 'BUY' else
        [mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP]
    )
    same_side_pendings = [o for o in orders if o.magic == MAGIC_NUMBER and o.type in pending_types]

    if len(same_side_pendings) >= MAX_ACTIVE_POSITIONS:
        logger.info(f'Maksimal pending {bias} ({MAX_ACTIVE_POSITIONS}) sudah aktif. HOLD.')
        return

    if not is_kill_zone_active():
        logger.info(f'[SESSION FILTER] Percobaan pending/market {bias} dibatalkan karena berada di luar Kill Zone.')
        return

    symbol_upper = symbol.upper()
    if 'XAU' in symbol_upper or 'GOLD' in symbol_upper:
        target_sl = 3.0
        target_tp = TP_FROM_FIRST_ENTRY_PIPS * XAU_PIP_VALUE
    else:
        atr_m1 = get_atr_value(symbol, mt5.TIMEFRAME_M1)
        target_sl = 2.0 * atr_m1 if atr_m1 > 0 else 0.003
        target_tp = 2.0 * target_sl

    if is_high_confidence and USE_MARKET_ORDERS_FOR_HIGH_CONFIDENCE:
        tick = mt5.symbol_info_tick(symbol)
        entry_price = (tick.ask if bias == 'BUY' else tick.bid) if tick is not None else current_price
        sl = None
        if USE_INITIAL_SL:
            sl = entry_price - target_sl if bias == 'BUY' else entry_price + target_sl
        tp = entry_price + target_tp if bias == 'BUY' else entry_price - target_tp
        logger.info(f"[HIGH CONFIDENCE MARKET {bias}] Membuka posisi market {bias} Layer 1. Price: {entry_price:.3f}, TP: {tp:.3f}")
        order_type = mt5.ORDER_TYPE_BUY if bias == 'BUY' else mt5.ORDER_TYPE_SELL
        if not open_position(symbol, order_type, lot_size, sl, tp):
            logger.error(f'Market {bias} gagal dikirim.')
    else:
        setup = analyze_ltf_setup(symbol, bias)
        if setup is not None:
            order_type, entry, sl, tp = setup
            pending_sl = sl if USE_INITIAL_SL else None
            if not place_pending_order(symbol, order_type, lot_size, entry, pending_sl, tp):
                logger.error(f'Pending {bias} gagal dikirim.')


def apply_breakeven(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return
    positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    if len(positions) != 1:
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f"[BREAKEVEN] Gagal mengambil tick untuk {symbol} saat evaluasi breakeven. Error: {mt5.last_error()}")
        return

    for pos in positions:
        ticket = pos.ticket
        open_price = pos.price_open
        tp_price = pos.tp
        sl_price = pos.sl

        if tp_price == 0.0:
            continue

        if pos.type == 0:
            current_price = tick.bid
            if sl_price >= open_price:
                continue
            gain = current_price - open_price
        elif pos.type == 1:
            current_price = tick.ask
            if sl_price > 0.0 and sl_price <= open_price:
                continue
            gain = open_price - current_price
        else:
            continue

        total_distance = abs(tp_price - open_price)

        if gain >= (total_distance * 0.5):
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info is not None else 3

            sl_target = round(open_price, digits)
            tp_target = round(tp_price, digits)
            min_distance = get_min_trade_distance(symbol) or 0.0
            if pos.type == 0:
                sl_target = min(sl_target, tick.bid - min_distance)
            else:
                sl_target = max(sl_target, tick.ask + min_distance)
            sl_target = round(sl_target, digits)
            if sl_target <= 0:
                logger.warning(f"[BREAKEVEN] SL tiket {ticket} tidak valid setelah validasi broker; update dilewati.")
                continue
            if sl_price > 0.0 and ((pos.type == 0 and sl_target <= sl_price) or (pos.type == 1 and sl_target >= sl_price)):
                logger.info(f"[BREAKEVEN] SL tiket {ticket} tidak lebih ketat dari SL saat ini; update dilewati.")
                continue

            if DRY_RUN:
                logger.info(f"[DRY RUN] [BREAKEVEN] SL untuk tiket {ticket} dipindahkan ke {sl_target} (Open: {open_price}, TP: {tp_price})")
                continue

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": sl_target,
                "tp": tp_target
            }
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"[BREAKEVEN] Gagal memindahkan SL untuk tiket {ticket}: hasil order_send None. Error: {mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[BREAKEVEN] SL untuk tiket {ticket} dipindahkan ke {sl_target}")
            else:
                logger.error(f"[BREAKEVEN] Gagal memindahkan SL untuk tiket {ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}")


def update_trailing_stop(symbol, atr_value):
    """Hanya menggeser SL jika posisi tunggal sudah berada di zona profit (di atas/bawah harga open)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return
    positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]
    if len(positions) != 1:
        return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk update trailing stop. Error: {mt5.last_error()}')
        return
    min_distance = get_min_trade_distance(symbol) or 0.0
    symbol_info = mt5.symbol_info(symbol)
    digits = symbol_info.digits if symbol_info is not None else 3
    for pos in positions:
        trail_dist = atr_value * 1.5
        if pos.type == 0:
            new_sl = round(tick.bid - trail_dist, digits)
            if min_distance > 0:
                new_sl = round(min(new_sl, tick.bid - min_distance), digits)
            # Hanya pasang/geser trailing SL jika sudah di atas harga open (zona profit)
            if new_sl > pos.price_open and new_sl > pos.sl:
                if DRY_RUN:
                    logger.info(f'[DRY RUN] Update Trailing SL untuk tiket {pos.ticket} (BUY): SL lama {pos.sl} -> SL baru {new_sl}')
                    continue
                request = {'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'symbol': symbol, 'sl': new_sl, 'tp': pos.tp}
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f'[SUCCESS] Trailing SL diperbarui untuk tiket {pos.ticket} (BUY) ke {new_sl}')
        elif pos.type == 1:
            new_sl = round(tick.ask + trail_dist, digits)
            if min_distance > 0:
                new_sl = round(max(new_sl, tick.ask + min_distance), digits)
            # Hanya pasang/geser trailing SL jika sudah di bawah harga open (zona profit)
            if new_sl < pos.price_open and (pos.sl == 0.0 or new_sl < pos.sl):
                if DRY_RUN:
                    logger.info(f'[DRY RUN] Update Trailing SL untuk tiket {pos.ticket} (SELL): SL lama {pos.sl} -> SL baru {new_sl}')
                    continue
                request = {'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'symbol': symbol, 'sl': new_sl, 'tp': pos.tp}
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f'[SUCCESS] Trailing SL diperbarui untuk tiket {pos.ticket} (SELL) ke {new_sl}')


def get_smc_bias_from_df(df):
    if df is None or len(df) == 0:
        return 'HOLD'
    last = df.iloc[-1]
    bos = int(last.get('bos_status', 0))
    mss = int(last.get('mss_status', 0))
    sweep = int(last.get('liquidity_sweep', 0))
    fvg = int(last.get('fvg_status', 0))
    score = bos + mss + sweep + fvg
    if score >= 2:
        return 'BUY'
    if score <= -2:
        return 'SELL'
    return 'HOLD'


def model_action_to_bias(action, action_count):
    if action_count == 3:
        return {0: 'HOLD', 1: 'BUY', 2: 'SELL'}.get(action, 'HOLD')
    if action_count == 2:
        return {0: 'SELL', 1: 'BUY'}.get(action, 'HOLD')
    raise ValueError(f'Action space model tidak didukung: {action_count}')


def align_observation_to_model(obs):
    try:
        expected_dim = int(model.observation_space.shape[0])
        current_dim = int(obs.shape[0])
        if current_dim != expected_dim:
            logger.error(f'[ML CONTRACT] Observasi {current_dim} tidak cocok dengan model {expected_dim}. Trade dibatalkan.')
            return None
    except Exception as e:
        logger.error(f'[ML CONTRACT] Gagal validasi bentuk observasi: {e}')
        return None
    return obs


def aggressive_bot_job():
    logger.info('=== [SCALPER] Memulai Siklus Evaluasi Live Trader ===')
    start_time = time.time()
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            logger.warning('Koneksi MT5 terputus atau tidak terinisialisasi. Mencoba login kembali...')
            initialize_mt5()

        # 1. Update micro trailing stop jika hanya ada 1 posisi yang sedang profit
        micro_trailing_stop(SYMBOL)

        # 1.5 SMART LAYERING & BASKET AVERAGING (Hold Until Profit)
        manage_grid(SYMBOL, MAGIC_NUMBER)
        if check_high_impact_news():
            logger.info('[NEWS FILTER] Entry baru ditangguhkan; manajemen layer/TP basket tetap berjalan.')
            return

        # 2. Ambil data candle M1 untuk analisis volatilitas & breakout
        rates_m1 = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, 40)
        confirmed_signal = "NEUTRAL"

        if rates_m1 is not None and len(rates_m1) >= 20:
            df_m1 = pd.DataFrame(rates_m1)
            df_m1['close'] = df_m1['close'].astype(float)
            df_m1['high'] = df_m1['high'].astype(float)
            df_m1['low'] = df_m1['low'].astype(float)

            df_m1['sma_20'] = df_m1['close'].rolling(20).mean()
            df_m1['std_20'] = df_m1['close'].rolling(20).std()
            df_m1['bb_upper'] = df_m1['sma_20'] + 1.5 * df_m1['std_20']
            df_m1['bb_lower'] = df_m1['sma_20'] - 1.5 * df_m1['std_20']

            high_low = df_m1['high'] - df_m1['low']
            high_close = (df_m1['high'] - df_m1['close'].shift()).abs()
            low_close = (df_m1['low'] - df_m1['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_m1['atr_14'] = tr.rolling(14).mean()

            last_closed = df_m1.iloc[-2]
            prev_close = last_closed['close']
            prev_high = last_closed['high']
            prev_low = last_closed['low']
            bb_upper = last_closed['bb_upper']
            bb_lower = last_closed['bb_lower']
            atr_val = last_closed['atr_14']

            candle_range = prev_high - prev_low
            is_impulsive = candle_range > (SCALPER_ATR_MULT * atr_val) if not pd.isna(atr_val) else False

            if is_impulsive:
                if prev_close > bb_upper:
                    confirmed_signal = "BUY"
                    logger.info(f"[M1 BREAKOUT] Bullish Breakout terdeteksi: Close {prev_close:.2f} > Upper BB {bb_upper:.2f} | Range {candle_range:.2f} > {SCALPER_ATR_MULT}xATR ({SCALPER_ATR_MULT * atr_val:.2f})")
                elif prev_close < bb_lower:
                    confirmed_signal = "SELL"
                    logger.info(f"[M1 BREAKOUT] Bearish Breakout terdeteksi: Close {prev_close:.2f} < Lower BB {bb_lower:.2f} | Range {candle_range:.2f} > {SCALPER_ATR_MULT}xATR ({SCALPER_ATR_MULT * atr_val:.2f})")

            if confirmed_signal == "NEUTRAL":
                logger.info(f"[SCALPER ANALYS] M1 Close: {prev_close:.2f} | BB Upper: {bb_upper:.2f} | BB Lower: {bb_lower:.2f} | Candle Range: {candle_range:.2f} | ATR Threshold: {SCALPER_ATR_MULT * (atr_val if not pd.isna(atr_val) else 0):.2f} -> Sinyal: NEUTRAL")

        # 4. Ambil Posisi Aktif & Pending Orders
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            positions = []
        scalper_positions = [pos for pos in positions if pos.magic == MAGIC_NUMBER]

        orders = mt5.orders_get(symbol=SYMBOL)
        if orders is None:
            orders = []
        scalper_orders = [ord for ord in orders if ord.magic == MAGIC_NUMBER]

        # 5. Logika Posisi Aktif: HOLD UNTIL PROFIT (Tidak Cut Loss saat minus!)
        if len(scalper_positions) > 0:
            total_floating = sum(p.profit + p.swap for p in scalper_positions)
            logger.info(f"[HOLD & LAYER MODE] Posisi aktif: {len(scalper_positions)} layer | Total Floating PnL: ${total_floating:.2f}")

            # Jika kebetulan hanya 1 posisi, sinyal berbalik arah, DAN posisi tersebut SUDAH PROFIT (>= $0.50), boleh bungkus profitnya
            if len(scalper_positions) == 1 and total_floating >= 0.50:
                pos = scalper_positions[0]
                if (pos.type == 0 and confirmed_signal == "SELL") or (pos.type == 1 and confirmed_signal == "BUY"):
                    logger.info(f"[SCALPER TAKE PROFIT ON REVERSAL] Sinyal berlawanan saat posisi profit (${total_floating:.2f}). Menutup tiket {pos.ticket} dalam keadaan Profit!")
                    close_position(pos.ticket, pos.type, SYMBOL, pos.volume, reason='REVERSAL_PROFIT')

            # Bersihkan pending order yang belum tersentuh karena sudah ada posisi utama yang dikelola manage_grid
            for ord in scalper_orders:
                logger.info(f"[SCALPER CLEANUP] Menghapus pending order tiket {ord.ticket} karena ada posisi aktif.")
                cancel_pending_order(ord.ticket)

        else:
            # 6. Logika Kelola Pending Orders ketika TIDAK ada posisi aktif (Mencari Entry Layer 1)
            lot_size = get_next_lot(1, SYMBOL)
            tick = mt5.symbol_info_tick(SYMBOL)

            if tick is not None:
                symbol_info = mt5.symbol_info(SYMBOL)
                digits = symbol_info.digits if symbol_info is not None else 3
                first_tp_usd = TP_FROM_FIRST_ENTRY_PIPS * XAU_PIP_VALUE  # Default: 12 pips = $1.20 (mudah tersentuh TP)

                if confirmed_signal == "NEUTRAL":
                    if len(scalper_orders) > 0:
                        logger.info("[SCALPER HOLD] Sinyal NEUTRAL. Membiarkan pending order tetap aktif.")

                elif confirmed_signal == "BUY":
                    buy_stops = [o for o in scalper_orders if o.type == mt5.ORDER_TYPE_BUY_STOP]
                    sell_stops = [o for o in scalper_orders if o.type == mt5.ORDER_TYPE_SELL_STOP]
                    for ord in sell_stops:
                        logger.info(f"[SCALPER CLEANUP] Hapus SELL_STOP tiket {ord.ticket} karena sinyal saat ini BUY.")
                        cancel_pending_order(ord.ticket)

                    target_price = round(tick.ask + (SCALPER_PENDING_DISTANCE_PIPS * XAU_PIP_VALUE), digits)
                    sl_price = round(target_price - SCALPER_SL_USD, digits) if (USE_INITIAL_SL and SCALPER_SL_USD > 0) else None
                    tp_price = round(target_price + first_tp_usd, digits)

                    if len(buy_stops) == 0:
                        logger.info(f"[SCALPER PENDING] Sinyal BUY terkonfirmasi. Menempatkan BUY_STOP Layer 1 di {target_price} | TP: {tp_price} | SL: {sl_price}")
                        place_pending_order(SYMBOL, mt5.ORDER_TYPE_BUY_STOP, lot_size, price=target_price, sl=sl_price, tp=tp_price)
                    else:
                        for ord in buy_stops:
                            dist_to_market = abs(ord.price_open - tick.ask)
                            if dist_to_market < PENDING_LOCK_ZONE_USD:
                                logger.info(f"[SCALPER LOCK] BUY_STOP {ord.ticket} terlalu dekat dengan market ({dist_to_market:.2f}). Dikunci.")
                                continue
                            price_diff = abs(ord.price_open - target_price)
                            if price_diff <= PENDING_ADJUST_THRESHOLD_USD:
                                logger.info(f"[SCALPER HOLD] BUY_STOP {ord.ticket} dalam toleransi ({price_diff:.2f})")
                            else:
                                logger.info(f"[SCALPER UPDATE] BUY_STOP tiket {ord.ticket} bergeser ({price_diff:.2f}). Adjusting...")
                                modify_pending_order(ord.ticket, new_price=target_price, new_sl=sl_price, new_tp=tp_price)

                elif confirmed_signal == "SELL":
                    sell_stops = [o for o in scalper_orders if o.type == mt5.ORDER_TYPE_SELL_STOP]
                    buy_stops = [o for o in scalper_orders if o.type == mt5.ORDER_TYPE_BUY_STOP]
                    for ord in buy_stops:
                        logger.info(f"[SCALPER CLEANUP] Hapus BUY_STOP tiket {ord.ticket} karena sinyal saat ini SELL.")
                        cancel_pending_order(ord.ticket)

                    target_price = round(tick.bid - (SCALPER_PENDING_DISTANCE_PIPS * XAU_PIP_VALUE), digits)
                    sl_price = round(target_price + SCALPER_SL_USD, digits) if (USE_INITIAL_SL and SCALPER_SL_USD > 0) else None
                    tp_price = round(target_price - first_tp_usd, digits)

                    if len(sell_stops) == 0:
                        logger.info(f"[SCALPER PENDING] Sinyal SELL terkonfirmasi. Menempatkan SELL_STOP Layer 1 di {target_price} | TP: {tp_price} | SL: {sl_price}")
                        place_pending_order(SYMBOL, mt5.ORDER_TYPE_SELL_STOP, lot_size, price=target_price, sl=sl_price, tp=tp_price)
                    else:
                        for ord in sell_stops:
                            dist_to_market = abs(tick.bid - ord.price_open)
                            if dist_to_market < PENDING_LOCK_ZONE_USD:
                                logger.info(f"[SCALPER LOCK] SELL_STOP {ord.ticket} terlalu dekat dengan market ({dist_to_market:.2f}). Dikunci.")
                                continue
                            price_diff = abs(ord.price_open - target_price)
                            if price_diff <= PENDING_ADJUST_THRESHOLD_USD:
                                logger.info(f"[SCALPER HOLD] SELL_STOP {ord.ticket} dalam toleransi ({price_diff:.2f})")
                            else:
                                logger.info(f"[SCALPER UPDATE] SELL_STOP tiket {ord.ticket} bergeser ({price_diff:.2f}). Adjusting...")
                                modify_pending_order(ord.ticket, new_price=target_price, new_sl=sl_price, new_tp=tp_price)

    except Exception as e:
        logger.exception(f'Error tidak terduga dalam aggressive_bot_job: {e}')
    elapsed = time.time() - start_time
    logger.info(f'=== [SCALPER] Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')


def bot_job():
    logger.info('=== Memulai Siklus Evaluasi Live Trader ===')
    start_time = time.time()
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            logger.warning('Koneksi MT5 terputus atau tidak terinisialisasi. Mencoba login kembali...')
            initialize_mt5()

        # Kelola Smart Layering & Basket TP terlebih dahulu
        manage_grid(SYMBOL, MAGIC_NUMBER)
        apply_breakeven(SYMBOL)

        state_data = get_live_state(SYMBOL)
        if state_data is None:
            return
        (legacy_obs, current_price, atr_value, df) = state_data
        expected_dim = int(model.observation_space.shape[0])
        if expected_dim == len(MODEL_MTF_FEATURES) * 10 + 8:
            obs = get_model_mtf_observation(SYMBOL)
            if obs is None:
                return
        elif expected_dim == len(LEGACY_MODEL_FEATURES) * 10:
            obs = legacy_obs
        else:
            logger.error(f'[ML CONTRACT] Dimensi model {expected_dim} tidak memiliki pipeline fitur live.')
            return
        obs = align_observation_to_model(obs)
        if obs is None:
            return
        obs_reshaped = obs.reshape(1, -1)
        probs_log = ''
        action_count = int(model.action_space.n)
        probs = np.full(action_count, 1.0 / action_count, dtype=np.float32)
        try:
            import torch
            (obs_tensor, _) = model.policy.obs_to_tensor(obs_reshaped)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            if action_count == 3:
                probs_log = f' | Probabilities - HOLD: {probs[0]:.4f}, BUY: {probs[1]:.4f}, SELL: {probs[2]:.4f}'
            else:
                probs_log = f' | Probabilities - SELL: {probs[0]:.4f}, BUY: {probs[1]:.4f}'
        except Exception as e:
            logger.debug(f'Could not compute policy probabilities: {e}')
        update_trailing_stop(SYMBOL, atr_value)

        if not is_kill_zone_active():
            logger.info('[SESSION FILTER] Di luar London/NY Kill Zone. Menunggu volatilitas pasar...')
            elapsed = time.time() - start_time
            logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')
            return
        if check_high_impact_news():
            logger.info('[NEWS FILTER] Entry AI ditangguhkan; manajemen layer/proteksi tetap berjalan.')
            return

        adx_value = df['ADX_14'].iloc[-1]
        final_bias = 'HOLD'

        if adx_value >= 20:
            action, _states = model.predict(obs_reshaped, deterministic=True)
            action_val = int(action[0])
            mode_str = "AI Trend Mode"
            ppo_bias = model_action_to_bias(action_val, action_count)
            logger.info(f'[{mode_str}] Prediksi Model: Action={action_val} | Bias={ppo_bias}{probs_log}')

            macro_trend = check_macro_trend(SYMBOL)

            vol_col = 'real_volume' if ('real_volume' in df.columns and df['real_volume'].astype(float).sum() > 0) else 'tick_volume'
            if vol_col not in df.columns:
                vol_col = 'volume'
            current_volume = float(df[vol_col].iloc[-1])
            volume_sma_20 = float(df['volume_sma_20'].iloc[-1])

            is_volume_breakout = current_volume > (2.0 * volume_sma_20)

            if ppo_bias == 'BUY':
                if macro_trend is not None and macro_trend == 0:
                    logger.info("Sinyal BUY dibatalkan: Melawan tren makro Bearish (H1).")
                else:
                    final_bias = 'BUY'
            elif ppo_bias == 'SELL':
                if macro_trend is not None and macro_trend == 1:
                    logger.info("Sinyal SELL dibatalkan: Melawan tren makro Bullish (H1).")
                else:
                    final_bias = 'SELL'

            if is_volume_breakout:
                if ppo_bias in ('BUY', 'SELL'):
                    final_bias = ppo_bias
                    logger.info(f"[VOLUME BREAKOUT] Override aktif! Izinkan {ppo_bias}.")
        else:
            close_price = df['close'].iloc[-1]
            lower_band = df['BBL_20_2.0'].iloc[-1]
            upper_band = df['BBU_20_2.0'].iloc[-1]
            mode_str = "Rule-Based Sideways Mode"

            if close_price <= lower_band:
                final_bias = 'BUY'
            elif close_price >= upper_band:
                final_bias = 'SELL'
            else:
                final_bias = 'HOLD'

            logger.info(f'[{mode_str}] Evaluasi Aturan: Close={close_price:.3f} | BB Lower={lower_band:.3f} | BB Upper={upper_band:.3f} -> Bias={final_bias}')

        confidence = 0.5
        if final_bias == 'BUY':
            confidence = float(probs[1])
        elif final_bias == 'SELL':
            confidence = float(probs[2] if action_count == 3 else probs[0])

        manage_pending_orders_lifecycle(SYMBOL, final_bias)
        execute_trade(final_bias, SYMBOL, atr_value, current_price, confidence)

    except Exception as e:
        logger.exception(f'Error tidak terduga dalam bot_job: {e}')
    elapsed = time.time() - start_time
    logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')


def main():
    global model, SYMBOL, LOT_SIZE, MAGIC_NUMBER, DEVIATION, DRY_RUN, ACCOUNT_TYPE
    parser = argparse.ArgumentParser(description='PPO RL Live Trader for MT5 with Authentication')
    parser.add_argument('--model', type=str, default='models/ppo_gold_agresif_mtf_v1', help='Path to SB3 PPO model file')
    parser.add_argument('--symbol', type=str, default=SYMBOL, help='Trading symbol (e.g. XAUUSD, XAUUSDm)')
    parser.add_argument('--lot-size', type=float, default=0.01, help='Lot size for trade deals')
    parser.add_argument('--magic', type=int, default=202606, help='MT5 Magic Number')
    parser.add_argument('--deviation', type=int, default=20, help='MT5 Order Slippage Deviation')
    parser.add_argument('--dry-run', action='store_true', help='Run without sending real trade requests to MT5')
    parser.add_argument('--account-type', type=str, default=None, choices=['standard', 'cent', 'premium', 'zero'],
                        help='Tipe akun: standard, cent, premium, zero')
    args = parser.parse_args()
    LOT_SIZE = args.lot_size
    MAGIC_NUMBER = args.magic
    DEVIATION = args.deviation
    DRY_RUN = args.dry_run

    load_dotenv()
    if args.account_type is not None:
        ACCOUNT_TYPE = args.account_type
    else:
        ACCOUNT_TYPE = os.getenv('ACCOUNT_TYPE', 'standard').strip().lower()

    if args.symbol == 'XAUUSDb':
        if ACCOUNT_TYPE == 'cent':
            SYMBOL = 'XAUUSDc'
        elif ACCOUNT_TYPE in ['premium', 'standard', 'zero']:
            SYMBOL = 'XAUUSD'
        else:
            SYMBOL = args.symbol
    else:
        SYMBOL = args.symbol

    initialize_mt5()
    model_path = args.model
    if not model_path.endswith('.zip'):
        model_path_zip = model_path + '.zip'
    else:
        model_path_zip = model_path
        model_path = model_path[:-4]
    if not os.path.exists(model_path_zip):
        logger.error(f'File model tidak ditemukan di: {model_path_zip}')
        mt5.shutdown()
        sys.exit(1)
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        if ACCOUNT_TYPE == 'cent':
            alternatives = ['XAUUSDc', 'XAUUSDb', 'XAUUSD', 'XAUUSDm', 'GOLD']
        else:
            alternatives = ['XAUUSD', 'XAUUSDm', 'XAUUSDb', 'XAUUSDc', 'GOLD']
        found_symbol = None
        for alt in alternatives:
            info = mt5.symbol_info(alt)
            if info is not None:
                found_symbol = alt
                break
        if found_symbol is None:
            logger.error(f"Symbol '{SYMBOL}' tidak ditemukan di Market Watch.")
            mt5.shutdown()
            sys.exit(1)
        else:
            logger.info(f"Symbol '{SYMBOL}' tidak ditemukan. Menggunakan varian: '{found_symbol}'")
            SYMBOL = found_symbol
    if not mt5.symbol_select(SYMBOL, True):
        logger.warning(f"Gagal menyeleksi symbol '{SYMBOL}' di Market Watch.")
    logger.info(f"Memuat model PPO dari '{args.model}'...")
    try:
        model = PPO.load(args.model)
    except Exception as e:
        logger.error(f'Gagal memuat model PPO: {e}')
        mt5.shutdown()
        sys.exit(1)
    logger.info('Model berhasil dimuat.')
    logger.info(f'Config: Symbol={SYMBOL}, Lot Size={LOT_SIZE}, Magic={MAGIC_NUMBER}, Deviation={DEVIATION}, Dry Run={DRY_RUN}, Account Type={ACCOUNT_TYPE}')
    logger.info(f'Aggressive Config: Mode={AGGRESSIVE_MODE}, Max Positions={MAX_ACTIVE_POSITIONS}, High Confidence Threshold={HIGH_CONFIDENCE_THRESHOLD}, Lot Multiplier={HIGH_CONFIDENCE_LOT_MULTIPLIER}x, Market Orders={USE_MARKET_ORDERS_FOR_HIGH_CONFIDENCE}')
    logger.info(f'Layering Config: Max Layers={MAX_LAYERS}, Layer Distance={GRID_LAYER_DISTANCE_PIPS} pips, Basket TP=${BASKET_TP_USD}, Initial SL={USE_INITIAL_SL}')

    if SCALPER_MODE:
        logger.info(f'SCALPER MODE aktif. Menjalankan aggressive_bot_job setiap {SCALPER_INTERVAL_SECONDS} detik. Tekan Ctrl+C untuk berhenti.')
        while True:
            try:
                aggressive_bot_job()
                time.sleep(SCALPER_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                logger.info('Menerima KeyboardInterrupt. Menghentikan bot...')
                break
            except Exception as e:
                logger.error(f'Error dalam scalper execution loop: {e}')
                time.sleep(5)
    else:
        bot_job()
        schedule.every(1).minutes.do(bot_job)
        logger.info('Scheduler bot diaktifkan (M1 interval). Tekan Ctrl+C untuk berhenti.')
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info('Menerima KeyboardInterrupt. Menghentikan bot...')
                break
            except Exception as e:
                logger.error(f'Error dalam execution loop: {e}')
                time.sleep(5)
    mt5.shutdown()
    logger.info('Koneksi MT5 ditutup. Selesai.')


if __name__ == '__main__':
    main()
