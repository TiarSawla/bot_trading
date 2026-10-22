import os
import sys
import argparse
import time
import math
from datetime import datetime
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('live_trader.log', encoding='utf-8')])
logger = logging.getLogger('LiveTrader')
model = None
SYMBOL = 'XAUUSDc'
LOT_SIZE = 0.3
MAGIC_NUMBER = 202606
DEVIATION = 20
DRY_RUN = False

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

def get_dynamic_lot(symbol):
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    margin_free = account_info.margin_free
    step_size = 1500000.0
    base_lot = 0.01
    max_lot_limit = 0.5
    steps = math.floor(margin_free / step_size)
    calculated_lot = steps * base_lot
    final_lot = max(0.01, min(calculated_lot, max_lot_limit))
    return round(final_lot, 2)

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
    bb = df.ta.bbands(length=20, std=2)
    kc = df.ta.kc(length=20, scalar=1.5)
    if bb is not None and kc is not None:
        bbu_col = [c for c in bb.columns if 'BBU' in c][0]
        bbl_col = [c for c in bb.columns if 'BBL' in c][0]
        kcu_col = [c for c in kc.columns if 'KCU' in c][0]
        kcl_col = [c for c in kc.columns if 'KCL' in c][0]
        df['is_squeeze'] = ((bb[bbu_col] < kc[kcu_col]) & (bb[bbl_col] > kc[kcl_col])).astype(int)
    else:
        logger.warning('BBands atau KC gagal dikalkulasi. Set is_squeeze ke 0.')
        df['is_squeeze'] = 0
    df['fvg_status'] = 0
    df.loc[df['low'] > df['high'].shift(2), 'fvg_status'] = 1
    df.loc[df['high'] < df['low'].shift(2), 'fvg_status'] = -1
    df = add_smc_features(df)
    return df

def add_smc_features(df):
    """
    Menghitung Smart Money Concepts (SMC) features:
    - Swing High/Low, BOS, MSS, FVG Strength, Liquidity Distance, Liquidity Sweep
    """
    atr = df['atr_14'].copy() if 'atr_14' in df.columns else (df['high'] - df['low']).rolling(14).mean()
    atr = atr.replace(0, 0.01).fillna(0.01)

    # --- 1. Swing High / Swing Low (lookback 3) ---
    lb = 3
    df['swing_high'] = 0
    df['swing_low'] = 0
    n = len(df)
    for i in range(lb, n - lb):
        if df['high'].iloc[i] == max(df['high'].iloc[i - lb:i + lb + 1]):
            df.at[df.index[i], 'swing_high'] = 1
        if df['low'].iloc[i] == min(df['low'].iloc[i - lb:i + lb + 1]):
            df.at[df.index[i], 'swing_low'] = 1

    # --- 2. BOS (Break of Structure) ---
    df['bos_status'] = 0
    sh_idx = df.index[df['swing_high'] == 1].tolist()
    sl_idx = df.index[df['swing_low'] == 1].tolist()

    for i in range(2, n):
        if df['bos_status'].iloc[i] == 0:
            for idx in reversed(sh_idx):
                if idx < df.index[i] and df['high'].iloc[i] > df.loc[idx, 'high']:
                    df.at[df.index[i], 'bos_status'] = 1
                    break
        if df['bos_status'].iloc[i] == 0:
            for idx in reversed(sl_idx):
                if idx < df.index[i] and df['low'].iloc[i] < df.loc[idx, 'low']:
                    df.at[df.index[i], 'bos_status'] = -1
                    break

    # --- 3. MSS (Market Structure Shift) ---
    df['mss_status'] = 0
    for i in range(5, n):
        h1, h2, h3 = df['high'].iloc[i - 2], df['high'].iloc[i - 1], df['high'].iloc[i]
        l1, l2, l3 = df['low'].iloc[i - 2], df['low'].iloc[i - 1], df['low'].iloc[i]
        if h3 > h2 and l3 > l2 and h2 > h1:
            df.at[df.index[i], 'mss_status'] = 1
        elif h3 < h2 and l3 < l2 and h2 < h1:
            df.at[df.index[i], 'mss_status'] = -1

    # --- 4. FVG Strength (gap size / ATR) ---
    df['fvg_strength'] = 0.0
    for i in range(2, n):
        atr_i = atr.iloc[i]
        if df['low'].iloc[i] > df['high'].iloc[i - 2]:
            gap = df['low'].iloc[i] - df['high'].iloc[i - 2]
            df.at[df.index[i], 'fvg_strength'] = min(gap / atr_i, 3.0)
        elif df['high'].iloc[i] < df['low'].iloc[i - 2]:
            gap = df['low'].iloc[i - 2] - df['high'].iloc[i]
            df.at[df.index[i], 'fvg_strength'] = min(gap / atr_i, 3.0)

    # --- 5. Liquidity Distance (nearest swing zone / ATR) ---
    df['liquidity_distance'] = 3.0
    for i in range(1, n):
        price = df['close'].iloc[i]
        atr_i = atr.iloc[i]
        d_high = [((df.loc[idx, 'high'] - price) / atr_i) for idx in reversed(sh_idx) if idx <= df.index[i] and df.loc[idx, 'high'] > price]
        d_low = [((price - df.loc[idx, 'low']) / atr_i) for idx in reversed(sl_idx) if idx <= df.index[i] and df.loc[idx, 'low'] < price]
        cands = []
        if d_high:
            cands.append(d_high[0])
        if d_low:
            cands.append(d_low[0])
        if cands:
            df.at[df.index[i], 'liquidity_distance'] = min(min(cands), 5.0)

    # --- 6. Liquidity Sweep (spike reversal) ---
    df['liquidity_sweep'] = 0
    for i in range(2, n):
        if df['low'].iloc[i] < df['low'].iloc[i - 1] and df['close'].iloc[i] > df['close'].iloc[i - 1]:
            df.at[df.index[i], 'liquidity_sweep'] = 1
        elif df['high'].iloc[i] > df['high'].iloc[i - 1] and df['close'].iloc[i] < df['close'].iloc[i - 1]:
            df.at[df.index[i], 'liquidity_sweep'] = -1

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
    logger.info(f"Is Squeeze: {last_row['is_squeeze']} | FVG Status: {last_row['fvg_status']}")
    logger.info(f"Spread: {last_row['spread']} | Volume: {last_row['volume']}")
    logger.info('---------------------------------------------')
    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'volume', 'vwap', 'vwap_dist_pct', 'ema_50', 'ema_50_dist_pct', 'atr_14', 'is_squeeze', 'fvg_status', 'fvg_strength', 'bos_status', 'mss_status', 'liquidity_distance', 'liquidity_sweep']
    try:
        features_subset = last_10_rows[feature_cols]
    except KeyError as e:
        logger.error(f'Kolom fitur tidak lengkap: {e}')
        return None
    obs = features_subset.to_numpy(dtype=np.float32).flatten()
    return (obs, current_price, atr_value, df)

def get_filling_mode(symbol):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return mt5.ORDER_FILLING_FOK
    filling = symbol_info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_RETURN

def close_position(position_ticket, position_type, symbol, volume):
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
    close_request = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': symbol, 'volume': volume, 'type': order_type, 'position': position_ticket, 'price': price, 'deviation': 20, 'magic': 100, 'comment': 'Reversal Close', 'type_time': mt5.ORDER_TIME_GTC, 'type_filling': mt5.ORDER_FILLING_IOC}
    if DRY_RUN:
        logger.info(f'[DRY RUN] CLOSE order untuk tiket: {position_ticket}, volume: {volume} di harga {price}')
        return True
    result = mt5.order_send(close_request)
    if result is None:
        logger.error(f'Gagal menutup tiket {position_ticket}: hasil order_send None. Error: {mt5.last_error()}')
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f'Gagal menutup tiket {position_ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}')
        return False
    logger.info(f'[SUCCESS] Berhasil menutup tiket {position_ticket} (Vol: {volume}) di harga {result.price}')
    return True

def open_position(symbol, order_type, volume, sl=None, tp=None):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk open. Error: {mt5.last_error()}')
        return False
    if order_type == mt5.ORDER_TYPE_BUY:
        price = tick.ask
        type_str = 'BUY'
    elif order_type == mt5.ORDER_TYPE_SELL:
        price = tick.bid
        type_str = 'SELL'
    else:
        logger.error(f'Tipe order tidak didukung: {order_type}')
        return False
    filling_mode = get_filling_mode(symbol)
    request = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': symbol, 'volume': volume, 'type': order_type, 'price': price, 'deviation': DEVIATION, 'magic': MAGIC_NUMBER, 'comment': f'Open {type_str} position via RL Bot', 'type_time': mt5.ORDER_TIME_GTC, 'type_filling': filling_mode}
    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp
    if DRY_RUN:
        logger.info(f'[DRY RUN] OPEN {type_str} order dengan volume: {volume} di harga {price}')
        return True
    result = mt5.order_send(request)
    if result is None:
        logger.error(f'Gagal membuka posisi {type_str}: hasil order_send None. Error: {mt5.last_error()}')
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f'Gagal membuka posisi {type_str}. Retcode: {result.retcode}, Deskripsi: {result.comment}')
        if result.retcode in [10014, 10013]:
            logger.info('Mencoba fallback open dengan ORDER_FILLING_RETURN...')
            request['type_filling'] = mt5.ORDER_FILLING_RETURN
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f'[SUCCESS] Berhasil membuka posisi {type_str} dengan fallback di harga {result.price}')
                return True
        return False
    logger.info(f'[SUCCESS] Berhasil membuka posisi {type_str} (Ticket: {result.order}, Vol: {volume}) di harga {result.price}')
    return True

def execute_trade(action, symbol, atr_value, current_price, df=None):
    # --- SMC Alignment Check ---
    if df is not None and len(df) > 0:
        last = df.iloc[-1]
        bos = last.get('bos_status', 0)
        mss = last.get('mss_status', 0)
        sweep = last.get('liquidity_sweep', 0)
        is_buy = action == 1
        
        # SMC alignment: trade must align with BOS/MSS
        if (is_buy and (bos < 0 or mss < 0)) or (not is_buy and (bos > 0 or mss > 0)):
            logger.info(f'[SMC REJECT] Trade dibatalkan (BOS={bos}, MSS={mss}).')
            return
        logger.info(f"[SMC ALIGNMENT] Trade diterima (BOS={bos}, MSS={mss}, Sweep={sweep})")

    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f'Gagal mengambil posisi dari MT5. Error: {mt5.last_error()}')
        return
    logger.info(f'Status Posisi MT5 saat ini untuk {symbol}: Total={len(positions)}')
    sl_mult = 1.5
    tp_mult = 3.0
    lot_size = get_dynamic_lot(symbol)
    logger.info(f'Ukuran lot dinamis terhitung untuk order baru: {lot_size}')
    if action == 1:
        opposite_closed = False
        for pos in positions:
            if pos.type == 1:
                logger.info(f'Aksi BUY terdeteksi. Menutup posisi SELL tiket {pos.ticket}...')
                close_position(pos.ticket, pos.type, symbol, pos.volume)
                time.sleep(1)
                opposite_closed = True
        if opposite_closed:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                positions = []
        buy_positions = [pos for pos in positions if pos.type == 0]
        if len(buy_positions) > 0:
            logger.info('Posisi BUY sudah aktif. HOLD.')
        else:
            logger.info('Belum ada posisi BUY. Membuka posisi BUY...')
            sl_price = round(current_price - atr_value * sl_mult, 3)
            tp_price = round(current_price + atr_value * tp_mult, 3)
            open_position(symbol, mt5.ORDER_TYPE_BUY, lot_size, sl=sl_price, tp=tp_price)
    elif action == 0:
        opposite_closed = False
        for pos in positions:
            if pos.type == 0:
                logger.info(f'Aksi SELL terdeteksi. Menutup posisi BUY tiket {pos.ticket}...')
                close_position(pos.ticket, pos.type, symbol, pos.volume)
                time.sleep(1)
                opposite_closed = True
        if opposite_closed:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None:
                positions = []
        sell_positions = [pos for pos in positions if pos.type == 1]
        if len(sell_positions) > 0:
            logger.info('Posisi SELL sudah aktif. HOLD.')
        else:
            logger.info('Belum ada posisi SELL. Membuka posisi SELL...')
            sl_price = round(current_price + atr_value * sl_mult, 3)
            tp_price = round(current_price - atr_value * tp_mult, 3)
            open_position(symbol, mt5.ORDER_TYPE_SELL, lot_size, sl=sl_price, tp=tp_price)
    else:
        logger.error(f'Aksi model tidak dikenal: {action}')

def align_observation_to_model(obs):
    try:
        expected_dim = int(model.observation_space.shape[0])
        current_dim = int(obs.shape[0])
        if current_dim > expected_dim:
            logger.info(f'[ML ENRICHMENT] Obs {current_dim} > model dim {expected_dim}. Menggunakan kompatibilitas model lama.')
            return obs[:expected_dim]
        if current_dim < expected_dim:
            logger.info(f'[ML ENRICHMENT] Obs {current_dim} < model dim {expected_dim}. Padding zeros.')
            return np.pad(obs, (0, expected_dim - current_dim), mode='constant')
    except Exception as e:
        logger.debug(f'Could not align observation shape: {e}')
    return obs

def update_trailing_stop(symbol, atr_value):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logger.error(f'Gagal mengambil tick untuk update trailing stop. Error: {mt5.last_error()}')
        return
    for pos in positions:
        trail_dist = atr_value * 1.5
        if pos.type == 0:
            new_sl = round(tick.bid - trail_dist, 3)
            if new_sl > pos.sl:
                if DRY_RUN:
                    logger.info(f'[DRY RUN] Update Trailing SL untuk tiket {pos.ticket} (BUY): SL lama {pos.sl} -> SL baru {new_sl}')
                    continue
                request = {'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'symbol': symbol, 'sl': new_sl, 'tp': pos.tp}
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f'[SUCCESS] Trailing SL diperbarui untuk tiket {pos.ticket} (BUY) ke {new_sl}')
                elif result is None:
                    logger.error(f'Gagal update trailing SL untuk tiket {pos.ticket}: hasil order_send None. Error: {mt5.last_error()}')
                else:
                    logger.error(f'Gagal update trailing SL untuk tiket {pos.ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}')
        elif pos.type == 1:
            new_sl = round(tick.ask + trail_dist, 3)
            if pos.sl == 0.0 or new_sl < pos.sl:
                if DRY_RUN:
                    logger.info(f'[DRY RUN] Update Trailing SL untuk tiket {pos.ticket} (SELL): SL lama {pos.sl} -> SL baru {new_sl}')
                    continue
                request = {'action': mt5.TRADE_ACTION_SLTP, 'position': pos.ticket, 'symbol': symbol, 'sl': new_sl, 'tp': pos.tp}
                result = mt5.order_send(request)
                if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f'[SUCCESS] Trailing SL diperbarui untuk tiket {pos.ticket} (SELL) ke {new_sl}')
                elif result is None:
                    logger.error(f'Gagal update trailing SL untuk tiket {pos.ticket}: hasil order_send None. Error: {mt5.last_error()}')
                else:
                    logger.error(f'Gagal update trailing SL untuk tiket {pos.ticket}. Retcode: {result.retcode}, Deskripsi: {result.comment}')

def bot_job():
    logger.info('=== Memulai Siklus Evaluasi Live Trader ===')
    start_time = time.time()
    try:
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not terminal_info.connected:
            logger.warning('Koneksi MT5 terputus atau tidak terinisialisasi. Mencoba login kembali...')
            initialize_mt5()
        state_data = get_live_state(SYMBOL)
        if state_data is None:
            return
        (obs, current_price, atr_value, df) = state_data
        obs = align_observation_to_model(obs)
        obs_reshaped = obs.reshape(1, -1)
        probs_log = ''
        try:
            import torch
            (obs_tensor, _) = model.policy.obs_to_tensor(obs_reshaped)
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]
            probs_log = f' | Probabilities - SELL: {probs[0]:.4f}, BUY: {probs[1]:.4f}'
        except Exception as e:
            logger.debug(f'Could not compute policy probabilities: {e}')
        update_trailing_stop(SYMBOL, atr_value)
        (action, _states) = model.predict(obs_reshaped, deterministic=True)
        action_val = int(action[0])
        action_name = 'BUY' if action_val == 1 else 'SELL'
        logger.info(f'Prediksi Model: Action={action_val} ({action_name}){probs_log}')
        execute_trade(action_val, SYMBOL, atr_value, current_price, df)
    except Exception as e:
        logger.exception(f'Error tidak terduga dalam bot_job: {e}')
    elapsed = time.time() - start_time
    logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')

def main():
    global model, SYMBOL, LOT_SIZE, MAGIC_NUMBER, DEVIATION, DRY_RUN
    parser = argparse.ArgumentParser(description='PPO RL Live Trader for MT5 with Authentication')
    parser.add_argument('--model', type=str, default='models/ppo_gold_m5_smc', help='Path to SB3 PPO model file')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Trading symbol (e.g. XAUUSD, XAUUSDm, XAUUSDc)')
    parser.add_argument('--lot-size', type=float, default=0.01, help='Lot size for trade deals')
    parser.add_argument('--magic', type=int, default=202606, help='MT5 Magic Number')
    parser.add_argument('--deviation', type=int, default=20, help='MT5 Order Slippage Deviation')
    parser.add_argument('--dry-run', action='store_true', help='Run without sending real trade requests to MT5')
    args = parser.parse_args()
    SYMBOL = args.symbol
    LOT_SIZE = args.lot_size
    MAGIC_NUMBER = args.magic
    DEVIATION = args.deviation
    DRY_RUN = args.dry_run
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
        alternatives = [SYMBOL, SYMBOL + 'm', 'GOLD', 'XAUUSDc', 'XAUUSDm']
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
    logger.info(f'Config: Symbol={SYMBOL}, Lot Size={LOT_SIZE}, Magic={MAGIC_NUMBER}, Deviation={DEVIATION}, Dry Run={DRY_RUN}')
    bot_job()
    schedule.every(15).minutes.do(bot_job)
    logger.info('Scheduler bot diaktifkan (M15 interval). Tekan Ctrl+C untuk berhenti.')
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
