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
SYMBOL = 'XAUUSDb'
LOT_SIZE = 0.01
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

def get_dynamic_lot_idr(margin_free: float, base_usd_step: float = 1000.0) -> float:
    """
    Menghitung lot dinamis berdasarkan free margin dalam IDR.
    Mengambil kurs live USDIDR dari MT5 dan melakukan fallback jika gagal/pasar tutup.
    
    Args:
        margin_free (float): Jumlah margin bebas dalam IDR.
        base_usd_step (float): Step modal dalam USD (default $1000 per 0.01 Lot).
        
    Returns:
        float: Nilai lot akhir (min 0.01, max 0.10).
    """
    symbol_idr = "USDIDR"
    fallback_rate = 16300.0
    
    # Ambil tick terbaru untuk simbol USDIDR
    tick = mt5.symbol_info_tick(symbol_idr)
    
    if tick is not None and tick.bid > 0:
        rate = tick.bid
        logger.info(f"[Live Rate] Kurs {symbol_idr} berhasil diambil secara real-time: {rate:.2f}")
    else:
        rate = fallback_rate
        logger.info(f"[Fallback Rate] Gagal mengambil kurs {symbol_idr} live (pasar tutup/koneksi terputus). Menggunakan kurs fallback: {rate:.2f}")
        
    # Kalikan step dasar dengan kurs untuk mendapatkan dynamic step dalam IDR
    dynamic_step = base_usd_step * rate
    
    # Hitung lot berdasarkan kelipatan dynamic step
    calculated_lot = (margin_free // dynamic_step) * 0.01
    
    # Batasi lot maksimal 0.10 dan minimal 0.01
    final_lot = max(0.01, min(calculated_lot, 0.10))
    
    return round(final_lot, 2)

def get_dynamic_lot(symbol: str) -> float:
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    margin_free = account_info.margin_free
    return get_dynamic_lot_idr(margin_free)

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
    feature_cols = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume', 'volume', 'vwap', 'vwap_dist_pct', 'ema_50', 'ema_50_dist_pct', 'atr_14', 'is_squeeze', 'fvg_status']
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

    # Ambil Data Limits Broker
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error("Symbol info tidak ditemukan.")
        return False
        
    # Dapatkan stops level (jika 0, set default aman misalnya 30 poin)
    stops_level = symbol_info.trade_stops_level
    if stops_level == 0:
        stops_level = 30
        
    # Hitung jarak absolut minimum (tambah buffer 10 poin keamanan)
    min_distance = (stops_level + 10) * symbol_info.point

    sl_price = sl
    tp_price = tp

    if order_type == mt5.ORDER_TYPE_BUY:
        price = tick.ask
        type_str = 'BUY'
        
        current_bid = tick.bid
        # Validasi SL/TP untuk Transaksi BUY:
        if sl_price is not None:
            # SL Buy harus di BAWAH Bid, berjarak minimal min_distance
            if sl_price > (current_bid - min_distance):
                adjusted_sl = round(current_bid - min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan SL BUY untuk {symbol}: {sl_price} -> {adjusted_sl} (min_distance: {min_distance:.5f})")
                sl_price = adjusted_sl
                
        if tp_price is not None:
            # TP Buy harus di ATAS Bid, berjarak minimal min_distance
            if tp_price < (current_bid + min_distance):
                adjusted_tp = round(current_bid + min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan TP BUY untuk {symbol}: {tp_price} -> {adjusted_tp} (min_distance: {min_distance:.5f})")
                tp_price = adjusted_tp

    elif order_type == mt5.ORDER_TYPE_SELL:
        price = tick.bid
        type_str = 'SELL'
        
        current_ask = tick.ask
        # Validasi SL/TP untuk Transaksi SELL:
        if sl_price is not None:
            # SL Sell harus di ATAS Ask, berjarak minimal min_distance
            if sl_price < (current_ask + min_distance):
                adjusted_sl = round(current_ask + min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan SL SELL untuk {symbol}: {sl_price} -> {adjusted_sl} (min_distance: {min_distance:.5f})")
                sl_price = adjusted_sl
                
        if tp_price is not None:
            # TP Sell harus di BAWAH Ask, berjarak minimal min_distance
            if tp_price > (current_ask - min_distance):
                adjusted_tp = round(current_ask - min_distance, symbol_info.digits)
                logger.info(f"[Validation] Menyesuaikan TP SELL untuk {symbol}: {tp_price} -> {adjusted_tp} (min_distance: {min_distance:.5f})")
                tp_price = adjusted_tp
    else:
        logger.error(f'Tipe order tidak didukung: {order_type}')
        return False

    filling_mode = get_filling_mode(symbol)
    request = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': symbol, 'volume': volume, 'type': order_type, 'price': price, 'deviation': DEVIATION, 'magic': MAGIC_NUMBER, 'comment': f'Open {type_str} position via RL Bot', 'type_time': mt5.ORDER_TIME_GTC, 'type_filling': filling_mode}
    if sl_price is not None:
        request['sl'] = sl_price
    if tp_price is not None:
        request['tp'] = tp_price
    if DRY_RUN:
        logger.info(f'[DRY RUN] OPEN {type_str} order dengan volume: {volume} di harga {price} (SL: {sl_price}, TP: {tp_price})')
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
def evade_round_number(price, is_buy_sl):
    remainder = price % 10
    if remainder < 2 or remainder > 8: 
        buffer = 2.5 
        if is_buy_sl:
            return price - buffer
        else:
            return price + buffer
    return price

def is_kill_zone_active():
    """
    Memeriksa apakah waktu saat ini berada di dalam salah satu Kill Zone (WIB - UTC+7).
    Zone 1 (London): 14:00 - 17:00 WIB
    Zone 2 (New York Overlap): 19:00 - 22:30 WIB
    """
    import datetime
    now = datetime.datetime.now()
    current_time = now.time()
    
    zone1_start = datetime.time(14, 0)
    zone1_end = datetime.time(17, 0)
    
    zone2_start = datetime.time(19, 0)
    zone2_end = datetime.time(22, 30)
    
    in_zone1 = zone1_start <= current_time <= zone1_end
    in_zone2 = zone2_start <= current_time <= zone2_end
    
    return in_zone1 or in_zone2


def check_macro_trend(symbol):
    try:
        # Tarik 250 candle terakhir dari timeframe H4
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 250)
        if rates is None or len(rates) < 200:
            logger.warning(f"[MTF Filter] Gagal mengambil data H4 cukup ({len(rates) if rates is not None else 0}/200). Filter dilewati.")
            return None # Lewati filter jika data gagal ditarik
        
        df = pd.DataFrame(rates)
        df['close'] = df['close'].astype(float)
        df['ema_200'] = ta.ema(df['close'], length=200)
        
        last_close = df['close'].iloc[-1]
        last_ema = df['ema_200'].iloc[-1]
        
        if pd.isna(last_ema):
            logger.warning("[MTF Filter] Kalkulasi EMA 200 H4 menghasilkan NaN. Filter dilewati.")
            return None
            
        logger.info(f"[MTF Filter] H4 Close: {last_close:.3f} | H4 EMA 200: {last_ema:.3f}")
        # Return 1 untuk Bullish (Hanya boleh BUY), 0 untuk Bearish (Hanya boleh SELL)
        if last_close > last_ema:
            return 1
        elif last_close < last_ema:
            return 0
    except Exception as e:
        logger.error(f"[MTF Filter] Error dalam check_macro_trend: {e}")
    return None


def detect_order_block(symbol, timeframe, lookback=30):
    """
    Mendeteksi Order Block (OB) dari data OHLCV historis.
    
    Args:
        symbol (str): Simbol trading.
        timeframe: Timeframe MT5 (misal: mt5.TIMEFRAME_M15).
        lookback (int): Jumlah candle yang ditarik untuk dianalisis.
        
    Returns:
        tuple: (ob_top, ob_bottom, ob_type) jika terdeteksi, jika tidak None.
    """
    try:
        # Ambil data rates dari MT5. Menggunakan start_pos=1 untuk mengambil candle yang sudah selesai (close)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, lookback)
        if rates is None or len(rates) < 5:
            logger.warning(f"[OB DETECTION] Gagal mengambil rates dari MT5 atau data kurang ({len(rates) if rates is not None else 0}).")
            return None
        
        df = pd.DataFrame(rates)
        
        # Hitung range body tiap candle (Absolut dari Open - Close)
        df['body_size'] = (df['close'] - df['open']).abs()
        
        # Identifikasi candle paling impulsif (body terbesar)
        # Kita mencari candle impulsif dari index 1 ke atas agar selalu ada candle sebelumnya
        sub_df = df.iloc[1:]
        if sub_df.empty:
            return None
            
        imp_idx = sub_df['body_size'].idxmax()
        imp_row = df.loc[imp_idx]
        
        # Tentukan tipe OB (bullish jika candle impulsif bullish, bearish jika candle impulsif bearish)
        is_bullish_imp = imp_row['close'] > imp_row['open']
        ob_type = "bullish" if is_bullish_imp else "bearish"
        
        # Cari candle berlawanan (opposite candle) yang tepat berada sebelum gerakan impulsif tersebut dimulai
        opposite_idx = None
        if ob_type == "bullish":
            # Mencari candle bearish pertama (close < open) ke belakang dari imp_idx - 1
            for j in range(imp_idx - 1, -1, -1):
                if df.loc[j, 'close'] < df.loc[j, 'open']:
                    opposite_idx = j
                    break
        else:
            # Mencari candle bullish pertama (close > open) ke belakang dari imp_idx - 1
            for j in range(imp_idx - 1, -1, -1):
                if df.loc[j, 'close'] > df.loc[j, 'open']:
                    opposite_idx = j
                    break
                    
        if opposite_idx is None:
            # Jika tidak ditemukan candle berlawanan, gunakan candle tepat sebelum impulsif sebagai fallback
            opposite_idx = imp_idx - 1
            
        ob_top = float(df.loc[opposite_idx, 'high'])
        ob_bottom = float(df.loc[opposite_idx, 'low'])
        
        return ob_top, ob_bottom, ob_type
        
    except Exception as e:
        logger.error(f"[OB DETECTION] Error saat mendeteksi Order Block: {e}")
        return None


def execute_trade(action, symbol, atr_value, current_price):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.error(f'Gagal mengambil posisi dari MT5. Error: {mt5.last_error()}')
        return
    logger.info(f'Status Posisi MT5 saat ini untuk {symbol}: Total={len(positions)}')
    ATR_SL_MULTIPLIER = 2.0
    ATR_TP_MULTIPLIER = 4.0
    sl_distance = atr_value * ATR_SL_MULTIPLIER
    tp_distance = atr_value * ATR_TP_MULTIPLIER
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
            if not is_kill_zone_active():
                logger.info('[SESSION FILTER] Percobaan buka posisi BUY dibatalkan karena berada di luar Kill Zone.')
            else:
                logger.info('Belum ada posisi BUY. Membuka posisi BUY...')
                raw_sl = current_price - sl_distance
                tp_price = round(current_price + tp_distance, 3)
                sl_price = round(evade_round_number(raw_sl, is_buy_sl=True), 3)
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
            if not is_kill_zone_active():
                logger.info('[SESSION FILTER] Percobaan buka posisi SELL dibatalkan karena berada di luar Kill Zone.')
            else:
                logger.info('Belum ada posisi SELL. Membuka posisi SELL...')
                raw_sl = current_price + sl_distance
                tp_price = round(current_price - tp_distance, 3)
                sl_price = round(evade_round_number(raw_sl, is_buy_sl=False), 3)
                open_position(symbol, mt5.ORDER_TYPE_SELL, lot_size, sl=sl_price, tp=tp_price)
    elif action == 2:
        logger.info('Aksi HOLD terdeteksi. Tidak melakukan aksi transaksi baru.')
    else:
        logger.error(f'Aksi model tidak dikenal: {action}')

def apply_breakeven(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
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
        
        # Hindari modifikasi jika TP tidak diset
        if tp_price == 0.0:
            continue
            
        # Tentukan current price dan gain berdasarkan tipe transaksi
        if pos.type == 0: # BUY / mt5.POSITION_TYPE_BUY
            current_price = tick.bid
            # Hindari modifikasi jika SL sudah di titik Breakeven atau lebih baik
            if sl_price >= open_price:
                continue
            gain = current_price - open_price
        elif pos.type == 1: # SELL / mt5.POSITION_TYPE_SELL
            current_price = tick.ask
            # Hindari modifikasi jika SL sudah di titik Breakeven atau lebih baik (dan SL sudah di-set)
            if sl_price > 0.0 and sl_price <= open_price:
                continue
            gain = open_price - current_price
        else:
            continue
            
        total_distance = abs(tp_price - open_price)
        
        # Jika gain sudah mencapai atau melewati 50% dari jarak TP
        if gain >= (total_distance * 0.5):
            # Tentukan presisi digit symbol
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info is not None else 3
            
            # Format/round open_price dan tp_price sesuai presisi symbol
            sl_target = round(open_price, digits)
            tp_target = round(tp_price, digits)
            
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
        
        # Evaluasi pengamanan breakeven sebelum mengambil state baru
        apply_breakeven(SYMBOL)
        
        state_data = get_live_state(SYMBOL)
        if state_data is None:
            return
        (obs, current_price, atr_value, df) = state_data
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
        
        # --- SESSION FILTER (KILL ZONES) ---
        positions = mt5.positions_get(symbol=SYMBOL)
        if not is_kill_zone_active():
            if positions is None or len(positions) == 0:
                logger.info('[SESSION FILTER] Di luar London/NY Kill Zone. Menunggu volatilitas pasar...')
                elapsed = time.time() - start_time
                logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')
                return
        
        # --- REGIME SWITCHER ---
        adx_value = df['ADX_14'].iloc[-1]
        
        if adx_value >= 20:
            # --- REGIME TREN (AI V3.0) ---
            action, _states = model.predict(obs_reshaped, deterministic=True)
            action_val = int(action[0])
            mode_str = "AI Trend Mode"
            action_name = 'BUY' if action_val == 1 else 'SELL'
            logger.info(f'[{mode_str}] Prediksi Model: Action={action_val} ({action_name}){probs_log}')
            
            # --- SMC FILTER (ORDER BLOCK GATEKEEPER) ---
            ob_info = detect_order_block(SYMBOL, mt5.TIMEFRAME_M15)
            if ob_info is not None:
                ob_top, ob_bottom, ob_type = ob_info
                logger.info(f"[SMC FILTER] Order Block Terdeteksi: Tipe={ob_type}, Top={ob_top:.3f}, Bottom={ob_bottom:.3f} | Harga Saat Ini: {current_price:.3f}")
                if action_val == 1: # BUY
                    if not (ob_type == "bullish" and ob_bottom <= current_price <= ob_top):
                        logger.info("[SMC FILTER] Harga tidak berada di zona Order Block. Mengabaikan sinyal AI.")
                        action_val = 2
                elif action_val == 0: # SELL
                    if not (ob_type == "bearish" and ob_bottom <= current_price <= ob_top):
                        logger.info("[SMC FILTER] Harga tidak berada di zona Order Block. Mengabaikan sinyal AI.")
                        action_val = 2
            else:
                logger.warning("[SMC FILTER] Gagal mendeteksi Order Block. Filter dilewati.")
            
            macro_trend = check_macro_trend(SYMBOL)
            if macro_trend is not None:
                if action_val == 1 and macro_trend == 0:
                    logger.info("Sinyal BUY dibatalkan: Melawan tren makro Bearish (H4).")
                    elapsed = time.time() - start_time
                    logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')
                    return
                elif action_val == 0 and macro_trend == 1:
                    logger.info("Sinyal SELL dibatalkan: Melawan tren makro Bullish (H4).")
                    elapsed = time.time() - start_time
                    logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')
                    return
        else:
            # --- REGIME SIDEWAYS (Ping-Pong BB) ---
            close_price = df['close'].iloc[-1]
            lower_band = df['BBL_20_2.0'].iloc[-1]
            upper_band = df['BBU_20_2.0'].iloc[-1]
            mode_str = "Rule-Based Sideways Mode"
            
            if close_price <= lower_band:
                action_val = 1 # Kode untuk BUY
            elif close_price >= upper_band:
                action_val = 0 # Kode untuk SELL
            else:
                action_val = 2 # Kode untuk HOLD
            
            action_name = 'BUY' if action_val == 1 else ('SELL' if action_val == 0 else 'HOLD')
            logger.info(f'[{mode_str}] Evaluasi Aturan: Close={close_price:.3f} | BB Lower={lower_band:.3f} | BB Upper={upper_band:.3f} -> Action={action_val} ({action_name})')
            
        execute_trade(action_val, SYMBOL, atr_value, current_price)
    except Exception as e:
        logger.exception(f'Error tidak terduga dalam bot_job: {e}')
    elapsed = time.time() - start_time
    logger.info(f'=== Siklus Evaluasi Selesai (Durasi: {elapsed:.2f} detik) ===')

def main():
    global model, SYMBOL, LOT_SIZE, MAGIC_NUMBER, DEVIATION, DRY_RUN
    parser = argparse.ArgumentParser(description='PPO RL Live Trader for MT5 with Authentication')
    parser.add_argument('--model', type=str, default='models/ppo_gold_scalper_v3', help='Path to SB3 PPO model file')
    parser.add_argument('--symbol', type=str, default=SYMBOL, help='Trading symbol (e.g. XAUUSD, XAUUSDm)')
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
        alternatives = [SYMBOL, SYMBOL + 'm', SYMBOL + 'b', 'GOLD', 'XAUUSD']
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
    logger.info("Memuat model PPO dari 'models/ppo_gold_scalper_v3.zip'...")
    try:
        model = PPO.load('models/ppo_gold_scalper_v3')
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
