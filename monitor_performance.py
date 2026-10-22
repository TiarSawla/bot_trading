import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv
import MetaTrader5 as mt5
import logging

# Setup log monitor
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('MonitorBot')

load_dotenv()

def initialize_mt5():
    account_str = os.getenv('MT5_ACCOUNT')
    password = os.getenv('MT5_PASSWORD')
    server = os.getenv('MT5_SERVER')
    if not all([account_str, password, server]):
        logger.error('Kredensial MT5 tidak lengkap di file .env!')
        sys.exit(1)
    try:
        account = int(account_str)
    except ValueError:
        logger.error(f"MT5_ACCOUNT '{account_str}' bukan integer valid.")
        sys.exit(1)
    if not mt5.initialize():
        logger.error(f'mt5.initialize() gagal. Error: {mt5.last_error()}')
        sys.exit(1)
    if not mt5.login(login=account, password=password, server=server):
        logger.error(f'Gagal login. Error: {mt5.last_error()}')
        mt5.shutdown()
        sys.exit(1)
    logger.info(f'Login berhasil ke {server}')

def resolve_symbol(requested_symbol, account_type):
    """Auto-resolve symbol berdasarkan tipe akun jika symbol default tidak ditemukan."""
    info = mt5.symbol_info(requested_symbol)
    if info is not None:
        return requested_symbol
    if account_type == 'cent':
        alternatives = ['XAUUSDc', 'XAUUSDb', 'XAUUSD', 'XAUUSDm', 'GOLD']
    else:
        alternatives = ['XAUUSD', 'XAUUSDm', 'XAUUSDb', 'XAUUSDc', 'GOLD']
    for alt in alternatives:
        info = mt5.symbol_info(alt)
        if info is not None:
            logger.info(f"Symbol '{requested_symbol}' tidak ditemukan. Menggunakan: '{alt}'")
            return alt
    logger.error(f"Tidak ada symbol yang ditemukan dari daftar: {alternatives}")
    return requested_symbol

def get_performance_summary(symbol, magic_number):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return "Gagal ambil posisi (MT5 tidak terhubung?)"

    total_profit = 0.0
    active_count = 0
    total_volume = 0.0
    for pos in positions:
        if pos.magic == magic_number:
            total_profit += pos.profit + pos.swap
            active_count += 1
            total_volume += pos.volume

    # Ambil history bulan ini
    start_date = datetime(datetime.now().year, datetime.now().month, 1)
    history = mt5.history_deals_get(start_date, datetime.now())
    realized_profit = 0.0
    total_trades = 0
    if history:
        for deal in history:
            if deal.symbol == symbol and deal.magic == magic_number and deal.entry in [mt5.DEAL_ENTRY_IN, mt5.DEAL_ENTRY_OUT]:
                if deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL]:
                    realized_profit += deal.profit + deal.swap + deal.commission
                    total_trades += 1

    # Account info
    acct = mt5.account_info()
    balance = acct.balance if acct else 0
    equity = acct.equity if acct else 0
    margin = acct.margin if acct else 0

    lines = [
        f"=== {symbol} Performance ===",
        f"Balance: ${balance:.2f} | Equity: ${equity:.2f} | Margin: ${margin:.2f}",
        f"Posisi Aktif: {active_count} | Volume Total: {total_volume:.2f}",
        f"Unrealized PnL: ${total_profit:.2f}",
        f"Realized PnL (Bulan Ini): ${realized_profit:.2f} ({total_trades} trades)",
    ]

    if active_count > 0:
        avg_price = 0.0
        weighted_vol = 0.0
        for pos in positions:
            if pos.magic == magic_number:
                avg_price += pos.price_open * pos.volume
                weighted_vol += pos.volume
        if weighted_vol > 0:
            avg_price /= weighted_vol
            lines.append(f"Avg Entry: {avg_price:.3f} | Total Vol: {weighted_vol:.2f}")

    return "\n".join(lines)

def monitor_loop(symbol, magic_number, interval=60):
    print(f"Monitoring dimulai untuk {symbol} (Interval: {interval} detik)...")
    while True:
        try:
            status = get_performance_summary(symbol, magic_number)
            logger.info(f"\n{status}")
        except Exception as e:
            logger.error(f"Error monitor: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MT5 Performance Monitor (All Account Types)')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol untuk monitor')
    parser.add_argument('--magic', type=int, default=202606, help='Magic Number')
    parser.add_argument('--interval', type=int, default=60, help='Interval monitor (detik)')
    parser.add_argument('--account-type', type=str, default=None,
                        choices=['standard', 'cent', 'premium', 'zero'],
                        help='Tipe akun (default: baca dari .env)')
    args = parser.parse_args()

    account_type = args.account_type or os.getenv('ACCOUNT_TYPE', 'standard').strip().lower()

    # Tentukan symbol default berdasarkan tipe akun
    if args.symbol == 'XAUUSD' and account_type == 'cent':
        default_symbol = 'XAUUSDc'
    else:
        default_symbol = args.symbol

    initialize_mt5()
    symbol = resolve_symbol(default_symbol, account_type)
    if not mt5.symbol_select(symbol, True):
        logger.warning(f"Gagal menyeleksi {symbol} di Market Watch.")

    logger.info(f"Config: Symbol={symbol} | Magic={args.magic} | Account Type={account_type}")
    monitor_loop(symbol, args.magic, args.interval)
