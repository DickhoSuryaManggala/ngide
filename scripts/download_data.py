try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

import pandas as pd
from datetime import datetime
import time

def download_xauusd_data(symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5, count=5000):
    if not MT5_AVAILABLE:
        print("Error: Library MetaTrader5 tidak ditemukan. Skrip ini harus dijalankan di Windows dengan MT5 terinstal.")
        return

    if not mt5.initialize():
        print("Gagal inisialisasi MT5.")
        return

    print(f"Mencoba mendownload {count} data historis untuk {symbol}...")
    
    # Ambil data dari posisi sekarang ke belakang
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    
    if rates is None or len(rates) == 0:
        print(f"Gagal mengambil data untuk {symbol}. Pastikan simbol tersebut ada di Market Watch MT5 Anda.")
        mt5.shutdown()
        return

    # Konversi ke DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Simpan ke CSV
    filename = "data_xauusd.csv"
    df.to_csv(filename, index=False)
    print(f"Berhasil! Data disimpan di: {filename}")
    
    mt5.shutdown()

if __name__ == "__main__":
    # Ganti "XAUUSD" dengan nama simbol di broker Anda jika berbeda (misal: GOLD, XAUUSD.m)
    download_xauusd_data("XAUUSD")
