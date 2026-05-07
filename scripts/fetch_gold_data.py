import yfinance as yf
import pandas as pd
import os

def fetch_xauusd_data():
    print("--- Mendownload Data Open-Source XAUUSD (via Yahoo Finance) ---")
    
    # Simbol Emas di Yahoo Finance adalah 'GC=F' (Gold Futures) atau 'XAUUSD=X' (Spot)
    symbol = "GC=F" 
    
    try:
        # Download data 1 tahun terakhir dengan interval 1 jam (karena M5 terbatas di yfinance)
        # yfinance hanya menyediakan data M5 untuk 60 hari terakhir.
        print(f"Mengambil data {symbol}...")
        data = yf.download(symbol, period="2y", interval="1h")
        
        if data.empty:
            print("Gagal mengambil data. Coba lagi nanti.")
            return

        # Reset index agar 'Date' menjadi kolom
        data = data.reset_index()
        
        # Format kolom agar sesuai dengan backtest.py (time, open, high, low, close)
        data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
        
        if 'datetime' in data.columns:
            data = data.rename(columns={'datetime': 'time'})
        elif 'date' in data.columns:
            data = data.rename(columns={'date': 'time'})

        filename = "data_xauusd.csv"
        data.to_csv(filename, index=False)
        
        print(f"Berhasil! {len(data)} bar data disimpan di: {filename}")
        print("Anda sekarang bisa menjalankan: python3 backtest.py")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    fetch_xauusd_data()
