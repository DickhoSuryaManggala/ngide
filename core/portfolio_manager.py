import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import yfinance as yf

# Path Configuration
DATA_DIR = "data"

DEFAULT_SYMBOLS = ["GC=F", "XAUUSD=X", "EURUSD=X", "JPY=X", "^GSPC", "BTC-USD"]

YF_SYMBOL_MAP = {
    "XAUUSD": "XAUUSD=X",
    "XAUUSDm": "XAUUSD=X",
    "XAUUSD.i": "XAUUSD=X",
    "GOLD": "GC=F",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "BTCUSD": "BTC-USD",
    "SPX": "^GSPC",
    "US500": "^GSPC",
}

def _to_yfinance_symbols(symbols):
    out = []
    mapping_used = {}
    for s in symbols:
        raw = str(s).strip()
        if not raw:
            continue
        key = raw.upper()
        if any(token in raw for token in ("=X", "^", "-")) or raw.endswith("=F"):
            mapped = raw
        else:
            mapped = YF_SYMBOL_MAP.get(key, raw)
        if mapped not in out:
            out.append(mapped)
        if mapped != raw:
            mapping_used[raw] = mapped
    return out, mapping_used

def analyze_portfolio_correlation(symbols=None):
    """
    Menganalisis korelasi antar aset untuk manajemen portofolio institusi.
    Default mencakup XAUUSD spot (XAUUSD=X) selain Gold Futures (GC=F).
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    yf_symbols, mapping_used = _to_yfinance_symbols(symbols)

    print(f"--- Memulai Analisis Korelasi Portofolio: {yf_symbols} ---")
    if mapping_used:
        print(f"Info: Mapping symbol ke yfinance: {mapping_used}")
    
    # Download data 1 tahun terakhir
    downloaded = yf.download(yf_symbols, period="1y", progress=True)
    close = downloaded['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(name=yf_symbols[0])
    
    # Hitung return harian
    close = close.dropna(axis=1, how='all')
    returns = close.pct_change().dropna(how='all')
    
    # Hitung matriks korelasi
    corr_matrix = returns.corr()
    
    # Plotting
    os.makedirs("assets/plots", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0)
    plt.title("Institutional Asset Correlation Matrix (1 Year)")
    plt.tight_layout()
    plt.savefig("assets/plots/portfolio_correlation.png")
    plt.close()
    print(f"Plot korelasi disimpan di: assets/plots/portfolio_correlation.png")
    
    # Rekomendasi Diversifikasi
    print("\n--- Analisis Diversifikasi Institusi ---")
    for sym in corr_matrix.columns.tolist():
        highly_corr = corr_matrix[sym][(corr_matrix[sym] > 0.7) & (corr_matrix[sym] < 1.0)].index.tolist()
        if highly_corr:
            print(f"Peringatan: {sym} memiliki korelasi tinggi dengan {highly_corr}. Hindari membuka posisi besar secara bersamaan.")
        else:
            print(f"{sym}: Baik untuk diversifikasi (Korelasi rendah dengan aset lain).")
            
    return corr_matrix

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    analyze_portfolio_correlation()
