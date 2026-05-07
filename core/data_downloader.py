import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from io import StringIO

DATA_DIR = "data"
PAIRS = {
    "EURUSD=X": "data_eurusd.csv",
    "GBPUSD=X": "data_gbpusd.csv",
    "USDJPY=X": "data_usdjpy.csv",
    "BTC-USD": "data_btcusd.csv"
}

STOOQ_PAIRS = {
    "XAUUSD": "data_xauusd.csv",
    "EURUSD": "data_eurusd.csv",
    "USDJPY": "data_usdjpy.csv",
    "BTCUSD": "data_btcusd.csv",
    "^SPX": "data_spx.csv",
    "GC.F": "data_gold.csv",
}

STOOQ_BASE_URLS = ["https://stooq.com", "https://stooq.pl"]
FRED_FALLBACK = {
    "^SPX": "SP500",
    "GC.F": "GOLDAMGBD228NLBM",
}

YFINANCE_FALLBACK = {
    "^SPX": "^GSPC",
    "GC.F": "GC=F",
}

def download_institutional_data():
    print("--- Institutional Data Downloader: Multi-Pair Acquisition ---")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for symbol, filename in PAIRS.items():
        print(f"Downloading historical data for {symbol}...")
        try:
            # Download 2 years of hourly data (standard for institutional analysis)
            data = yf.download(symbol, period="2y", interval="1h")
            
            if data.empty:
                print(f"No data found for {symbol}")
                continue
                
            # Format to match system requirements
            df = data.reset_index()
            df = df[['Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']]
            df.columns = ['time', 'close', 'high', 'low', 'open', 'volume']
            
            # Save to CSV
            path = os.path.join(DATA_DIR, filename)
            df.to_csv(path, index=False)
            print(f"Successfully saved {len(df)} records to {path}")
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")

def _download_stooq_ohlc(symbol, interval="d", timeout_sec=12, start_date=None, end_date=None):
    sym = str(symbol)
    qs = f"s={quote_plus(sym.lower())}&i={quote_plus(interval)}"
    if sym.startswith("^"):
        qs = f"{qs}&c=0"
    if start_date is not None:
        qs = f"{qs}&d1={quote_plus(str(start_date))}"
    if end_date is not None:
        qs = f"{qs}&d2={quote_plus(str(end_date))}"

    last_err = None
    for base in STOOQ_BASE_URLS:
        url = f"{base}/q/d/l/?{qs}"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            if raw.strip():
                last_err = None
                break
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = e
            raw = ""
            continue
    if last_err is not None and not raw.strip():
        raise RuntimeError(f"Network error fetching {symbol} from Stooq: {last_err}")

    df_raw = pd.read_csv(StringIO(raw))
    df_raw.columns = [c.lower() for c in df_raw.columns]

    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(set(df_raw.columns)):
        raise ValueError(f"Unexpected Stooq format for {symbol}. Columns: {list(df_raw.columns)}")

    df = df_raw.rename(columns={"date": "time"})
    if "volume" not in df.columns:
        df["volume"] = 0
    df = df[["time", "close", "high", "low", "open", "volume"]]

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["close", "high", "low", "open", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0)
    df = df.dropna(subset=["time", "close", "high", "low", "open"]).sort_values("time")
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

def download_open_source_data(interval="d", overwrite=False, timeout_sec=12, lookback_days=3650):
    print("--- Open Source Data Downloader (Stooq): Multi-Asset Acquisition ---")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    try:
        end = datetime.utcnow().date()
        start = end - timedelta(days=int(lookback_days))
        d1 = start.strftime("%Y%m%d")
        d2 = end.strftime("%Y%m%d")
    except Exception:
        d1 = None
        d2 = None

    for symbol, filename in STOOQ_PAIRS.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path) and not overwrite:
            print(f"Skip (exists): {symbol} -> {path}")
            continue

        print(f"Downloading open data for {symbol} from Stooq...")
        try:
            df = _download_stooq_ohlc(
                symbol,
                interval=interval,
                timeout_sec=timeout_sec,
                start_date=d1,
                end_date=d2,
            )
            if df.empty:
                print(f"No data returned for {symbol}")
                continue
            df.to_csv(path, index=False)
            print(f"Successfully saved {len(df)} records to {path}")
        except Exception as e:
            series_id = FRED_FALLBACK.get(symbol)
            if series_id:
                print(f"Stooq failed for {symbol}. Trying FRED series {series_id}...")
                try:
                    df = _download_fred_series(series_id)
                    if df.empty:
                        print(f"No data returned for FRED series {series_id}")
                    else:
                        df.to_csv(path, index=False)
                        print(f"Successfully saved {len(df)} records to {path} (FRED fallback)")
                        continue
                except Exception as e2:
                    print(f"Error downloading {symbol} from FRED: {e2}")
            else:
                print(f"Error downloading {symbol} from Stooq: {e}")

            yf_symbol = YFINANCE_FALLBACK.get(symbol)
            if not yf_symbol:
                continue

            print(f"Trying Yahoo Finance fallback for {symbol} as {yf_symbol}...")
            try:
                df = _download_yfinance_ohlc(yf_symbol, lookback_days=lookback_days)
                if df.empty:
                    print(f"No data returned for yfinance symbol {yf_symbol}")
                    continue
                df.to_csv(path, index=False)
                print(f"Successfully saved {len(df)} records to {path} (yfinance fallback)")
            except Exception as e3:
                print(f"Error downloading {symbol} from yfinance: {e3}")

def _download_fred_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote_plus(str(series_id))}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=12) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    df_raw = pd.read_csv(StringIO(raw))
    if df_raw.shape[1] < 2:
        raise ValueError(f"Unexpected FRED format for {series_id}")
    df_raw.columns = ["time", "value"]
    df_raw["time"] = pd.to_datetime(df_raw["time"], errors="coerce")
    df_raw["value"] = pd.to_numeric(df_raw["value"], errors="coerce")
    df_raw = df_raw.dropna(subset=["time", "value"]).sort_values("time")

    df = pd.DataFrame(
        {
            "time": df_raw["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "close": df_raw["value"].astype(float),
        }
    )
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 0
    return df[["time", "close", "high", "low", "open", "volume"]]

def _download_yfinance_ohlc(symbol, lookback_days=3650):
    end = datetime.utcnow().date()
    start = end - timedelta(days=int(lookback_days))
    data = yf.download(symbol, start=str(start), end=str(end), interval="1d", progress=False)
    if data is None or data.empty:
        return pd.DataFrame(columns=["time", "close", "high", "low", "open", "volume"])

    df = data.reset_index()
    dt_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    df = df[[dt_col, "Close", "High", "Low", "Open", "Volume"]]
    df.columns = ["time", "close", "high", "low", "open", "volume"]
    df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["close", "high", "low", "open", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0)
    return df.dropna(subset=["time", "close", "high", "low", "open"])

if __name__ == "__main__":
    download_open_source_data()
