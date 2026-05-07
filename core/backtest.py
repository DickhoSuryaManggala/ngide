import pandas as pd
import numpy as np
import json
import os
import joblib
import ctypes

# --- C++ HFT Engine Integration ---
HFT_LIB_AVAILABLE = False
try:
    hft_lib = ctypes.CDLL('core/hft/hft_engine.so')
    
    # Define C++ argument and return types
    hft_lib.calculate_sma_cpp.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
    ]
    hft_lib.calculate_atr_cpp.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
        ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
    ]
    HFT_LIB_AVAILABLE = True
    print("HFT Engine (C++) successfully integrated.")
except Exception as e:
    print(f"HFT Engine (C++) failed to load: {e}")

# Advanced: LSTM Deep Learning Support
LSTM_MODEL_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model
    LSTM_MODEL_AVAILABLE = True
except ImportError:
    pass

# Path Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
BACKTEST_RESULTS = os.path.join(DATA_DIR, "backtest_results.csv")
AI_MODEL_FILE = os.path.join(MODELS_DIR, "trading_ai_model.joblib")
LSTM_MODEL_FILE = os.path.join(MODELS_DIR, "trading_lstm_model.keras")
SCALER_FILE = os.path.join(MODELS_DIR, "lstm_scaler.joblib")

# Global models
ai_model = None
lstm_model = None
lstm_scaler = None

def load_ai_models():
    global ai_model, lstm_model, lstm_scaler
    if os.path.exists(AI_MODEL_FILE):
        ai_model = joblib.load(AI_MODEL_FILE)
        print(f"AI Model (Random Forest) loaded from {AI_MODEL_FILE}")
    
    if LSTM_MODEL_AVAILABLE and os.path.exists(LSTM_MODEL_FILE) and os.path.exists(SCALER_FILE):
        lstm_model = load_model(LSTM_MODEL_FILE)
        lstm_scaler = joblib.load(SCALER_FILE)
        print(f"LSTM Deep Learning Model loaded from {LSTM_MODEL_FILE}")

load_ai_models()

def rma(series, period):
    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_indicators(df, conf):
    df = df.copy().reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # --- C++ Optimization Layer (HFT Standard) ---
    if HFT_LIB_AVAILABLE:
        size = len(df)
        high_ptr = df['high'].values.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        low_ptr = df['low'].values.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        close_ptr = df['close'].values.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        atr_result = np.zeros(size, dtype=np.float64)
        atr_ptr = atr_result.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Panggil kalkulator ATR C++ yang jauh lebih cepat dari Python
        hft_lib.calculate_atr_cpp(high_ptr, low_ptr, close_ptr, size, conf["st_period"], atr_ptr)
        df['atr'] = atr_result
        atr_rma = pd.Series(atr_result)
    else:
        # Fallback to Python RMA if C++ fails
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_rma = rma(tr, conf["st_period"])
        df['atr'] = atr_rma
    
    # Alligator (Tetap Python untuk fleksibilitas, tapi menggunakan RMA)
    df['jaw'] = rma(df['hl2'], conf["jaw_period"]).shift(conf["jaw_offset"])
    df['teeth'] = rma(df['hl2'], conf["teeth_period"]).shift(conf["teeth_offset"])
    df['lips'] = rma(df['hl2'], conf["lips_period"]).shift(conf["lips_offset"])
    
    df['ali_bull'] = (df['lips'] > df['teeth']) & (df['teeth'] > df['jaw'])
    df['ali_bear'] = (df['lips'] < df['teeth']) & (df['teeth'] < df['jaw'])

    upper_band = (df['hl2'] + (conf["st_factor"] * atr_rma)).values.copy()
    lower_band = (df['hl2'] - (conf["st_factor"] * atr_rma)).values.copy()
    close_prices = df['close'].values
    
    direction = np.ones(len(df))
    for i in range(1, len(df)):
        if close_prices[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_prices[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
    df['st_up'] = direction == 1
    
    # Multi-Timeframe Trend Filter (Higher TF approximation)
    df['trend_ma'] = df['close'].rolling(window=200).mean()
    df['trend_filter'] = df['close'] > df['trend_ma']
    
    # Market Sessions Filter (GMT/UTC)
    # London: 08-16, NY: 13-21
    df['hour'] = df['time'].dt.hour
    df['is_active_session'] = (df['hour'] >= 7) & (df['hour'] <= 20)
    
    # Richer Features for AI (Institutional Grade)
    df['rsi'] = 100 - (100 / (1 + df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                             df['close'].diff().apply(lambda x: -x if x < 0 else 0).rolling(14).mean()))
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # New Institutional Features
    df['volatility'] = df['atr'] / df['close']
    df['returns_1h'] = df['close'].pct_change(12)
    df['dist_from_jaw'] = (df['close'] - df['jaw']) / df['close']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Institutional Volume Profile (Price at Volume)
    # Mencari area di mana volume transaksi paling tinggi
    df['vol_price_std'] = df['close'].rolling(window=20).std()
    df['high_vol_area'] = (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5).astype(int)
    
    # --- Market Regime Detection (Institutional Standard) ---
    # Mengidentifikasi apakah pasar sedang Trending atau Ranging
    df['adx'] = calculate_adx(df) # Fungsi pembantu baru
    df['market_regime'] = np.where(df['adx'] > 25, "TRENDING", "RANGING")
    
    return df.dropna()

def calculate_adx(df, period=14):
    """Kalkulasi Average Directional Index (ADX) sederhana."""
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    tr = pd.concat([df['high'] - df['low'], 
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx

def infer_periods_per_year_from_timestamps(timestamps, trading_days_per_year=252, fallback_periods_per_year=252*24):
    try:
        ts = pd.to_datetime(pd.Series(timestamps), errors="coerce").dropna()
        if len(ts) < 3:
            return int(fallback_periods_per_year)
        deltas = ts.diff().dropna().dt.total_seconds()
        deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
        if deltas.empty:
            return int(fallback_periods_per_year)
        median_seconds = float(deltas.median())
        if not np.isfinite(median_seconds) or median_seconds <= 0:
            return int(fallback_periods_per_year)
        periods_per_day = 86400.0 / median_seconds
        if not np.isfinite(periods_per_day) or periods_per_day <= 0:
            return int(fallback_periods_per_year)
        return int(max(1, round(float(trading_days_per_year) * periods_per_day)))
    except Exception:
        return int(fallback_periods_per_year)

class Metrics:
    @staticmethod
    def calculate(balance_history, initial_balance=10000, trades=None, periods_per_year=252*24):
        balance_series = pd.to_numeric(pd.Series(balance_history), errors="coerce").dropna()
        returns = balance_series.pct_change().dropna()

        win_rate = 0.0
        profits = None
        if trades is not None:
            try:
                if isinstance(trades, pd.DataFrame):
                    if "profit" in trades.columns:
                        profits = pd.to_numeric(trades["profit"], errors="coerce").dropna().astype(float)
                elif isinstance(trades, (list, tuple)):
                    if len(trades) > 0 and isinstance(trades[0], dict):
                        profits = pd.to_numeric([t.get("profit") for t in trades], errors="coerce").dropna().astype(float)
                    else:
                        profits = pd.to_numeric(list(trades), errors="coerce").dropna().astype(float)
            except Exception:
                profits = None

        if profits is not None and len(profits) > 0:
            win_rate = float((profits > 0).mean())
        else:
            nonzero_returns = returns[returns != 0]
            if len(nonzero_returns) > 0:
                win_rate = float((nonzero_returns > 0).mean())
            
        # Institutional Metrics
        sharpe = 0
        sortino = 0
        calmar = 0

        ann = 0.0
        try:
            ann = float(np.sqrt(float(periods_per_year)))
        except Exception:
            ann = float(np.sqrt(252 * 24))
        
        ret_std = returns.std(ddof=0)
        if np.isfinite(ret_std) and ret_std > 0:
            # Annualized Sharpe (assuming hourly data, 252 days * 24 hours)
            sharpe = (returns.mean() / ret_std) * ann
            
            # Sortino Ratio (only considers downside deviation)
            downside = np.minimum(returns.values, 0.0)
            downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) > 0 else 0.0
            if np.isfinite(downside_dev) and downside_dev > 0:
                sortino = (returns.mean() / downside_dev) * ann
            
        # Drawdown
        cum_max = balance_series.cummax()
        drawdown = (balance_series - cum_max) / cum_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        total_return_pct = (float(balance_series.iloc[-1]) - float(initial_balance)) / float(initial_balance)
        if np.isfinite(max_drawdown):
            dd_floor = 1e-4  # 0.01% floor to avoid exploding ratios on near-zero drawdowns
            dd = max(abs(float(max_drawdown)), dd_floor)
            calmar = total_return_pct / dd
        
        return {
            "Total Return": total_return_pct * 100,
            "Win Rate": win_rate * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown": max_drawdown * 100
        }

# --- 1. METODE VECTORIZED (Fast) ---
def vectorized_backtest(df, conf):
    print("\n--- Menjalankan Vectorized Backtest ---")
    df = calculate_indicators(df, conf)
    
    # Sinyal Entry/Exit
    df['signal'] = (df['st_up'] & df['ali_bull']).astype(int)
    df['exit_signal'] = (~df['st_up'] & df['ali_bear']).astype(int)
    
    # Hitung Return sederhana (Vectorized)
    df['market_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['market_returns']
    
    cum_returns = (1 + df['strategy_returns'].fillna(0)).cumprod()
    
    metrics = Metrics.calculate(10000 * cum_returns.values)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    return metrics

# --- 2. METODE EVENT-DRIVEN (Detailed with Advanced Features) ---
def event_driven_backtest(df, conf, use_ai=True, ai_type='rf', symbol=None):
    print(f"\n--- Menjalankan Advanced Event-Driven Backtest (AI: {use_ai}, Type: {ai_type}) ---")
    df = calculate_indicators(df, conf)
    
    initial_balance = 10000
    balance = initial_balance
    position = None
    entry_price = 0
    sl = 0
    tp = 0
    risk_per_trade = 0.01 # 1% risk
    balance_history = [{'time': df.iloc[0]['time'], 'balance': balance}]
    trades = []
    
    lookback = 60 # For LSTM
    sym = symbol if symbol is not None else str(conf.get("symbol", "")).split(",")[0].strip()
    symbol_contract_size = conf.get("symbol_contract_size", {}) if isinstance(conf.get("symbol_contract_size", {}), dict) else {}
    contract_size = float(symbol_contract_size.get(sym, 100000.0))

    for i in range(max(1, lookback), len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Current time check for session filter
        is_active = row['is_active_session']
        
        if position is None:
            # 3. Multi-Timeframe Filter & 6. Session Filter
            desired_side = None
            if prev_row['st_up'] and prev_row['ali_bull'] and prev_row['trend_filter'] and is_active:
                desired_side = "BUY"
            elif (not prev_row['st_up']) and prev_row['ali_bear'] and (not prev_row['trend_filter']) and is_active:
                desired_side = "SELL"

            if desired_side is not None:
                should_trade = True
                if use_ai:
                    if ai_type == 'rf' and ai_model is not None:
                        features = pd.DataFrame([[
                            row['close'], row['rsi'], row['body_size'],
                            row['upper_wick'], row['lower_wick'], row['volatility'],
                            row['returns_1h'], row['dist_from_jaw'], row['hour_sin'], row['hour_cos']
                        ]], columns=['close', 'rsi', 'body_size', 'upper_wick', 'lower_wick',
                                     'volatility', 'returns_1h', 'dist_from_jaw', 'hour_sin', 'hour_cos'])
                        prediction = int(ai_model.predict(features)[0])
                        if desired_side == "BUY" and prediction == 0:
                            should_trade = False
                        elif desired_side == "SELL" and prediction == 1:
                            should_trade = False

                    elif ai_type == 'lstm' and lstm_model is not None:
                        feature_cols = ['hl2', 'rsi', 'atr', 'volatility', 'dist_from_jaw', 'hour_sin', 'hour_cos']
                        seq_df = df.iloc[i-lookback:i][feature_cols]
                        scaled_seq = lstm_scaler.transform(seq_df.values)
                        scaled_seq = np.expand_dims(scaled_seq, axis=0)

                        pred_next_hl2_scaled = float(lstm_model.predict(scaled_seq, verbose=0)[0][0])
                        last_vec = scaled_seq[0, -1, :].astype(float)
                        pred_vec = last_vec.copy()
                        pred_vec[0] = pred_next_hl2_scaled

                        last_hl2 = float(lstm_scaler.inverse_transform([last_vec])[0][0])
                        pred_hl2 = float(lstm_scaler.inverse_transform([pred_vec])[0][0])
                        delta_price = pred_hl2 - last_hl2

                        atr = float(row['atr'])
                        min_atr_ratio = float(conf.get("lstm_min_atr_ratio", 0.0))
                        min_move = max(0.0, atr * min_atr_ratio)
                        if not np.isfinite(delta_price):
                            should_trade = False
                        elif desired_side == "BUY" and delta_price <= min_move:
                            should_trade = False
                        elif desired_side == "SELL" and delta_price >= -min_move:
                            should_trade = False

                if should_trade:
                    atr = float(row['atr'])
                    sl_dist = atr * 2
                    tp_dist = atr * 4

                    temp_risk_amount = balance * risk_per_trade
                    denom = float(sl_dist) * float(contract_size)
                    temp_lot_size = (temp_risk_amount / denom) if denom > 0 else 0.0
                    slippage_pips = (temp_lot_size * 0.2) / 10.0
                    spread_price_map = conf.get("symbol_spread_price", {}) if isinstance(conf.get("symbol_spread_price", {}), dict) else {}
                    spread = spread_price_map.get(sym, None)
                    if spread is None:
                        spread_atr_ratio = float(conf.get("spread_atr_ratio", 0.0))
                        spread = float(max(0.0, atr * spread_atr_ratio))
                    spread = float(max(0.0, float(spread)))

                    risk_amount = balance * risk_per_trade
                    trade_lots = (risk_amount / denom) if denom > 0 else 0.0

                    entry_time = row['time']
                    if desired_side == "BUY":
                        position = 'BUY'
                        entry_price = float(row['open']) + (spread / 2.0) + slippage_pips
                        sl = entry_price - sl_dist
                        tp = entry_price + tp_dist
                    else:
                        position = 'SELL'
                        entry_price = float(row['open']) - (spread / 2.0) - slippage_pips
                        sl = entry_price + sl_dist
                        tp = entry_price - tp_dist

                    trades.append({
                        'type': position,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'lots': trade_lots,
                        'contract_size': contract_size,
                        'spread': spread
                    })
        
        elif position in ('BUY', 'SELL'):
            exit_reason = None
            exit_price = float(row['open'])
            
            if position == "BUY":
                if float(row['low']) <= sl:
                    exit_price = float(sl)
                    exit_reason = "STOP_LOSS"
                elif float(row['high']) >= tp:
                    exit_price = float(tp)
                    exit_reason = "TAKE_PROFIT"
                elif not (prev_row['st_up'] and prev_row['ali_bull']):
                    exit_reason = "SIGNAL_EXIT"
            else:
                if float(row['high']) >= sl:
                    exit_price = float(sl)
                    exit_reason = "STOP_LOSS"
                elif float(row['low']) <= tp:
                    exit_price = float(tp)
                    exit_reason = "TAKE_PROFIT"
                elif not ((not prev_row['st_up']) and prev_row['ali_bear']):
                    exit_reason = "SIGNAL_EXIT"
            
            if exit_reason:
                trade = trades[-1]
                spread = float(trade.get('spread', 0.0))
                cs = float(trade.get("contract_size", contract_size))
                if position == "BUY":
                    exit_price = float(exit_price) - (spread / 2.0)
                    profit = (exit_price - entry_price) * (float(trade['lots']) * cs)
                else:
                    exit_price = float(exit_price) + (spread / 2.0)
                    profit = (entry_price - exit_price) * (float(trade['lots']) * cs)
                commission_per_lot = float(conf.get("commission_per_lot", 0.0))
                commission = commission_per_lot * float(trade['lots']) * 2.0
                profit -= commission
                balance += profit
                position = None
                trade.update({
                    'profit': profit, 
                    'exit_time': row['time'], 
                    'exit_price': exit_price,
                    'reason': exit_reason,
                    'commission': commission
                })
        
        balance_history.append({'time': row['time'], 'balance': balance})

    trading_days_per_year = int(conf.get("trading_days_per_year", 252))
    periods_per_year = infer_periods_per_year_from_timestamps(df["time"], trading_days_per_year=trading_days_per_year)
    metrics = Metrics.calculate(
        [b['balance'] for b in balance_history],
        initial_balance=initial_balance,
        trades=pd.DataFrame(trades),
        periods_per_year=periods_per_year
    )
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    # Simpan hasil
    pd.DataFrame(trades).to_csv(BACKTEST_RESULTS, index=False)
    pd.DataFrame(balance_history).to_csv(os.path.join(DATA_DIR, "balance_history.csv"), index=False)
    return metrics, balance_history

# --- 3. METODE VALIDASI (Train/Test Split) ---
def run_validation(df, conf):
    print("\n--- Menjalankan Validasi (Train/Test Split 70/30) ---")
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print("Training Period Performance:")
    event_driven_backtest(train_df, conf, use_ai=False)
    
    print("\nTesting Period Performance (Out-of-Sample):")
    
    print("\n--- Model: Random Forest (RF) ---")
    event_driven_backtest(test_df, conf, use_ai=True, ai_type='rf')
    
    if LSTM_MODEL_AVAILABLE and lstm_model is not None:
        print("\n--- Model: Deep Learning (LSTM) ---")
        event_driven_backtest(test_df, conf, use_ai=True, ai_type='lstm')

# --- 4. METRIK TOLOK UKUR (Benchmark) ---
def compare_to_benchmark(df, initial_balance=10000):
    print("\n--- Tolok Ukur: Buy & Hold (Benchmark) ---")
    first_price = df['close'].iloc[0]
    last_price = df['close'].iloc[-1]
    
    buy_hold_return = (last_price - first_price) / first_price * 100
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Final Balance Buy & Hold: {initial_balance * (1 + buy_hold_return/100):.2f}")

if __name__ == "__main__":
    data_file = os.path.join(DATA_DIR, "data_xauusd.csv")
    if not os.path.exists(data_file):
        data_file = os.path.join(DATA_DIR, "data.csv")
    
    if os.path.exists(data_file):
        with open(CONFIG_FILE, 'r') as f:
            conf = json.load(f)
            
        df = pd.read_csv(data_file)
        df.columns = [c.lower() for c in df.columns]
        
        # 1. Benchmark
        compare_to_benchmark(df)
        
        # 2. Vectorized (Fast Analysis)
        vectorized_backtest(df, conf)
        
        # 3. Event-Driven (Realistic Simulation)
        metrics, history = event_driven_backtest(df, conf)
        
        # 4. Validation (Train/Test)
        run_validation(df, conf)
        
    else:
        print("Error: Tidak ditemukan file data.")
