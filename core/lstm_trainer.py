import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import json
import joblib
from datetime import datetime
from core import backtest

# Path Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
LSTM_MODEL_FILE = os.path.join(MODELS_DIR, "trading_lstm_model.keras")
SCALER_FILE = os.path.join(MODELS_DIR, "lstm_scaler.joblib")
DATA_XAUUSD = os.path.join(DATA_DIR, "data_xauusd.csv")

def prepare_lstm_data(df, lookback=60):
    """
    Menyiapkan data sequence untuk LSTM dengan fitur institusi.
    """
    # --- Institutional Feature Engineering ---
    df['volatility'] = df['atr'] / df['close']
    df['dist_from_jaw'] = (df['close'] - df['jaw']) / df['close']
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Features: HL2, RSI, ATR, Volatility, DistFromJaw, TimeCyclic
    feature_cols = ['hl2', 'rsi', 'atr', 'volatility', 'dist_from_jaw', 'hour_sin', 'hour_cos']
    features = df[feature_cols].values
    
    # Scaling data ke range 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        # Target: Harga HL2 di candle berikutnya
        y.append(scaled_data[i, 0]) 
        
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    from tensorflow.keras.layers import BatchNormalization
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(units=100, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(units=50, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_ai():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("--- Memulai Institusi AI Step: Multi-Pair Cross-Asset LSTM Training ---")
    
    # Load config
    with open(os.path.join(DATA_DIR, 'config.json'), 'r') as f:
        conf = json.load(f)
        
    # Mencari semua file data backtest (multi-pair)
    all_dfs = []
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".csv")]
    
    for dfile in data_files:
        pair_name = dfile.replace("data_", "").replace(".csv", "").upper()
        print(f"Loading data for: {pair_name}")
        df_raw = pd.read_csv(os.path.join(DATA_DIR, dfile))
        df_raw.columns = [c.lower() for c in df_raw.columns]
        df_p = backtest.calculate_indicators(df_raw, conf)
        all_dfs.append(df_p)
    
    if not all_dfs:
        print("Tidak ada data untuk training LSTM.")
        return
        
    df = pd.concat(all_dfs).dropna()
    
    lookback = 60
    X, y, scaler = prepare_lstm_data(df, lookback)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training on {len(X_train)} sequences...")
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Train model (Epochs dinaikkan untuk akurasi institusi)
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, batch_size=64, epochs=30, 
                        validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # Save training report for UI
    report = {
        "loss": history.history['loss'],
        "val_loss": history.history['val_loss'],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("assets/reports/lstm_report.json", "w") as f:
        json.dump(report, f)
        
    # Save model and scaler
    model.save(LSTM_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print(f"LSTM Model saved to {LSTM_MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")

if __name__ == "__main__":
    train_lstm_ai()
