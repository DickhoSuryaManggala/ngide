import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime
import backtest

# Path Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
BACKTEST_RESULTS = os.path.join(DATA_DIR, "backtest_results.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "trading_ai_model.joblib")
DATA_XAUUSD = os.path.join(DATA_DIR, "data_xauusd.csv")

def prepare_advanced_data(data_csv, conf):
    """
    Menyiapkan data training dengan fitur institusi dan Triple Barrier Method.
    """
    if not os.path.exists(data_csv):
        return None

    df_raw = pd.read_csv(data_csv)
    df_raw.columns = [c.lower() for c in df_raw.columns]
    if 'time' in df_raw.columns:
        df_raw['time'] = pd.to_datetime(df_raw['time'], errors='coerce')
    df = backtest.calculate_indicators(df_raw, conf)
    
    # --- 1. Institutional Feature Engineering ---
    df['volatility'] = df['atr'] / df['close']
    df['returns_1h'] = df['close'].pct_change(12) # 1 hour if M5
    df['dist_from_jaw'] = (df['close'] - df['jaw']) / df['close']
    
    # Cyclic Time Features (Institutional Standard)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # --- 2. Triple Barrier Labeling (Institutional Standard) ---
    # Mencari apakah profit tercapai sebelum stop loss atau waktu habis
    lookforward = 24 # 2 jam ke depan (M5)
    labels = []
    
    close = df['close'].values
    atr = df['atr'].values
    
    for i in range(len(df) - lookforward):
        entry_p = close[i]
        sl = entry_p - (atr[i] * 1.5)
        tp = entry_p + (atr[i] * 3.0)
        
        label = 0 # Default: Loss or Time-out
        for j in range(1, lookforward + 1):
            future_p = close[i + j]
            if future_p >= tp:
                label = 1
                break
            if future_p <= sl:
                label = 0
                break
        labels.append(label)
    
    # Sinkronisasi data dengan labels
    df = df.iloc[:len(labels)].copy()
    df['target'] = labels
    
    # Features Selection
    features_cols = ['close', 'rsi', 'body_size', 'upper_wick', 'lower_wick', 
                     'volatility', 'returns_1h', 'dist_from_jaw', 'hour_sin', 'hour_cos']
    
    df = df.dropna(subset=features_cols + ['target'])
    
    X = df[features_cols]
    y = df['target']
    t = df['time'] if 'time' in df.columns else None
    
    return X, y, t

def generate_synthetic_data(X, y, samples=5000):
    """
    Langkah 1: Generative AI/Synthetic Data.
    Menggunakan SMOTE-like approach untuk mensimulasikan skenario pasar baru.
    """
    print(f"--- Generating {samples} Synthetic Market Scenarios (Institutional Standard) ---")
    if samples is None:
        samples = 0
    try:
        samples = int(samples)
    except Exception:
        samples = 0
    if samples <= 0:
        return X, y
    
    # Simple Synthetic Generation: Jittering and Interpolation
    new_X = []
    new_y = []
    
    X_values = X.values
    y_values = y.values
    
    for _ in range(samples):
        idx = np.random.randint(0, len(X))
        # Add small Gaussian noise (jittering)
        noise = np.random.normal(0, X_values[idx].std() * 0.05, X_values[idx].shape)
        new_X.append(X_values[idx] + noise)
        new_y.append(y_values[idx])
        
    synthetic_X = pd.DataFrame(new_X, columns=X.columns)
    synthetic_y = pd.Series(new_y)
    
    return pd.concat([X, synthetic_X]), pd.concat([y, synthetic_y])

def prepare_advanced_data_from_df(df, conf):
    """
    Helper untuk Online Learning: Menyiapkan data dari DataFrame yang sudah ada.
    """
    # Features Selection (Match with training)
    features_cols = ['close', 'rsi', 'body_size', 'upper_wick', 'lower_wick', 
                     'volatility', 'returns_1h', 'dist_from_jaw', 'hour_sin', 'hour_cos']
    
    # Check if target exists, if not, create it for online learning
    if 'target' not in df.columns:
        # Simple labeling for online learning (directional)
        df = df.copy()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna(subset=features_cols + ['target'])
    if len(df) < 5: return None, None
    
    X = df[features_cols]
    y = df['target']
    return X, y

def train_advanced_ai():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("--- Memulai Institutional Step: Multi-Pair Cross-Asset Training ---")
    
    # Load config
    with open(os.path.join(DATA_DIR, 'config.json'), 'r') as f:
        import json
        conf = json.load(f)
        
    # Mencari semua file data backtest (multi-pair)
    all_X = []
    all_y = []
    all_t = []
    
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("data_") and f.endswith(".csv")]
    
    if not data_files:
        print("Tidak ada file data backtest ditemukan.")
        return

    for dfile in data_files:
        pair_name = dfile.replace("data_", "").replace(".csv", "").upper()
        print(f"Processing data for: {pair_name}")
        res = prepare_advanced_data(os.path.join(DATA_DIR, dfile), conf)
        if res is None:
            continue
        X_pair, y_pair, t_pair = res
        if X_pair is not None and y_pair is not None:
            all_X.append(X_pair)
            all_y.append(y_pair)
            if t_pair is not None:
                all_t.append(t_pair)

    if not all_X:
        print("Data tidak cukup untuk training.")
        return

    # Menggabungkan data dari semua pair (Cross-Asset Intelligence)
    X = pd.concat(all_X)
    y = pd.concat(all_y)
    if all_t:
        t = pd.to_datetime(pd.concat(all_t), errors='coerce')
    else:
        t = None
    
    print(f"Total Combined Samples: {len(X)}")
    
    # Time-based split (no leakage)
    split_ratio = float(conf.get("train_split_ratio", 0.8))
    split_ratio = max(0.5, min(0.95, split_ratio))
    if t is not None and t.notna().any():
        merged = X.copy()
        merged["target"] = y.values
        merged["time"] = t.values
        merged = merged.dropna(subset=["time"])
        merged = merged.sort_values("time")

        split_idx = int(len(merged) * split_ratio)
        train_df = merged.iloc[:split_idx]
        test_df = merged.iloc[split_idx:]

        X_train = train_df.drop(columns=["target", "time"])
        y_train = train_df["target"]
        X_test = test_df.drop(columns=["target", "time"])
        y_test = test_df["target"]
        split_info = {
            "method": "time",
            "split_ratio": split_ratio,
            "split_time": str(test_df["time"].iloc[0]) if len(test_df) else None,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)
        split_info = {
            "method": "random",
            "split_ratio": split_ratio,
            "split_time": None,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    # Synthetic augmentation only on training set
    synthetic_samples = conf.get("synthetic_samples", 5000)
    X_train, y_train = generate_synthetic_data(X_train, y_train, samples=synthetic_samples)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Advanced Model Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate Feature Importance (Explainable AI - XAI)
    importances = model.feature_importances_
    feature_importance_dict = {feat: float(imp) for feat, imp in zip(X.columns, importances)}
    
    # Save training report for UI
    report = {
        "accuracy": accuracy,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": list(X.columns),
        "feature_importances": feature_importance_dict,
        "split": split_info,
        "synthetic_samples": int(synthetic_samples) if synthetic_samples is not None else 0
    }
    os.makedirs("assets/reports", exist_ok=True)
    with open("assets/reports/rf_report.json", "w") as f:
        json.dump(report, f)
        
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_advanced_ai()
