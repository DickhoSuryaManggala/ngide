import sys
import os

# Add root directory to sys.path to allow importing from 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import json
import sqlite3
import pandas as pd
import numpy as np
from core import backtest
from core import sentiment_analysis
from core import risk_manager

app = Flask(__name__)

CONFIG_FILE = "data/config.json"
# Tambahkan path ke live trades
LIVE_TRADES_CSV = "data/live_trades.csv"
RISK_CONFIG = "data/risk_config.json"
DB_FILE = "data/trading_data.db"
AI_MODEL_FILE = "models/trading_ai_model.joblib"
DATA_FILE = "data/data_xauusd.csv"
BACKTEST_RESULTS = "data/backtest_results.csv"
BALANCE_HISTORY = "data/balance_history.csv"

bot_running = False

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def get_live_stats():
    """Mengambil data live trading untuk dashboard."""
    live_data = {
        "status": "OFFLINE",
        "sentiment": "NEUTRAL",
        "sentiment_score": 0.0,
        "daily_pnl": 0.0,
        "active_trades": 0,
        "circuit_breaker": "SAFE"
    }
    
    # 1. Get Sentiment
    score = sentiment_analysis.get_current_sentiment()
    live_data["sentiment_score"] = round(score, 2)
    live_data["sentiment"] = "BULLISH" if score > 0.05 else ("BEARISH" if score < -0.05 else "NEUTRAL")
    
    # 2. Get Risk Status
    if os.path.exists(RISK_CONFIG):
        with open(RISK_CONFIG, 'r') as f:
            r_conf = json.load(f)
            live_data["circuit_breaker"] = "LOCKED" if r_conf.get("circuit_breaker_active") else "SAFE"

    # 3. Get Live Trades count
    if os.path.exists(LIVE_TRADES_CSV):
        df = pd.read_csv(LIVE_TRADES_CSV)
        live_data["active_trades"] = len(df[df['status'] == 'OPEN'])
        # Simple daily PnL logic (can be expanded)
        live_data["status"] = "ONLINE" if len(df) > 0 else "OFFLINE"
        
    return live_data

def get_ai_stats():
    accuracy = 52.38  # Default or last known
    total_trades = 0
    metrics = {
        "event_driven": {"Total Return": 0, "Win Rate": 0, "Sharpe Ratio": 0, "Max Drawdown": 0},
        "vectorized": {"Total Return": 0, "Win Rate": 0, "Sharpe Ratio": 0, "Max Drawdown": 0},
        "benchmark": {"Buy & Hold Return": 0}
    }
    
    if os.path.exists(DATA_FILE) and os.path.exists(CONFIG_FILE):
        df = pd.read_csv(DATA_FILE)
        df.columns = [c.lower() for c in df.columns]
        with open(CONFIG_FILE, "r") as f: conf = json.load(f)
        
        # Calculate Benchmark
        first_p = df['close'].iloc[0]
        last_p = df['close'].iloc[-1]
        metrics["benchmark"]["Buy & Hold Return"] = (last_p - first_p) / first_p * 100
        
        # Get Event Driven (from balance history)
        if os.path.exists(BALANCE_HISTORY):
            bh_df = pd.read_csv(BALANCE_HISTORY)
            if 'balance' in bh_df.columns:
                metrics["event_driven"] = backtest.Metrics.calculate(bh_df['balance'].tolist())
        
        if os.path.exists(BACKTEST_RESULTS):
            bt_df = pd.read_csv(BACKTEST_RESULTS)
            total_trades = len(bt_df)

        # Vectorized (calculated live for dashboard)
        metrics["vectorized"] = backtest.vectorized_backtest(df, conf)

    return {"accuracy": accuracy, "total_trades": total_trades, "metrics": metrics}

def get_chart_data():
    if os.path.exists(BALANCE_HISTORY):
        df = pd.read_csv(BALANCE_HISTORY)
        df['time'] = pd.to_datetime(df['time'])
        
        # Equity Curve
        equity_curve = {
            "labels": df['time'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            "data": df['balance'].tolist()
        }
        
        # Drawdown
        cum_max = df['balance'].cummax()
        drawdown = ((df['balance'] - cum_max) / cum_max * 100).tolist()
        drawdown_data = {
            "labels": equity_curve["labels"],
            "data": drawdown
        }
        
        # Returns for Monthly/Yearly
        df['returns'] = df['balance'].pct_change().fillna(0)
        
        # Monthly Returns Heatmap data
        monthly_returns = df.set_index('time')['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_data = []
        for date, val in monthly_returns.items():
            monthly_data.append({
                "year": date.year,
                "month": date.strftime('%b'),
                "return": round(val * 100, 2)
            })
            
        # Yearly Returns
        yearly_returns = df.set_index('time')['returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
        yearly_data = {
            "labels": [str(d.year) for d in yearly_returns.index],
            "data": [round(v * 100, 2) for v in yearly_returns.values]
        }
        
        # Returns Distribution (Histogram)
        returns_dist = df['returns'][df['returns'] != 0].tolist()
        
        # Rolling Sharpe (60-day approx, assuming hourly data: 60*24)
        window = 60 * 24 
        if len(df) > window:
            rolling_returns = df['returns'].rolling(window)
            rolling_sharpe = (rolling_returns.mean() / rolling_returns.std() * np.sqrt(252 * 24)).fillna(0).tolist()
        else:
            rolling_sharpe = [0] * len(df)

        return {
            "equity": equity_curve,
            "drawdown": drawdown_data,
            "monthly": monthly_data,
            "yearly": yearly_data,
            "distribution": returns_dist,
            "rolling_sharpe": {
                "labels": equity_curve["labels"],
                "data": rolling_sharpe
            }
        }
    return {}

@app.route('/')
def index():
    config = load_config()
    conn = get_db_connection()
    # Konversi sqlite3.Row ke dictionary agar JSON serializable
    rows = conn.execute('SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 10').fetchall()
    transactions = [dict(row) for row in rows]
    conn.close()
    
    ai_stats = get_ai_stats()
    chart_data = get_chart_data()
    live_stats = get_live_stats()
    
    return render_template('index.html', 
                           config=config, 
                           transactions=transactions, 
                           ai_stats=ai_stats,
                           chart_data=chart_data,
                           live_stats=live_stats)

@app.route('/settings/save', methods=['POST'])
def save_settings():
    new_config = request.json
    # Load old config to preserve numeric types
    old_config = load_config()
    for key in new_config:
        if key in old_config:
            if isinstance(old_config[key], int):
                new_config[key] = int(new_config[key])
            elif isinstance(old_config[key], float):
                new_config[key] = float(new_config[key])
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(new_config, f, indent=4)
    return jsonify({"status": "success"})

@app.route('/bot/<action>', methods=['POST'])
def bot_control(action):
    global bot_running
    if action == 'start':
        bot_running = True
        # In real case, here you'd start the trading_thread
    else:
        bot_running = False
    return jsonify({"running": bot_running})

@app.route('/api/status')
def bot_status():
    return jsonify({"running": bot_running})

if __name__ == '__main__':
    # Initialize DB if not exists
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        conn.execute('''CREATE TABLE IF NOT EXISTS transactions 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, 
                        symbol TEXT, type TEXT, price REAL, lots REAL, status TEXT)''')
        conn.close()
        
    app.run(debug=True, host='0.0.0.0', port=5000)
