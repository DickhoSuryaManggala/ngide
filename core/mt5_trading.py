try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    try:
        from mt5linux import MetaTrader5 as mt5
        MT5_AVAILABLE = True
        print("Using mt5linux bridge.")
    except ImportError:
        MT5_AVAILABLE = False
        class MockMT5:
            TIMEFRAME_M5 = 5
            ORDER_TYPE_BUY = 0
            ORDER_TYPE_SELL = 1
            POSITION_TYPE_BUY = 0
            POSITION_TYPE_SELL = 1
            TRADE_ACTION_DEAL = 1
            ORDER_TIME_GTC = 0
            ORDER_FILLING_IOC = 1
            TRADE_RETCODE_DONE = 10009
            def initialize(self, *_args): return False
            def shutdown(self, *_args): pass
            def copy_rates_from_pos(self, *_args): return None
            def positions_get(self, *_args, **_kwargs): return []
            def symbol_info(self, *_args): return None
            def symbol_info_tick(self, *_args): return None
            def order_send(self, *_args): return None
        mt5 = MockMT5()
        print("WARNING: MetaTrader5 library not found. Running in MOCK mode (No real trades).")

import pandas as pd
import numpy as np
import time
import sqlite3
import os
from datetime import datetime

# --- Path Configuration ---
DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "trading_data.db")
LIVE_TRADES_CSV = os.path.join(DATA_DIR, "live_trades.csv")

def log_transaction(symbol, trade_type, price, lots, magic, status, order_id=0):
    """Log transaction to SQLite and export to CSV for Docker/Jupyter access."""
    try:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                type TEXT,
                price REAL,
                lots REAL,
                magic INTEGER,
                status TEXT,
                order_id INTEGER
            )
        ''')
        
        timestamp = datetime.now()
        cursor.execute('''
            INSERT INTO transactions (timestamp, symbol, type, price, lots, magic, status, order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, trade_type, price, lots, magic, status, order_id))
        
        conn.commit()
        
        # Export to CSV for easy access in Jupyter/Docker
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        df.to_csv(LIVE_TRADES_CSV, index=False)
        
        conn.close()
        print(f"Transaction logged and synced to {LIVE_TRADES_CSV}")
    except Exception as e:
        print(f"Logging Error: {e}")

# --- Configuration ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.1

# Supertrend Parameters
ST_PERIOD = 10
ST_FACTOR = 3.0

# Alligator Parameters
JAW_PERIOD = 13
JAW_OFFSET = 8
TEETH_PERIOD = 8
TEETH_OFFSET = 5
LIPS_PERIOD = 5
LIPS_OFFSET = 3

def get_data(symbol, timeframe, count=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        print(f"Failed to get data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_rma(series, period):
    """Running Moving Average (RMA) as used in Pine Script."""
    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_alligator(df):
    # Alligator uses hl2 (median price)
    df['hl2'] = (df['high'] + df['low']) / 2
    
    # Calculate RMA and then shift for offsets
    df['jaw'] = calculate_rma(df['hl2'], JAW_PERIOD).shift(JAW_OFFSET)
    df['teeth'] = calculate_rma(df['hl2'], TEETH_PERIOD).shift(TEETH_OFFSET)
    df['lips'] = calculate_rma(df['hl2'], LIPS_PERIOD).shift(LIPS_OFFSET)
    
    # Conditions
    df['ali_bull'] = (df['lips'] > df['teeth']) & (df['teeth'] > df['jaw'])
    df['ali_bear'] = (df['lips'] < df['teeth']) & (df['teeth'] < df['jaw'])
    
    return df

def calculate_supertrend(df, period=10, factor=3):
    hl2 = (df['high'] + df['low']) / 2
    
    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    # Pine Script ta.atr uses RMA, but usually Supertrend uses SMA or EMA ATR. 
    # Pine's ta.supertrend uses RMA for ATR.
    atr_rma = calculate_rma(true_range, period)
    
    upper_band = hl2 + (factor * atr_rma)
    lower_band = hl2 - (factor * atr_rma)
    
    supertrend = np.zeros(len(df))
    direction = np.ones(len(df)) # 1 for up, -1 for down
    
    for i in range(1, len(df)):
        if df['close'][i] > upper_band[i-1]:
            direction[i] = 1
        elif df['close'][i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
        
    df['st_val'] = supertrend
    df['st_up'] = direction == 1
    return df

def open_buy_position(symbol, lot):
    price = mt5.symbol_info_tick(symbol).ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "magic": 123456,
        "comment": "Alligator Supertrend Buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    
    # Log the attempt/result
    status = "OPEN" if result.retcode == mt5.TRADE_RETCODE_DONE else f"FAILED: {result.comment}"
    order_id = result.order if result.retcode == mt5.TRADE_RETCODE_DONE else 0
    log_transaction(symbol, "BUY", price, lot, 123456, status, order_id)
    
    return result

def close_position(symbol, ticket):
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return None
    
    pos = position[0]
    type_dict = {
        mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL,
        mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY
    }
    price_dict = {
        mt5.POSITION_TYPE_BUY: mt5.symbol_info_tick(symbol).bid,
        mt5.POSITION_TYPE_SELL: mt5.symbol_info_tick(symbol).ask
    }
    
    price_close = price_dict[pos.type]
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": pos.volume,
        "type": type_dict[pos.type],
        "position": ticket,
        "price": price_close,
        "magic": 123456,
        "comment": "Close Position",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    
    # Log the result
    status = "CLOSED" if result.retcode == mt5.TRADE_RETCODE_DONE else f"CLOSE_FAILED: {result.comment}"
    log_transaction(symbol, "CLOSE", price_close, pos.volume, 123456, status, ticket)
    
    return result

def main():
    if not mt5.initialize():
        print("MT5 initialization failed")
        return

    print("Connected to MT5. Starting trading loop...")
    
    try:
        while True:
            # Fetch data
            df = get_data(SYMBOL, TIMEFRAME, 200)
            if df is None:
                time.sleep(10)
                continue
            
            # Calculate Indicators
            df = calculate_alligator(df)
            df = calculate_supertrend(df, ST_PERIOD, ST_FACTOR)
            
            # --- NEW FILTERS ---
            # 3. Multi-Timeframe Trend Filter (200 MA)
            df['trend_ma'] = df['close'].rolling(window=200).mean()
            df['trend_filter'] = df['close'] > df['trend_ma']
            
            # 6. Session Filter
            df['hour'] = df['time'].dt.hour
            df['is_active_session'] = (df['hour'] >= 7) & (df['hour'] <= 20)
            
            # 5. Richer Features for AI
            df['rsi'] = 100 - (100 / (1 + df['close'].diff().apply(lambda x: x if x > 0 else 0).rolling(14).mean() / 
                                     df['close'].diff().apply(lambda x: -x if x < 0 else 0).rolling(14).mean()))
            df['body_size'] = np.abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['atr'] = calculate_rma(np.max(pd.concat([df['high'] - df['low'], 
                                                      np.abs(df['high'] - df['close'].shift()), 
                                                      np.abs(df['low'] - df['close'].shift())], axis=1), axis=1), 14)

            # Last closed bar (index -2) or current bar (index -1)? 
            last_bar = df.iloc[-2]
            
            st_up = last_bar['st_up']
            ali_bull = last_bar['ali_bull']
            ali_bear = last_bar['ali_bear']
            trend_ok = last_bar['trend_filter']
            session_ok = last_bar['is_active_session']
            
            # Check current positions
            positions = mt5.positions_get(symbol=SYMBOL)
            
            if not positions:
                # 3, 6. Trend & Session Filters
                if st_up and ali_bull and trend_ok and session_ok:
                    # 5. AI Filtering
                    should_trade = True
                    # Import ai_model logic if needed or use from backtest
                    from core import backtest
                    if backtest.ai_model is not None:
                        features = pd.DataFrame([[
                            last_bar['close'], last_bar['rsi'], last_bar['body_size'], 
                            last_bar['upper_wick'], last_bar['lower_wick'], last_bar['hour']
                        ]], columns=['entry_price', 'rsi', 'body_size', 'upper_wick', 'lower_wick', 'hour'])
                        prediction = backtest.ai_model.predict(features)[0]
                        if prediction == 0: 
                            should_trade = False
                            print("AI filtered signal.")
                    
                    if should_trade:
                        # 2. Position Sizing
                        account_info = mt5.account_info()
                        balance = account_info.balance if account_info else 10000
                        risk_per_trade = 0.01
                        sl_dist = last_bar['atr'] * 2
                        
                        risk_amount = balance * risk_per_trade
                        lots = (risk_amount / sl_dist) / 100000
                        lots = round(max(0.01, min(lots, 10.0)), 2)
                        
                        print(f"BUY Signal at {last_bar['time']} | Lots: {lots}")
                        res = open_buy_position(SYMBOL, lots)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Buy Order Executed: {res.order}")
                    else:
                        print(f"Buy Order Failed: {res.comment}")
            else:
                # Position exists, check for Close condition
                # Instruction: "ketika close hatus indikator menjadi bear semua"
                if not st_up and ali_bear:
                    print(f"CLOSE Signal at {last_bar['time']}")
                    for pos in positions:
                        res = close_position(SYMBOL, pos.ticket)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            print(f"Position {pos.ticket} Closed")
                        else:
                            print(f"Close Failed: {res.comment}")
            
            # Wait for next check (e.g., every 1 minute)
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("Stopping script...")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
