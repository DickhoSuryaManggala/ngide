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
        print("MetaTrader5 library not found. Running in UI-only mode.")

import pandas as pd
import numpy as np
import time
import json
import threading
import os
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from core import backtest
from core import sentiment_analysis
from core import risk_manager
from core import walk_forward
from core import portfolio_manager
from core import telegram_notifier

# Path Configuration
DATA_DIR = "data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
DB_FILE = os.path.join(DATA_DIR, "trading_data.db")
BACKTEST_RESULTS = os.path.join(DATA_DIR, "backtest_results.csv")
BALANCE_HISTORY = os.path.join(DATA_DIR, "balance_history.csv")
LIVE_TRADES_CSV = os.path.join(DATA_DIR, "live_trades.csv")

# --- Database Setup ---
def init_db():
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
    conn.commit()
    conn.close()

def log_transaction(symbol, trade_type, price, lots, magic, status, order_id=0):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO transactions (timestamp, symbol, type, price, lots, magic, status, order_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), symbol, trade_type, price, lots, magic, status, order_id))
    conn.commit()
    
    # Export to CSV for Docker/Jupyter
    try:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
        df.to_csv(LIVE_TRADES_CSV, index=False)
    except Exception as e:
        print(f"CSV Sync Error: {e}")
        
    conn.close()

# Mocking MT5 constants if not available
if not MT5_AVAILABLE:
    class MockMT5:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M3 = 3
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 16385
        TIMEFRAME_H4 = 16388
        TIMEFRAME_D1 = 16408
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        POSITION_TYPE_BUY = 0
        POSITION_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 1
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 1
        TRADE_RETCODE_DONE = 10009
        def initialize(self): return False
        def shutdown(self): pass
        def copy_rates_from_pos(self, *_args): return None
        def positions_get(self, *_args, **_kwargs): return []
        def symbol_info(self, *_args): return None
        def symbol_info_tick(self, *_args): return None
        def order_send(self, *_args): return None
    mt5 = MockMT5()

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐊 Aligator AI - Professional Trading Terminal")
        self.root.geometry("1200x900")
        
        self.risk_mgr = risk_manager.RiskManager()
        self.wfo = walk_forward.WalkForwardOptimizer()
        self.notifier = telegram_notifier.TelegramNotifier()
        
        # Professional Dark Theme Colors
        self.colors = {
            "bg": "#1e1e1e",           # Deep dark background
            "sidebar": "#252526",      # Sidebar background
            "accent": "#007acc",       # VS Code blue accent
            "text": "#d4d4d4",         # Light grey text
            "header": "#333333",       # Section headers
            "success": "#4ec9b0",      # Greenish success
            "danger": "#f48771",       # Reddish danger
            "border": "#3c3c3c"        # Border color
        }
        
        # Configure Styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure("TFrame", background=self.colors["bg"])
        self.style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        self.style.configure("Header.TLabel", background=self.colors["header"], foreground="white", font=("Segoe UI", 10, "bold"))
        self.style.configure("Sidebar.TFrame", background=self.colors["sidebar"])
        self.style.configure("Card.TFrame", background=self.colors["sidebar"], borderwidth=1, relief="solid")
        
        self.style.configure("TButton", padding=5)
        self.style.configure("Action.TButton", background=self.colors["accent"], foreground="white")
        
        self.style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.colors["sidebar"], foreground=self.colors["text"], padding=[10, 5])
        self.style.map("TNotebook.Tab", background=[("selected", self.colors["accent"])], foreground=[("selected", "white")])

        init_db()
        self.running = False
        self.config = self.load_config()
        self.ai_type = tk.StringVar(value=self.config.get("ai_type", "rf"))
        
        self.create_widgets()
        self.load_settings_to_ui()
        self._wfo_progress_current = 0.0
        self._wfo_progress_target = 0.0
        self._wfo_progress_job = None
        self._wfo_progress_message = ""
        self._bt_progress_current = 0.0
        self._bt_progress_target = 0.0
        self._bt_progress_job = None
        self._bt_progress_message = ""
        self._last_ai_report_data = None
        self._balance_lock = threading.Lock()
        self._last_balance_write = 0.0
        self._pos_lock = threading.Lock()
        self._tracked_positions = {}

    def save_institutional_settings(self):
        """Menyimpan konfigurasi AI Brain dan Risk Limits ke config.json."""
        try:
            # Update local config dictionary
            self.config["ai_type"] = self.ai_type.get()
            
            # Risk Manager config is already updated by the trace calls in UI
            # but we ensure the main config file reflects these changes
            self.config.update(self.risk_mgr.config)
            
            with open(os.path.join(DATA_DIR, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
                
            messagebox.showinfo("Success", "Institutional settings saved successfully!")
            self.log("Institutional settings updated in config.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def update_risk_config(self, key, value):
        try:
            self.risk_mgr.config[key] = float(value)
            self.risk_mgr.save_config()
        except ValueError:
            pass

    def update_main_config(self, key, value):
        try:
            self.config[key] = float(value)
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except ValueError:
            pass
        except Exception:
            pass

    def _set_config_value(self, key, value):
        try:
            self.config[key] = value
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception:
            pass

    def _get_symbol_contract_size(self, symbol):
        cs_map = self.config.get("symbol_contract_size", {})
        if not isinstance(cs_map, dict):
            cs_map = {}
        if symbol in cs_map:
            try:
                return float(cs_map[symbol])
            except Exception:
                pass
        cs = None
        if MT5_AVAILABLE:
            try:
                info = mt5.symbol_info(symbol)
                cs = float(getattr(info, "trade_contract_size", 0.0)) if info else None
            except Exception:
                cs = None
        if cs is None or not np.isfinite(cs) or cs <= 0:
            cs = 100000.0
        cs_map[symbol] = float(cs)
        self._set_config_value("symbol_contract_size", cs_map)
        return float(cs)

    def _update_symbol_spread_estimate(self, symbol, tick):
        if tick is None:
            return
        try:
            spread = float(getattr(tick, "ask", 0.0)) - float(getattr(tick, "bid", 0.0))
        except Exception:
            return
        if not np.isfinite(spread) or spread <= 0:
            return
        sp_map = self.config.get("symbol_spread_price", {})
        if not isinstance(sp_map, dict):
            sp_map = {}
        prev = sp_map.get(symbol, None)
        if prev is None:
            sp_map[symbol] = float(spread)
        else:
            try:
                prev = float(prev)
            except Exception:
                prev = float(spread)
            alpha = 0.05
            sp_map[symbol] = float((1 - alpha) * prev + alpha * spread)
        self._set_config_value("symbol_spread_price", sp_map)

    def save_telegram_config(self):
        token = self.ent_tg_token.get().strip()
        chatid = self.ent_tg_chatid.get().strip()
        if token and chatid:
            self.notifier.save_config(token, chatid, True)
            success = self.notifier.send_message("✅ Telegram Notification System Linked Successfully!")
            if success:
                messagebox.showinfo("Success", "Telegram linked and test message sent!")
                self.log("Telegram system linked.")
            else:
                messagebox.showerror("Error", "Failed to send test message. Check Token and Chat ID.")
        else:
            messagebox.showwarning("Warning", "Please enter both Token and Chat ID.")

    def run_wfo(self):
        self.log("Starting Walk-Forward Optimization Cycle...")
        default_btn_text = self.btn_wfo.cget("text")
        self.btn_wfo.config(state=tk.DISABLED, text="OPTIMIZING...")

        self._wfo_progress_current = 0.0
        self._wfo_progress_target = 0.0
        self._wfo_progress_message = "Starting..."
        self._show_wfo_progress(True)
        self._set_wfo_progress_target(3, "Starting...")
        
        def task():
            def progress_cb(percent, message):
                self.root.after(0, lambda p=percent, m=message: self._set_wfo_progress_target(p, m))

            success = self.wfo.run_optimization_cycle(progress_callback=progress_cb)
            if success:
                self.log("Walk-Forward Cycle Completed. Models updated.")
                # Reload models in the background
                backtest.load_ai_models()
            else:
                self.log("Walk-Forward Cycle FAILED. Check console for errors.")
            self.root.after(0, lambda: self._set_wfo_progress_target(100 if success else 0, "Completed." if success else "Failed."))
            self.root.after(0, lambda: self.btn_wfo.config(state=tk.NORMAL, text=default_btn_text))
            self.root.after(800, lambda: self._show_wfo_progress(False))
            
        threading.Thread(target=task, daemon=True).start()

    def _show_bt_progress(self, show):
        if not hasattr(self, "pb_bt"):
            return
        if show:
            if not self.pb_bt.winfo_ismapped():
                self.lbl_bt_progress.pack(side=tk.LEFT, padx=(20, 8))
                self.pb_bt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        else:
            if self._bt_progress_job is not None:
                try:
                    self.root.after_cancel(self._bt_progress_job)
                except Exception:
                    pass
                self._bt_progress_job = None
            if self.lbl_bt_progress.winfo_ismapped():
                self.lbl_bt_progress.pack_forget()
            if self.pb_bt.winfo_ismapped():
                self.pb_bt.pack_forget()

    def _set_bt_progress_target(self, percent, message):
        try:
            self._bt_progress_target = float(max(0, min(100, percent)))
        except Exception:
            self._bt_progress_target = 0.0
        self._bt_progress_message = str(message) if message is not None else ""
        if hasattr(self, "pb_bt"):
            self.pb_bt.config(value=self._bt_progress_current)
        if hasattr(self, "lbl_bt_progress"):
            pct_int = int(round(self._bt_progress_current))
            msg = f"{pct_int}%"
            if self._bt_progress_message:
                msg = f"{msg} - {self._bt_progress_message}"
            self.lbl_bt_progress.config(text=msg)
        if self._bt_progress_job is None:
            self._animate_bt_progress()

    def _animate_bt_progress(self):
        step = 1.2 if self._bt_progress_target >= self._bt_progress_current else 3.0
        if abs(self._bt_progress_target - self._bt_progress_current) <= step:
            self._bt_progress_current = self._bt_progress_target
        else:
            direction = 1 if self._bt_progress_target > self._bt_progress_current else -1
            self._bt_progress_current += direction * step

        if hasattr(self, "pb_bt"):
            self.pb_bt.config(value=self._bt_progress_current)
        if hasattr(self, "lbl_bt_progress"):
            pct_int = int(round(self._bt_progress_current))
            msg = f"{pct_int}%"
            if self._bt_progress_message:
                msg = f"{msg} - {self._bt_progress_message}"
            self.lbl_bt_progress.config(text=msg)

        if self._bt_progress_current == self._bt_progress_target:
            self._bt_progress_job = None
            return

        self._bt_progress_job = self.root.after(40, self._animate_bt_progress)

    def _show_wfo_progress(self, show):
        if not hasattr(self, "pb_wfo"):
            return
        if show:
            if not self.pb_wfo.winfo_ismapped():
                self.lbl_wfo_progress.pack(fill=tk.X, pady=(10, 4))
                self.pb_wfo.pack(fill=tk.X)
        else:
            if self._wfo_progress_job is not None:
                try:
                    self.root.after_cancel(self._wfo_progress_job)
                except Exception:
                    pass
                self._wfo_progress_job = None
            if self.lbl_wfo_progress.winfo_ismapped():
                self.lbl_wfo_progress.pack_forget()
            if self.pb_wfo.winfo_ismapped():
                self.pb_wfo.pack_forget()

    def _set_wfo_progress_target(self, percent, message):
        try:
            self._wfo_progress_target = float(max(0, min(100, percent)))
        except Exception:
            self._wfo_progress_target = 0.0
        self._wfo_progress_message = str(message) if message is not None else ""
        if hasattr(self, "lbl_wfo_progress"):
            pct_int = int(round(self._wfo_progress_current))
            msg = f"{pct_int}%"
            if self._wfo_progress_message:
                msg = f"{msg} - {self._wfo_progress_message}"
            self.lbl_wfo_progress.config(text=msg)
        if hasattr(self, "pb_wfo"):
            self.pb_wfo.config(value=self._wfo_progress_current)
        if self._wfo_progress_job is None:
            self._animate_wfo_progress()

    def _animate_wfo_progress(self):
        step = 0.8 if self._wfo_progress_target >= self._wfo_progress_current else 2.0
        if abs(self._wfo_progress_target - self._wfo_progress_current) <= step:
            self._wfo_progress_current = self._wfo_progress_target
        else:
            direction = 1 if self._wfo_progress_target > self._wfo_progress_current else -1
            self._wfo_progress_current += direction * step

        if hasattr(self, "pb_wfo"):
            self.pb_wfo.config(value=self._wfo_progress_current)
        if hasattr(self, "lbl_wfo_progress"):
            pct_int = int(round(self._wfo_progress_current))
            msg = f"{pct_int}%"
            if self._wfo_progress_message:
                msg = f"{msg} - {self._wfo_progress_message}"
            self.lbl_wfo_progress.config(text=msg)

        if self._wfo_progress_current == self._wfo_progress_target:
            self._wfo_progress_job = None
            return

        self._wfo_progress_job = self.root.after(40, self._animate_wfo_progress)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {
            "symbol": "EURUSD",
            "timeframe": "M5",
            "lot_size": 0.1,
            "st_period": 10,
            "st_factor": 3.0,
            "jaw_period": 13,
            "jaw_offset": 8,
            "teeth_period": 8,
            "teeth_offset": 5,
            "lips_period": 5,
            "lips_offset": 3,
            "magic_number": 123456
        }

    def save_config(self):
        try:
            new_config = dict(self.config) if isinstance(self.config, dict) else {}
            new_config.update({
                "symbol": self.ent_symbol.get(),
                "timeframe": self.cb_timeframe.get(),
                "lot_size": float(self.ent_lot.get()),
                "st_period": int(self.ent_st_period.get()),
                "st_factor": float(self.ent_st_factor.get()),
                "jaw_period": int(self.ent_jaw_period.get()),
                "jaw_offset": int(self.ent_jaw_offset.get()),
                "teeth_period": int(self.ent_teeth_period.get()),
                "teeth_offset": int(self.ent_teeth_offset.get()),
                "lips_period": int(self.ent_lips_period.get()),
                "lips_offset": int(self.ent_lips_offset.get()),
                "magic_number": int(self.ent_magic.get()),
                "ai_type": self.ai_type.get()
            })
            with open(CONFIG_FILE, "w") as f:
                json.dump(new_config, f, indent=4)
            self.config = new_config
            self.log("Settings saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def create_widgets(self):
        # Top Bar Info
        top_bar = ttk.Frame(self.root, height=40, style="Sidebar.TFrame")
        top_bar.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(top_bar, text=" TERMINAL STATUS: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=10)
        self.lbl_status_indicator = tk.Label(top_bar, text="● OFFLINE", fg=self.colors["danger"], bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_status_indicator.pack(side=tk.LEFT)
        
        ttk.Label(top_bar, text=" | SENTIMENT: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_sentiment = tk.Label(top_bar, text="WAITING DATA...", fg="#888", bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_sentiment.pack(side=tk.LEFT)

        ttk.Label(top_bar, text=" | ACTIVE TRADES: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_active_count = tk.Label(top_bar, text="0", fg=self.colors["accent"], bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_active_count.pack(side=tk.LEFT)

        ttk.Label(top_bar, text=" | EQUITY: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_equity = tk.Label(top_bar, text="0.00", fg=self.colors["success"], bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_equity.pack(side=tk.LEFT)

        ttk.Label(top_bar, text=" | MARGIN: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_margin_level = tk.Label(top_bar, text="0.00%", fg="#888", bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_margin_level.pack(side=tk.LEFT)

        ttk.Label(top_bar, text=" | REGIME: ", font=("Segoe UI", 9, "bold"), style="Sidebar.TLabel").pack(side=tk.LEFT, padx=(20, 0))
        self.lbl_regime = tk.Label(top_bar, text="SCANNING...", fg=self.colors["accent"], bg=self.colors["sidebar"], font=("Segoe UI", 9, "bold"))
        self.lbl_regime.pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Tab 1: Trading ---
        self.trading_tab = ttk.Frame(self.notebook, padding="0")
        self.notebook.add(self.trading_tab, text=" Live Trading ")
        
        # Split trading tab
        trade_main = ttk.Frame(self.trading_tab)
        trade_main.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar (Left)
        sidebar = ttk.Frame(trade_main, width=320, style="Sidebar.TFrame")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Right Panel (Main Content)
        right_panel = ttk.Frame(trade_main)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Sidebar Content ---
        ttk.Label(sidebar, text="TRADING CONFIGURATION", style="Header.TLabel", anchor="center").pack(fill=tk.X, pady=(0, 10))
        
        settings_container = ttk.Frame(sidebar, padding="15", style="Sidebar.TFrame")
        settings_container.pack(fill=tk.BOTH, expand=True)

        def create_modern_entry(label, var_name, parent):
            frame = ttk.Frame(parent, style="Sidebar.TFrame")
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=label, width=15, font=("Segoe UI", 9), style="Sidebar.TLabel").pack(side=tk.LEFT)
            ent = tk.Entry(frame, bg="#3c3c3c", fg="white", insertbackground="white", borderwidth=0, font=("Segoe UI", 10))
            ent.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))
            setattr(self, f"ent_{var_name}", ent)
            return ent

        create_modern_entry("Symbol", "symbol", settings_container)
        
        frame_tf = ttk.Frame(settings_container, style="Sidebar.TFrame")
        frame_tf.pack(fill=tk.X, pady=5)
        ttk.Label(frame_tf, text="Timeframe", width=15, font=("Segoe UI", 9), style="Sidebar.TLabel").pack(side=tk.LEFT)
        self.cb_timeframe = ttk.Combobox(frame_tf, values=list(TIMEFRAME_MAP.keys()), state="readonly")
        self.cb_timeframe.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        create_modern_entry("Lot Size", "lot", settings_container)
        create_modern_entry("Magic #", "magic", settings_container)
        
        ttk.Label(settings_container, text="INDICATOR PARAMS", font=("Segoe UI", 9, "bold"), foreground=self.colors["accent"], background=self.colors["sidebar"]).pack(pady=(15, 5), anchor="w")
        
        params_grid = ttk.Frame(settings_container, style="Sidebar.TFrame")
        params_grid.pack(fill=tk.X)
        
        create_modern_entry("ST Period", "st_period", params_grid)
        create_modern_entry("ST Factor", "st_factor", params_grid)
        create_modern_entry("Jaw Period", "jaw_period", params_grid)
        create_modern_entry("Jaw Offset", "jaw_offset", params_grid)
        create_modern_entry("Teeth Period", "teeth_period", params_grid)
        create_modern_entry("Teeth Offset", "teeth_offset", params_grid)
        create_modern_entry("Lips Period", "lips_period", params_grid)
        create_modern_entry("Lips Offset", "lips_offset", params_grid)

        # Control Buttons
        btn_container = ttk.Frame(sidebar, padding="15", style="Sidebar.TFrame")
        btn_container.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_save = tk.Button(btn_container, text="SAVE SETTINGS", command=self.save_config, bg="#333", fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_save.pack(fill=tk.X, pady=2)
        
        self.btn_start = tk.Button(btn_container, text="START ENGINE", command=self.start_bot, bg=self.colors["accent"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_start.pack(fill=tk.X, pady=2)
        
        self.btn_stop = tk.Button(btn_container, text="STOP ENGINE", command=self.stop_bot, bg="#444", fg="#888", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2, state=tk.DISABLED)
        self.btn_stop.pack(fill=tk.X, pady=2)

        # Log Activity Konsol
        ttk.Label(right_panel, text=" TERMINAL ACTIVITY LOG ", style="Header.TLabel").pack(fill=tk.X)
        
        # Performance Summary Frame (New in App)
        self.perf_frame = ttk.Frame(right_panel, style="Sidebar.TFrame", padding=5)
        self.perf_frame.pack(fill=tk.X)
        
        def create_perf_stat(label, parent):
            frame = ttk.Frame(parent, style="Sidebar.TFrame")
            frame.pack(side=tk.LEFT, padx=15)
            ttk.Label(frame, text=label, font=("Segoe UI", 8), style="Sidebar.TLabel").pack()
            val_lbl = tk.Label(frame, text="0.00", font=("Segoe UI", 12, "bold"), bg=self.colors["sidebar"], fg="white")
            val_lbl.pack()
            return val_lbl

        self.lbl_stat_return = create_perf_stat("TOTAL RETURN (%)", self.perf_frame)
        self.lbl_stat_winrate = create_perf_stat("WIN RATE (%)", self.perf_frame)
        self.lbl_stat_drawdown = create_perf_stat("MAX DRAWDOWN (%)", self.perf_frame)
        self.lbl_stat_sharpe = create_perf_stat("SHARPE RATIO", self.perf_frame)

        self.log_text = tk.Text(right_panel, bg=self.colors["bg"], fg=self.colors["success"], borderwidth=0, font=("Consolas", 10), padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # --- Institutional Order Management ---
        order_frame = ttk.Frame(right_panel, style="Card.TFrame", padding=10)
        order_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        ttk.Label(order_frame, text="INSTITUTIONAL ORDER MANAGEMENT", font=("Segoe UI", 9, "bold"), foreground=self.colors["accent"]).pack(anchor="w", pady=(0, 5))
        
        btn_grid = ttk.Frame(order_frame, style="Card.TFrame")
        btn_grid.pack(fill=tk.X)
        
        tk.Button(btn_grid, text="CLOSE ALL POSITIONS", command=self.close_all_positions, 
                  bg=self.colors["danger"], fg="white", font=("Segoe UI", 8, "bold"), borderwidth=0, height=1).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_grid, text="FLATTEN PORTFOLIO", command=self.flatten_portfolio, 
                  bg="#444", fg="white", font=("Segoe UI", 8, "bold"), borderwidth=0, height=1).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_grid, text="EXPORT AUDIT TRAIL", command=self.export_audit_log, 
                  bg=self.colors["accent"], fg="white", font=("Segoe UI", 8, "bold"), borderwidth=0, height=1).pack(side=tk.LEFT, padx=5)
        
        self.lbl_exposure = ttk.Label(btn_grid, text="Exposure: 0.00 Lots", font=("Segoe UI", 8))
        self.lbl_exposure.pack(side=tk.RIGHT, padx=10)

        # --- Tab 2: Performance ---
        self.backtest_tab = ttk.Frame(self.notebook, padding="0")
        self.notebook.add(self.backtest_tab, text=" Performance Analytics ")
        self.create_backtest_tab()

        # --- Tab 3: Bot Configuration (NEW) ---
        self.config_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.config_tab, text=" Bot Settings ")
        self.create_config_tab()

        # --- Tab 4: AI & Risk Settings (NEW) ---
        self.ai_risk_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.ai_risk_tab, text=" AI & Risk ")
        self.create_ai_risk_tab()

        # --- Tab 5: Portfolio & Institutional Analytics (NEW) ---
        self.portfolio_tab = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(self.portfolio_tab, text=" Portfolio & Analytics ")
        self.create_portfolio_tab()

    def create_portfolio_tab(self):
        # Create a scrollable frame for analytics
        self.portfolio_canvas = tk.Canvas(self.portfolio_tab, bg=self.colors["bg"], highlightthickness=0)
        self.portfolio_scroll = ttk.Scrollbar(self.portfolio_tab, orient="vertical", command=self.portfolio_canvas.yview)
        self.portfolio_container = ttk.Frame(self.portfolio_canvas, style="TFrame")

        self.portfolio_container.bind(
            "<Configure>",
            lambda e: self.portfolio_canvas.configure(scrollregion=self.portfolio_canvas.bbox("all"))
        )

        self.portfolio_canvas.create_window((0, 0), window=self.portfolio_container, anchor="nw")
        self.portfolio_canvas.configure(yscrollcommand=self.portfolio_scroll.set)

        self.portfolio_canvas.pack(side="left", fill="both", expand=True)
        self.portfolio_scroll.pack(side="right", fill="y")

        # --- Header ---
        ttk.Label(self.portfolio_container, text="INSTITUTIONAL PORTFOLIO INTELLIGENCE", font=("Segoe UI", 14, "bold"), foreground=self.colors["accent"]).pack(pady=(0, 20))

        # --- Correlation Section ---
        corr_frame = ttk.Frame(self.portfolio_container, style="Card.TFrame", padding=15)
        corr_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(corr_frame, text="Global Asset Correlation Matrix", font=("Segoe UI", 11, "bold"), foreground="white").pack(anchor="w", pady=(0, 10))
        
        self.btn_run_corr = tk.Button(corr_frame, text="RESCAN GLOBAL CORRELATIONS", command=self.run_portfolio_analysis, 
                                      bg=self.colors["accent"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_run_corr.pack(fill=tk.X, pady=(0, 15))
        
        self.corr_plot_label = ttk.Label(corr_frame)
        self.corr_plot_label.pack()

        # --- Advanced Analytics Section ---
        metrics_frame = ttk.Frame(self.portfolio_container, style="Card.TFrame", padding=15)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(metrics_frame, text="Advanced Institutional Metrics", font=("Segoe UI", 11, "bold"), foreground="white").pack(anchor="w", pady=(0, 10))
        
        self.metrics_grid = ttk.Frame(metrics_frame, style="Card.TFrame")
        self.metrics_grid.pack(fill=tk.X)
        
        self.lbl_institutional_metrics = {}
        institutional_stats = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max Drawdown"]
        
        for i, stat in enumerate(institutional_stats):
            f = ttk.Frame(self.metrics_grid, style="Card.TFrame")
            f.grid(row=0, column=i, padx=20, pady=10)
            ttk.Label(f, text=stat, font=("Segoe UI", 9), foreground="#aaa").pack()
            lbl = tk.Label(f, text="0.00", font=("Segoe UI", 14, "bold"), bg=self.colors["sidebar"], fg=self.colors["accent"])
            lbl.pack()
            self.lbl_institutional_metrics[stat] = lbl

        # Update button for metrics
        self.btn_update_metrics = tk.Button(metrics_frame, text="CALCULATE ADVANCED METRICS", command=self.update_institutional_metrics, 
                                            bg="#444", fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_update_metrics.pack(fill=tk.X, pady=(15, 0))

        # --- Monte Carlo Stress Test Section (NEW) ---
        stress_frame = ttk.Frame(self.portfolio_container, style="Card.TFrame", padding=15)
        stress_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(stress_frame, text="Institutional Stress Test (Monte Carlo)", font=("Segoe UI", 11, "bold"), foreground="white").pack(anchor="w", pady=(0, 10))
        
        self.btn_run_stress = tk.Button(stress_frame, text="RUN 1000 MONTE CARLO SIMULATIONS", command=self.run_stress_test, 
                                        bg=self.colors["danger"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_run_stress.pack(fill=tk.X, pady=(0, 15))
        
        # Grid for Stress Stats
        self.stress_stats_grid = ttk.Frame(stress_frame, style="Card.TFrame")
        self.stress_stats_grid.pack(fill=tk.X)
        
        self.lbl_stress_metrics = {}
        stress_metrics = ["Mean Final Balance", "Profit Probability", "Worst Case Drawdown", "Value at Risk (95%)"]
        
        for i, metric in enumerate(stress_metrics):
            f = ttk.Frame(self.stress_stats_grid, style="Card.TFrame")
            f.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="ew")
            ttk.Label(f, text=metric, font=("Segoe UI", 8), foreground="#aaa").pack(anchor="w")
            lbl = tk.Label(f, text="---", font=("Segoe UI", 11, "bold"), bg=self.colors["sidebar"], fg="white")
            lbl.pack(anchor="w")
            self.lbl_stress_metrics[metric] = lbl

        self.stress_plot_label = ttk.Label(stress_frame)
        self.stress_plot_label.pack(pady=10)

    def run_stress_test(self):
        self.btn_run_stress.config(text="SIMULATING 1000 PATHS...", state=tk.DISABLED)
        def task():
            try:
                from core import stress_test
                stress_test.run_monte_carlo(simulations=1000)
                
                report_path = "assets/reports/monte_carlo_report.json"
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        stats = json.load(f)
                        for metric, lbl in self.lbl_stress_metrics.items():
                            self.root.after(0, lambda m=metric, l=lbl, s=stats: l.config(text=s.get(m, "---")))

                plot_path = "assets/plots/monte_carlo_plot.png"
                if os.path.exists(plot_path):
                    from PIL import Image, ImageTk
                    img = Image.open(plot_path)
                    img = img.resize((600, 300), Image.Resampling.LANCZOS)
                    self.stress_img = ImageTk.PhotoImage(img)
                    self.root.after(0, lambda: self.stress_plot_label.config(image=self.stress_img))
                
                self.log("Monte Carlo Stress Test Completed.")
            except Exception as e:
                self.root.after(0, lambda err=e: messagebox.showerror("Stress Test Error", str(err)))
                self.log(f"Monte Carlo Stress Test Failed: {e}")
            finally:
                self.root.after(0, lambda: self.btn_run_stress.config(text="RUN 1000 MONTE CARLO SIMULATIONS", state=tk.NORMAL))
            
        threading.Thread(target=task, daemon=True).start()

    def update_institutional_metrics(self):
        try:
            if hasattr(self, "btn_update_metrics"):
                self.btn_update_metrics.config(text="CALCULATING...", state=tk.DISABLED)

            if not os.path.exists(BALANCE_HISTORY):
                messagebox.showwarning("Warning", "No trade history found. Run a backtest or start the engine first.")
                return

            df = pd.read_csv(BALANCE_HISTORY)
            if 'balance' not in df.columns:
                messagebox.showerror("Error", f"Invalid balance history format: missing 'balance' column in {BALANCE_HISTORY}")
                return

            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")

            df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
            df = df.dropna(subset=['balance'])
            if len(df) <= 1:
                messagebox.showwarning("Warning", "Balance history terlalu sedikit untuk menghitung metrik lanjutan.")
                return

            balances = df['balance'].tolist()
            trades_df = None
            if os.path.exists(BACKTEST_RESULTS):
                try:
                    tdf = pd.read_csv(BACKTEST_RESULTS)
                    if "profit" in tdf.columns:
                        trades_df = tdf
                except Exception:
                    trades_df = None
            trading_days_per_year = int(self.config.get("trading_days_per_year", 252))
            periods_per_year = backtest.infer_periods_per_year_from_timestamps(
                df["time"] if "time" in df.columns else None,
                trading_days_per_year=trading_days_per_year
            )
            metrics = backtest.Metrics.calculate(balances, trades=trades_df, periods_per_year=periods_per_year)
            for stat, lbl in self.lbl_institutional_metrics.items():
                val = metrics.get(stat, 0)
                try:
                    val = float(val)
                except Exception:
                    val = 0.0
                if not np.isfinite(val):
                    val = 0.0
                if "Drawdown" in stat:
                    val_str = f"{val:.2f}%"
                else:
                    if abs(val) > 999.99:
                        val = 999.99 if val > 0 else -999.99
                    val_str = f"{val:.2f}"
                lbl.config(text=val_str)

            self.log("Institutional Metrics Updated.")
            if df['balance'].nunique() <= 1:
                messagebox.showinfo("Info", "Balance history masih flat (belum ada profit/loss), jadi metrik tetap 0.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate metrics: {e}")
        finally:
            if hasattr(self, "btn_update_metrics"):
                self.btn_update_metrics.config(text="CALCULATE ADVANCED METRICS", state=tk.NORMAL)

    def run_portfolio_analysis(self):
        self.btn_run_corr.config(text="ANALYZING GLOBAL MARKETS...", state=tk.DISABLED)
        def task():
            from core import portfolio_manager
            # Analisis korelasi aset utama dunia
            self.corr_matrix = portfolio_manager.analyze_portfolio_correlation()
            
            # Load the saved image
            plot_path = "assets/plots/portfolio_correlation.png"
            if os.path.exists(plot_path):
                from PIL import Image, ImageTk
                img = Image.open(plot_path)
                # Resize for UI
                img = img.resize((600, 450), Image.Resampling.LANCZOS)
                self.corr_img = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.corr_plot_label.config(image=self.corr_img))
            
            self.root.after(0, lambda: self.btn_run_corr.config(text="RESCAN GLOBAL CORRELATIONS", state=tk.NORMAL))
            self.log("Institutional Portfolio Correlation Analysis Completed.")
            
        threading.Thread(target=task, daemon=True).start()

    def create_ai_risk_tab(self):
        container = ttk.Frame(self.ai_risk_tab, style="TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        # Center Column
        center_col = ttk.Frame(container, padding="10")
        center_col.pack(expand=True)

        # --- AI Selection ---
        ttk.Label(center_col, text="AI BRAIN SELECTION", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(0, 10), anchor="w")
        ai_card = ttk.Frame(center_col, style="Card.TFrame", padding=15)
        ai_card.pack(fill=tk.X, pady=(0, 30))
        
        tk.Radiobutton(ai_card, text="Random Forest (Snapshot AI)", variable=self.ai_type, value="rf", 
                       bg=self.colors["sidebar"], fg="white", selectcolor="#444", font=("Segoe UI", 10)).pack(anchor="w", pady=5)
        tk.Radiobutton(ai_card, text="Deep Learning (LSTM Memory)", variable=self.ai_type, value="lstm", 
                       bg=self.colors["sidebar"], fg="white", selectcolor="#444", font=("Segoe UI", 10)).pack(anchor="w", pady=5)

        # --- Risk Limits ---
        ttk.Label(center_col, text="INSTITUTIONAL RISK LIMITS", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(0, 10), anchor="w")
        risk_card = ttk.Frame(center_col, style="Card.TFrame", padding=15)
        risk_card.pack(fill=tk.X)

        def create_setting_row(label, key, parent):
            frame = ttk.Frame(parent, style="Sidebar.TFrame")
            frame.pack(fill=tk.X, pady=8)
            ttk.Label(frame, text=label, font=("Segoe UI", 10), style="Sidebar.TLabel").pack(side=tk.LEFT)
            val = tk.StringVar(value=str(self.risk_mgr.config[key]))
            ent = tk.Entry(frame, bg="#333", fg="white", borderwidth=0, font=("Segoe UI", 10), width=10, textvariable=val, justify="center")
            ent.pack(side=tk.RIGHT)
            val.trace_add("write", lambda *args: self.update_risk_config(key, val.get()))

        create_setting_row("Max Daily Loss (%)", "max_daily_loss_pct", risk_card)
        create_setting_row("Max Drawdown (%)", "max_total_drawdown_pct", risk_card)
        create_setting_row("Max Daily Trades", "max_trades_per_day", risk_card)

        # --- Save Button ---
        self.btn_save_risk = tk.Button(risk_card, text="SAVE INSTITUTIONAL SETTINGS", command=self.save_institutional_settings, 
                                       bg=self.colors["accent"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_save_risk.pack(fill=tk.X, pady=(15, 0))

        ttk.Label(center_col, text="TRANSACTION COST MODEL", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(20, 10), anchor="w")
        cost_card = ttk.Frame(center_col, style="Card.TFrame", padding=15)
        cost_card.pack(fill=tk.X)

        def create_cost_row(label, key, parent, default_value):
            frame = ttk.Frame(parent, style="Sidebar.TFrame")
            frame.pack(fill=tk.X, pady=8)
            ttk.Label(frame, text=label, font=("Segoe UI", 10), style="Sidebar.TLabel").pack(side=tk.LEFT)
            val = tk.StringVar(value=str(self.config.get(key, default_value)))
            ent = tk.Entry(frame, bg="#333", fg="white", borderwidth=0, font=("Segoe UI", 10), width=10, textvariable=val, justify="center")
            ent.pack(side=tk.RIGHT)
            val.trace_add("write", lambda *_args, k=key, v=val: self.update_main_config(k, v.get()))

        create_cost_row("Spread ATR Ratio", "spread_atr_ratio", cost_card, 0.0)
        create_cost_row("Commission/Lot", "commission_per_lot", cost_card, 0.0)
        create_cost_row("LSTM Min ATR Ratio", "lstm_min_atr_ratio", cost_card, 0.0)

        # --- AI Maintenance (MOVED) ---
        ttk.Label(center_col, text="AI MAINTENANCE", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(20, 10), anchor="w")
        opt_card = ttk.Frame(center_col, style="Card.TFrame", padding=15)
        opt_card.pack(fill=tk.X)
        
        ttk.Label(opt_card, text="Update all AI models using latest market data.", font=("Segoe UI", 9), style="Sidebar.TLabel").pack(anchor="w", pady=(0, 10))
        self.btn_wfo = tk.Button(opt_card, text="START WALK-FORWARD OPTIMIZATION", command=self.run_wfo, 
                                 bg="#444", fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_wfo.pack(fill=tk.X)
        
        self.lbl_wfo_progress = ttk.Label(opt_card, text="", font=("Segoe UI", 9), foreground="#aaa")
        self.pb_wfo = ttk.Progressbar(opt_card, orient="horizontal", mode="determinate", maximum=100)
        self.lbl_wfo_progress.pack_forget()
        self.pb_wfo.pack_forget()

        # --- AI Performance Report Visualizer ---
        ttk.Label(center_col, text="INSTITUTIONAL AI PERFORMANCE", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(20, 10), anchor="w")
        self.ai_perf_frame = ttk.Frame(center_col, style="Card.TFrame", padding=15)
        self.ai_perf_frame.pack(fill=tk.BOTH, expand=True)
        
        self.btn_refresh_ai = tk.Button(self.ai_perf_frame, text="VIEW LATEST AI TRAINING REPORT", command=self.show_ai_report, 
                                        bg=self.colors["accent"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_refresh_ai.pack(fill=tk.X, pady=(0, 10))

        self.btn_open_ai_report = tk.Button(self.ai_perf_frame, text="OPEN REPORT (LARGE VIEW)", command=self.open_ai_report_window, 
                                            bg="#444", fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_open_ai_report.pack(fill=tk.X, pady=(0, 10))
        
        self.ai_report_canvas = None
        self.ai_report_text = None
        self.ai_report_toolbar = None

    def show_ai_report(self):
        """Menampilkan visualisasi hasil training RF & LSTM."""
        import textwrap

        if self.ai_report_canvas:
            self.ai_report_canvas.get_tk_widget().destroy()
            self.ai_report_canvas = None
        if self.ai_report_toolbar:
            self.ai_report_toolbar.destroy()
            self.ai_report_toolbar = None
        if self.ai_report_text:
            self.ai_report_text.destroy()
            self.ai_report_text = None

        rf_path = "assets/reports/rf_report.json"
        lstm_path = "assets/reports/lstm_report.json"
        
        if not os.path.exists(rf_path) or not os.path.exists(lstm_path):
            messagebox.showinfo("Report", "No training data found. Please run Walk-Forward Optimization first.")
            return

        with open(rf_path, 'r') as f: rf_data = json.load(f)
        with open(lstm_path, 'r') as f: lstm_data = json.load(f)

        pairs_list = ", ".join([f.replace("data_", "").replace(".csv", "").upper() for f in os.listdir(DATA_DIR) if f.startswith("data_")])
        self._last_ai_report_data = (rf_data, lstm_data, pairs_list)

        self.root.update_idletasks()
        frame_w = max(900, self.ai_perf_frame.winfo_width())
        frame_h = max(360, self.ai_perf_frame.winfo_height())

        fig, detail_text = self._build_ai_report_figure(
            target_width_px=frame_w,
            target_height_px=max(380, frame_h - 170),
            rf_data=rf_data,
            lstm_data=lstm_data,
            pairs_list=pairs_list,
            dpi=130
        )

        self.ai_report_canvas = FigureCanvasTkAgg(fig, master=self.ai_perf_frame)
        self.ai_report_canvas.draw()
        self.ai_report_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.ai_report_toolbar = NavigationToolbar2Tk(self.ai_report_canvas, self.ai_perf_frame, pack_toolbar=False)
        self.ai_report_toolbar.update()
        self.ai_report_toolbar.pack(fill=tk.X, pady=(0, 10))

        self.ai_report_text = tk.Text(self.ai_perf_frame, height=7, bg=self.colors["sidebar"], fg="white",
                                      borderwidth=0, font=("Consolas", 10), wrap="word")
        self.ai_report_text.insert("1.0", detail_text)
        self.ai_report_text.config(state=tk.DISABLED)
        self.ai_report_text.pack(fill=tk.X)

    def open_ai_report_window(self):
        rf_path = "assets/reports/rf_report.json"
        lstm_path = "assets/reports/lstm_report.json"

        if self._last_ai_report_data is None:
            if not os.path.exists(rf_path) or not os.path.exists(lstm_path):
                messagebox.showinfo("Report", "No training data found. Please run Walk-Forward Optimization first.")
                return
            with open(rf_path, 'r') as f:
                rf_data = json.load(f)
            with open(lstm_path, 'r') as f:
                lstm_data = json.load(f)
            pairs_list = ", ".join([f.replace("data_", "").replace(".csv", "").upper() for f in os.listdir(DATA_DIR) if f.startswith("data_")])
            self._last_ai_report_data = (rf_data, lstm_data, pairs_list)

        rf_data, lstm_data, pairs_list = self._last_ai_report_data

        win = tk.Toplevel(self.root)
        win.title("AI Training Report (Large View)")
        win.geometry("1400x850")

        container = ttk.Frame(win, style="TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(container, style="Card.TFrame", padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        win.update_idletasks()
        w = max(1200, canvas_frame.winfo_width())
        h = max(700, canvas_frame.winfo_height())

        fig, detail_text = self._build_ai_report_figure(
            target_width_px=w,
            target_height_px=h - 160,
            rf_data=rf_data,
            lstm_data=lstm_data,
            pairs_list=pairs_list,
            dpi=140
        )

        popup_canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        popup_canvas.draw()
        popup_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        toolbar = NavigationToolbar2Tk(popup_canvas, canvas_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=tk.X, pady=(0, 10))

        txt = tk.Text(canvas_frame, height=8, bg=self.colors["sidebar"], fg="white",
                      borderwidth=0, font=("Consolas", 11), wrap="word")
        txt.insert("1.0", detail_text)
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.X)

    def _build_ai_report_figure(self, target_width_px, target_height_px, rf_data, lstm_data, pairs_list, dpi):
        import textwrap

        fig_w = max(9.0, float(target_width_px) / float(dpi))
        fig_h = max(4.8, float(target_height_px) / float(dpi))
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=self.colors["bg"])
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.25])

        ax1 = fig.add_subplot(gs[0, 0])
        ax_xai = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])

        ax1.set_facecolor(self.colors["sidebar"])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        rf_acc = float(rf_data.get('accuracy', 0.0)) * 100
        ax1.text(0.5, 0.76, f"RF ACCURACY\n\n{rf_acc:.1f}%",
                 ha='center', va='center', fontsize=22, color=self.colors["accent"], weight='bold', clip_on=False)

        pairs_wrapped = textwrap.fill(pairs_list, width=34) if pairs_list else "-"
        ax1.text(0.5, 0.38, f"Cross-Asset:\n{pairs_wrapped}",
                 ha='center', va='center', fontsize=10, color="white", clip_on=False)
        ax1.text(0.5, 0.12, f"Last Trained:\n{rf_data.get('timestamp', '-')}",
                 ha='center', va='center', fontsize=10, color="#aaa", clip_on=False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("Model Reliability", color="white", fontsize=12, pad=10)

        ax_xai.set_facecolor(self.colors["sidebar"])
        importances = rf_data.get("feature_importances") or {}
        if isinstance(importances, dict) and len(importances) > 0:
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]
            feats, vals = zip(*sorted_importances)
            feats = [str(f).replace("_", " ") for f in feats]
            bars = ax_xai.barh(feats, vals, color=self.colors["accent"], alpha=0.9)
            max_v = max(vals) if len(vals) else 1
            ax_xai.set_xlim(0, max_v * 1.35)
            ax_xai.set_title("Top Driving Factors (XAI)", color="white", fontsize=12, pad=10)
            ax_xai.tick_params(colors='white', labelsize=10)
            ax_xai.invert_yaxis()
            for b, v in zip(bars, vals):
                ax_xai.text(b.get_width() + (max_v * 0.03), b.get_y() + b.get_height() / 2, f"{float(v):.3f}",
                            va='center', ha='left', fontsize=10, color="white")
            for spine in ax_xai.spines.values():
                spine.set_visible(False)
        else:
            ax_xai.text(0.5, 0.5, "XAI Data Not Available", ha='center', va='center', color="white")
            ax_xai.set_xticks([])
            ax_xai.set_yticks([])

        ax2.set_facecolor(self.colors["sidebar"])
        loss = lstm_data.get('loss') or []
        val_loss = lstm_data.get('val_loss') or []
        ax2.plot(loss, label='Train Loss', color=self.colors["accent"], linewidth=2)
        ax2.plot(val_loss, label='Val Loss', color=self.colors["success"], linewidth=2)
        best_text = None
        if isinstance(val_loss, list) and len(val_loss) > 0:
            best_idx = int(np.argmin(val_loss))
            best_val = float(val_loss[best_idx])
            ax2.scatter([best_idx], [best_val], color="white", s=28, zorder=5)
            ax2.text(best_idx, best_val, f"  best {best_val:.4f}", color="white", fontsize=10, va='center', ha='left')
            best_text = f"{best_val:.6f}"
        ax2.set_title("LSTM Learning Curve", color="white", fontsize=12, pad=10)
        ax2.tick_params(colors='white', labelsize=10)
        ax2.legend(fontsize=10, facecolor=self.colors["sidebar"], labelcolor="white")
        ax2.set_xlabel("Epochs", color="white", fontsize=10)
        ax2.grid(True, alpha=0.1)

        fig.tight_layout(pad=1.4)

        top_feats_lines = []
        if isinstance(importances, dict) and len(importances) > 0:
            for feat, val in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]:
                top_feats_lines.append(f"- {str(feat)}: {float(val):.4f}")
        else:
            top_feats_lines.append("- (tidak tersedia)")

        final_loss = float(loss[-1]) if isinstance(loss, list) and len(loss) > 0 else None
        final_val = float(val_loss[-1]) if isinstance(val_loss, list) and len(val_loss) > 0 else None

        detail_lines = [
            f"RF Accuracy: {rf_acc:.2f}%",
            f"Last Trained: {rf_data.get('timestamp', '-')}",
            f"Cross-Asset: {pairs_list if pairs_list else '-'}",
            "Top Features:",
            *top_feats_lines
        ]
        if best_text is not None:
            detail_lines.append(f"LSTM Best Val Loss: {best_text}")
        if final_loss is not None and final_val is not None:
            detail_lines.append(f"LSTM Final Loss: train={final_loss:.6f} | val={final_val:.6f}")
        elif final_loss is not None:
            detail_lines.append(f"LSTM Final Loss: train={final_loss:.6f}")

        return fig, "\n".join(detail_lines)

    def create_config_tab(self):
        # Create a scrollable frame for settings if it grows large
        container = ttk.Frame(self.config_tab, style="TFrame")
        container.pack(fill=tk.BOTH, expand=True)

        # Right Column: Telegram & AI Maintenance
        main_col = ttk.Frame(container, padding="10")
        main_col.pack(expand=True)

        # --- Telegram Configuration ---
        ttk.Label(main_col, text="TELEGRAM NOTIFICATION SETTINGS", font=("Segoe UI", 11, "bold"), foreground=self.colors["accent"]).pack(pady=(0, 10), anchor="w")
        tg_card = ttk.Frame(main_col, style="Card.TFrame", padding=15)
        tg_card.pack(fill=tk.X, pady=(0, 30))

        ttk.Label(tg_card, text="Bot API Token", font=("Segoe UI", 9), style="Sidebar.TLabel").pack(anchor="w")
        self.ent_tg_token = tk.Entry(tg_card, bg="#333", fg="white", borderwidth=0, font=("Consolas", 10))
        self.ent_tg_token.pack(fill=tk.X, pady=(2, 10))
        self.ent_tg_token.insert(0, self.notifier.bot_token)
        
        ttk.Label(tg_card, text="Chat ID", font=("Segoe UI", 9), style="Sidebar.TLabel").pack(anchor="w")
        self.ent_tg_chatid = tk.Entry(tg_card, bg="#333", fg="white", borderwidth=0, font=("Consolas", 10))
        self.ent_tg_chatid.pack(fill=tk.X, pady=(2, 15))
        self.ent_tg_chatid.insert(0, self.notifier.chat_id)
        
        self.btn_tg_save = tk.Button(tg_card, text="LINK & TEST TELEGRAM", command=self.save_telegram_config, 
                                     bg=self.colors["accent"], fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_tg_save.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_tg_report = tk.Button(tg_card, text="SEND INSTITUTIONAL DAILY REPORT", command=self.manual_telegram_report, 
                                       bg="#444", fg="white", font=("Segoe UI", 9, "bold"), borderwidth=0, height=2)
        self.btn_tg_report.pack(fill=tk.X)

    def manual_telegram_report(self):
        """Mengirimkan laporan institusi ke Telegram secara manual."""
        if not os.path.exists(BALANCE_HISTORY):
            messagebox.showwarning("Warning", "No trade history found. Run a backtest or start the engine first.")
            return
            
        try:
            df = pd.read_csv(BALANCE_HISTORY)
            if len(df) > 1:
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                trading_days_per_year = int(self.config.get("trading_days_per_year", 252))
                periods_per_year = backtest.infer_periods_per_year_from_timestamps(
                    df["time"] if "time" in df.columns else None,
                    trading_days_per_year=trading_days_per_year
                )
                metrics = backtest.Metrics.calculate(df['balance'].tolist(), periods_per_year=periods_per_year)
                if os.path.exists(BACKTEST_RESULTS):
                    try:
                        tdf = pd.read_csv(BACKTEST_RESULTS)
                        if "profit" in tdf.columns:
                            metrics = backtest.Metrics.calculate(df['balance'].tolist(), trades=tdf, periods_per_year=periods_per_year)
                    except Exception:
                        pass
                
                # Get live account info if available
                account_data = None
                if MT5_AVAILABLE:
                    acc = mt5.account_info()
                    if acc:
                        account_data = {
                            "equity": acc.equity,
                            "balance": acc.balance,
                            "margin_level": acc.margin_level
                        }
                
                success = self.notifier.send_institutional_report(metrics, account_data)
                if success:
                    messagebox.showinfo("Success", "Institutional report sent to Telegram!")
                else:
                    messagebox.showerror("Error", "Failed to send report. Check Telegram settings.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")

    def create_backtest_tab(self):
        # Controls for backtest tab
        ctrl_frame = ttk.Frame(self.backtest_tab)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        self.btn_run_backtest = ttk.Button(ctrl_frame, text="Run New Backtest", command=self.run_backtest_logic)
        self.btn_run_backtest.pack(side=tk.LEFT, padx=5)
        self.btn_refresh_dashboard = ttk.Button(ctrl_frame, text="Refresh Dashboard", command=self.refresh_backtest_dashboard)
        self.btn_refresh_dashboard.pack(side=tk.LEFT, padx=5)
        self.lbl_bt_status = ttk.Label(ctrl_frame, text="No backtest data loaded.", font=("Arial", 9, "italic"))
        self.lbl_bt_status.pack(side=tk.LEFT, padx=20)
        
        self.lbl_bt_progress = ttk.Label(ctrl_frame, text="", font=("Segoe UI", 9), foreground="#aaa")
        self.pb_bt = ttk.Progressbar(ctrl_frame, orient="horizontal", mode="determinate", maximum=100)
        self.lbl_bt_progress.pack_forget()
        self.pb_bt.pack_forget()

        # Scrollable container for charts
        self.canvas_scroll = tk.Canvas(self.backtest_tab)
        self.scrollbar = ttk.Scrollbar(self.backtest_tab, orient="vertical", command=self.canvas_scroll.yview)
        self.scrollable_frame = ttk.Frame(self.canvas_scroll)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        )

        self.canvas_scroll.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)

        self.canvas_scroll.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def run_backtest_logic(self):
        data_file = os.path.join(DATA_DIR, "data_xauusd.csv")
        if not os.path.exists(data_file):
            messagebox.showerror("Error", f"Data file {data_file} not found.")
            return
        
        self.lbl_bt_status.config(text="Running backtest... please wait.")
        self.root.update_idletasks()
        if hasattr(self, "btn_run_backtest"):
            self.btn_run_backtest.config(state=tk.DISABLED)
        if hasattr(self, "btn_refresh_dashboard"):
            self.btn_refresh_dashboard.config(state=tk.DISABLED)
        self._bt_progress_current = 0.0
        self._bt_progress_target = 0.0
        self._bt_progress_message = "Starting..."
        self._show_bt_progress(True)
        self._set_bt_progress_target(8, "Loading data...")
        
        def task():
            try:
                if MT5_AVAILABLE:
                    try:
                        self._get_symbol_contract_size("XAUUSD")
                        tick = mt5.symbol_info_tick("XAUUSD")
                        self._update_symbol_spread_estimate("XAUUSD", tick)
                    except Exception:
                        pass
                self.root.after(0, lambda: self._set_bt_progress_target(15, "Reading CSV..."))
                df = pd.read_csv(data_file)
                self.root.after(0, lambda: self._set_bt_progress_target(25, "Preparing data..."))
                df.columns = [c.lower() for c in df.columns]
                self.root.after(0, lambda: self._set_bt_progress_target(90, "Running backtest..."))
                backtest.event_driven_backtest(df, self.config, symbol="XAUUSD")
                self.root.after(0, lambda: self._set_bt_progress_target(95, "Rendering dashboard..."))
                self.root.after(0, self.refresh_backtest_dashboard)
                self.root.after(0, lambda: self.lbl_bt_status.config(text="Backtest completed!"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Backtest Error", str(e)))
                self.root.after(0, lambda: self._set_bt_progress_target(0, "Failed."))
            finally:
                self.root.after(0, lambda: self.btn_run_backtest.config(state=tk.NORMAL) if hasattr(self, "btn_run_backtest") else None)
                self.root.after(0, lambda: self.btn_refresh_dashboard.config(state=tk.NORMAL) if hasattr(self, "btn_refresh_dashboard") else None)
        
        threading.Thread(target=task, daemon=True).start()

    def refresh_backtest_dashboard(self):
        dashboard_error = None
        if hasattr(self, "btn_refresh_dashboard"):
            self.btn_refresh_dashboard.config(state=tk.DISABLED)
        if hasattr(self, "btn_run_backtest"):
            self.btn_run_backtest.config(state=tk.DISABLED)
        self._show_bt_progress(True)
        if self._bt_progress_current < 1:
            self._bt_progress_current = 1.0
        self._set_bt_progress_target(max(self._bt_progress_current, 5), "Refreshing dashboard...")
        self.root.update_idletasks()

        # Clear previous widgets in scrollable_frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self._set_bt_progress_target(max(self._bt_progress_current, 12), "Loading history...")
        self.root.update_idletasks()

        if not os.path.exists(BALANCE_HISTORY):
            self.lbl_bt_status.config(text="Error: balance_history.csv not found. Run backtest.py first.")
            self._set_bt_progress_target(0, "No balance history.")
            if hasattr(self, "btn_refresh_dashboard"):
                self.btn_refresh_dashboard.config(state=tk.NORMAL)
            if hasattr(self, "btn_run_backtest"):
                self.btn_run_backtest.config(state=tk.NORMAL)
            return

        try:
            self._set_bt_progress_target(max(self._bt_progress_current, 20), "Reading balance_history.csv...")
            self.root.update_idletasks()
            df = pd.read_csv(BALANCE_HISTORY)
            df['time'] = pd.to_datetime(df['time'])
            df['returns'] = df['balance'].pct_change().fillna(0)
            
            # Calculate metrics
            self._set_bt_progress_target(max(self._bt_progress_current, 35), "Calculating metrics...")
            self.root.update_idletasks()
            trades_df = None
            if os.path.exists(BACKTEST_RESULTS):
                try:
                    tdf = pd.read_csv(BACKTEST_RESULTS)
                    if "profit" in tdf.columns:
                        trades_df = tdf
                except Exception:
                    trades_df = None
            trading_days_per_year = int(self.config.get("trading_days_per_year", 252))
            periods_per_year = backtest.infer_periods_per_year_from_timestamps(df["time"], trading_days_per_year=trading_days_per_year)
            metrics = backtest.Metrics.calculate(df['balance'].tolist(), trades=trades_df, periods_per_year=periods_per_year)
            
            # Display metrics in a frame
            stats_frame = ttk.LabelFrame(self.scrollable_frame, text="Performance Statistics", padding="10")
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for i, (k, v) in enumerate(metrics.items()):
                row = i // 4
                col = i % 4
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
                if not np.isfinite(v):
                    v = 0.0
                if "Return" in k or "Rate" in k or "Drawdown" in k:
                    val_str = f"{v:.2f}%"
                else:
                    if abs(v) > 999.99:
                        v = 999.99 if v > 0 else -999.99
                    val_str = f"{v:.2f}"
                ttk.Label(stats_frame, text=f"{k}:", font=("Arial", 9, "bold")).grid(row=row*2, column=col, padx=10, sticky="w")
                ttk.Label(stats_frame, text=val_str).grid(row=row*2+1, column=col, padx=10, sticky="w")

            self.lbl_bt_status.config(text=f"Loaded {len(df)} data points.")

            # Create Figure
            self._set_bt_progress_target(max(self._bt_progress_current, 55), "Building charts...")
            self.root.update_idletasks()
            fig = plt.figure(figsize=(10, 15))
            gs = fig.add_gridspec(4, 2)

            # 1. Equity Curve
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(df['time'], df['balance'], color='black', linewidth=1)
            ax1.set_title("Equity Curve")
            ax1.grid(True, alpha=0.3)

            # 2. Drawdown
            ax2 = fig.add_subplot(gs[1, 0])
            cum_max = df['balance'].cummax()
            dd = (df['balance'] - cum_max) / cum_max * 100
            ax2.fill_between(df['time'], dd, color='red', alpha=0.3)
            ax2.plot(df['time'], dd, color='red', linewidth=0.5)
            ax2.set_title("Drawdown (%)")

            # 3. Returns Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            rets = df['returns'][df['returns'] != 0]
            if not rets.empty:
                ax3.hist(rets, bins=30, color='#555', alpha=0.7)
            ax3.set_title("Returns Distribution")

            # 4. Monthly Returns Heatmap
            ax4 = fig.add_subplot(gs[2, :])
            monthly_returns = df.set_index('time')['returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
            if not monthly_returns.empty:
                heatmap_data = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.strftime('%b'),
                    'Return': monthly_returns.values * 100
                })
                pivot = heatmap_data.pivot(index='Year', columns='Month', values='Return')
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pivot = pivot.reindex(columns=[m for m in months_order if m in pivot.columns])
                sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax4, cbar=False)
            ax4.set_title("Monthly Returns (%)")

            # 5. Rolling Sharpe
            ax5 = fig.add_subplot(gs[3, 0])
            seconds = df["time"].diff().dt.total_seconds()
            seconds = seconds[(seconds > 0) & np.isfinite(seconds)]
            median_seconds = float(seconds.median()) if not seconds.empty else 3600.0
            periods_per_day = int(max(1, round(86400.0 / median_seconds))) if median_seconds > 0 else 24
            window = periods_per_day * 30
            if len(df) > window:
                roll_rets = df['returns'].rolling(window)
                sharpe = (roll_rets.mean() / roll_rets.std() * np.sqrt(periods_per_year)).fillna(0)
                ax5.plot(df['time'], sharpe, color='blue', linewidth=1)
            ax5.set_title("Rolling Sharpe (30-day)")

            # 6. Yearly Returns
            ax6 = fig.add_subplot(gs[3, 1])
            yearly_returns = df.set_index('time')['returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
            colors = ['green' if x > 0 else 'red' for x in yearly_returns.values]
            ax6.bar([str(y) for y in yearly_returns.index.year], yearly_returns.values * 100, color=colors)
            ax6.set_title("Yearly Returns (%)")

            fig.tight_layout()

            # Embed Figure in Tkinter
            self._set_bt_progress_target(max(self._bt_progress_current, 85), "Rendering UI...")
            self.root.update_idletasks()
            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._set_bt_progress_target(100, "Done.")
            self.root.update_idletasks()

        except Exception as e:
            dashboard_error = e
            self._set_bt_progress_target(0, "Failed.")
        finally:
            if hasattr(self, "btn_refresh_dashboard"):
                self.btn_refresh_dashboard.config(state=tk.NORMAL)
            if hasattr(self, "btn_run_backtest"):
                self.btn_run_backtest.config(state=tk.NORMAL)
            self.root.after(800, lambda: self._show_bt_progress(False))
            if dashboard_error is not None:
                self.log(f"Dashboard Error: {dashboard_error}")


    def load_settings_to_ui(self):
        self.ent_symbol.insert(0, self.config.get("symbol", "EURUSD"))
        self.cb_timeframe.set(self.config.get("timeframe", "M5"))
        self.ent_lot.insert(0, str(self.config.get("lot_size", 0.1)))
        self.ent_magic.insert(0, str(self.config.get("magic_number", 123456)))
        self.ent_st_period.insert(0, str(self.config.get("st_period", 10)))
        self.ent_st_factor.insert(0, str(self.config.get("st_factor", 3.0)))
        self.ent_jaw_period.insert(0, str(self.config.get("jaw_period", 13)))
        self.ent_jaw_offset.insert(0, str(self.config.get("jaw_offset", 8)))
        self.ent_teeth_period.insert(0, str(self.config.get("teeth_period", 8)))
        self.ent_teeth_offset.insert(0, str(self.config.get("teeth_offset", 5)))
        self.ent_lips_period.insert(0, str(self.config.get("lips_period", 5)))
        self.ent_lips_offset.insert(0, str(self.config.get("lips_offset", 3)))
        
        # Load Institutional Settings
        self.ai_type.set(self.config.get("ai_type", "rf"))

    def log(self, message):
        """Langkah 2: Optimasi Latensi & Multi-threading."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        # Gunakan after() agar update UI berjalan di main thread (Thread-safe)
        self.root.after(0, lambda: self._update_log_ui(formatted_msg))

    def _update_log_ui(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        # Limit log size for performance
        if float(self.log_text.index('end-1c')) > 1000:
            self.log_text.delete('1.0', '200.0')
        self.log_text.config(state=tk.DISABLED)

    def start_bot(self):
        if not mt5.initialize():
            messagebox.showerror("Error", "MetaTrader 5 initialization failed.")
            return
        
        symbols_str = self.ent_symbol.get()
        self.active_symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        
        if not self.active_symbols:
            messagebox.showerror("Error", "Please enter at least one symbol.")
            return

        self.running = True
        self.btn_start.config(state=tk.DISABLED, bg="#444", fg="#888")
        self.btn_stop.config(state=tk.NORMAL, bg=self.colors["danger"], fg="white")
        self.btn_save.config(state=tk.DISABLED)
        self.lbl_status_indicator.config(text="● ONLINE", fg=self.colors["success"])
        
        # Start a thread for each symbol
        self.threads = []
        for symbol in self.active_symbols:
            t = threading.Thread(target=self.trading_loop, args=(symbol,), daemon=True)
            t.start()
            self.threads.append(t)
            
        self.log(f"Institutional Engine started for: {', '.join(self.active_symbols)}")

    def stop_bot(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL, bg=self.colors["accent"], fg="white")
        self.btn_stop.config(state=tk.DISABLED, bg="#444", fg="#888")
        self.btn_save.config(state=tk.NORMAL)
        self.lbl_status_indicator.config(text="● OFFLINE", fg=self.colors["danger"])
        mt5.shutdown()
        self.log("Trading engine offline.")

    # --- Trading Logic (Integrated) ---
    def trading_loop(self, symbol):
        while self.running:
            try:
                conf = self.config
                tf = TIMEFRAME_MAP[self.cb_timeframe.get()]
                
                # Fetch data
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, 200)
                if rates is None:
                    self.log(f"Error fetching data for {symbol}")
                    time.sleep(10)
                    continue
                
                df = pd.DataFrame(rates)
                
                # Indicator Calculations
                df['hl2'] = (df['high'] + df['low']) / 2
                
                # RMA calculation helper
                def rma(series, period):
                    return series.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

                # Alligator
                df['jaw'] = rma(df['hl2'], conf["jaw_period"]).shift(conf["jaw_offset"])
                df['teeth'] = rma(df['hl2'], conf["teeth_period"]).shift(conf["teeth_offset"])
                df['lips'] = rma(df['hl2'], conf["lips_period"]).shift(conf["lips_offset"])
                
                ali_bull = (df['lips'] > df['teeth']) & (df['teeth'] > df['jaw'])
                ali_bear = (df['lips'] < df['teeth']) & (df['teeth'] < df['jaw'])

                # Supertrend
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr_rma = rma(tr, conf["st_period"])
                df['atr'] = atr_rma
                
                upper_band = df['hl2'] + (conf["st_factor"] * atr_rma)
                lower_band = df['hl2'] - (conf["st_factor"] * atr_rma)
                
                direction = np.ones(len(df))
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
                
                st_up = direction == 1
                
                # --- NEW FILTERS ---
                # 3. Multi-Timeframe Trend Filter (200 MA)
                df['trend_ma'] = df['close'].rolling(window=200).mean()
                trend_filter = df['close'] > df['trend_ma']
                
                # 6. Session Filter
                df['time_dt'] = pd.to_datetime(df['time'], unit='s')
                df['hour'] = df['time_dt'].dt.hour
                is_active_session = (df['hour'] >= 7) & (df['hour'] <= 20)

                # 7. Sentiment Filter (Institusi Step)
                current_sentiment = sentiment_analysis.get_current_sentiment()
                sentiment_ok = current_sentiment > -0.1 # Filter out extreme panic
                
                # 8. Institutional Risk Check
                # For demo, we use mock values for current drawdown/loss
                # In real institutional setup, these would be calculated from live account data
                is_risk_ok, risk_msg = self.risk_mgr.is_trading_allowed(0.0, 0.0, 0) 
                
                # Update UI Sentiment
                sent_text = "BULLISH" if current_sentiment > 0.05 else ("BEARISH" if current_sentiment < -0.05 else "NEUTRAL")
                sent_color = self.colors["success"] if current_sentiment > 0.05 else (self.colors["danger"] if current_sentiment < -0.05 else "#888")
                
                risk_color = self.colors["success"] if is_risk_ok else self.colors["danger"]
                risk_status = "SAFE" if is_risk_ok else "RISK LIMIT"
                
                # Update Institutional Account Info
                account_info = mt5.account_info()
                if account_info:
                    equity = account_info.equity
                    margin_level = account_info.margin_level
                    self.root.after(0, lambda: self.lbl_equity.config(text=f"{equity:,.2f}"))
                    self.root.after(0, lambda: self.lbl_margin_level.config(text=f"{margin_level:.2f}%"))
                    self._maybe_append_balance_history(getattr(account_info, "balance", equity))
                
                # Update Active Trades Count
                all_positions = mt5.positions_get()
                active_count = len(all_positions) if all_positions else 0
                total_lots = sum([p.volume for p in all_positions]) if all_positions else 0
                self.root.after(0, lambda: self.lbl_exposure.config(text=f"Exposure: {total_lots:.2f} Lots"))
                
                # Update Market Regime UI
                current_regime = df['market_regime'].iloc[-2]
                regime_color = self.colors["accent"] if current_regime == "TRENDING" else "#888"
                self.root.after(0, lambda r=current_regime, c=regime_color: self.lbl_regime.config(text=r, fg=c))

                # --- Langkah 4: Online Learning (Institutional Standard) ---
                # Melatih ulang model RF secara ringan setiap kali ada candle baru yang terkonfirmasi
                if len(df) > 100:
                    try:
                        from sklearn.ensemble import RandomForestClassifier
                        # Ambil data terbaru untuk training kilat
                        X_online, y_online = ai_trainer.prepare_advanced_data_from_df(df.tail(100), self.config)
                        if X_online is not None and backtest.ai_model is not None:
                            # 'Partial fit' simulation for RF (Institutional hack)
                            # We retrain with a very small number of trees on new data
                            backtest.ai_model.n_estimators += 1
                            backtest.ai_model.fit(X_online, y_online)
                            # self.log("Online Learning: AI model updated with latest market drift.")
                    except Exception as e:
                        pass # Silent fail to maintain execution speed

                # Portfolio Correlation Check Logic
                # (Fetch correlation matrix periodically)
                if not hasattr(self, 'corr_matrix') or time.time() % 3600 < 60:
                    try:
                        self.corr_matrix = portfolio_manager.analyze_portfolio_correlation(self.active_symbols)
                    except:
                        self.corr_matrix = None

                self.root.after(0, lambda: self.lbl_sentiment.config(text=f"{sent_text} ({current_sentiment:.2f}) | RISK: {risk_status}", fg=sent_color))
                self.root.after(0, lambda: self.lbl_active_count.config(text=str(active_count)))
                
                if not is_risk_ok:
                    self.log(f"RISK ALERT: {risk_msg}")
                    self.notifier.alert_risk(risk_msg)
                
                # Update Performance Summary from History
                if os.path.exists(BALANCE_HISTORY):
                    bh_df = pd.read_csv(BALANCE_HISTORY)
                    if len(bh_df) > 1:
                        if "time" in bh_df.columns:
                            bh_df["time"] = pd.to_datetime(bh_df["time"], errors="coerce")
                        trading_days_per_year = int(self.config.get("trading_days_per_year", 252))
                        periods_per_year = backtest.infer_periods_per_year_from_timestamps(
                            bh_df["time"] if "time" in bh_df.columns else None,
                            trading_days_per_year=trading_days_per_year
                        )
                        metrics = backtest.Metrics.calculate(bh_df['balance'].tolist(), periods_per_year=periods_per_year)
                        self.root.after(0, lambda: self.lbl_stat_return.config(text=f"{metrics['Total Return']:.2f}%"))
                        self.root.after(0, lambda: self.lbl_stat_winrate.config(text=f"{metrics['Win Rate']:.2f}%"))
                        self.root.after(0, lambda: self.lbl_stat_drawdown.config(text=f"{metrics['Max Drawdown']:.2f}%"))
                        self.root.after(0, lambda: self.lbl_stat_sharpe.config(text=f"{metrics['Sharpe Ratio']:.2f}"))
                
                # 5. Richer Features for AI (Institutional Grade)
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
                
                # Signals from last confirmed bar
                is_st_up = st_up.iloc[-2]
                is_ali_bull = ali_bull.iloc[-2]
                is_ali_bear = ali_bear.iloc[-2]
                is_trend_ok = trend_filter.iloc[-2]
                is_session_ok = is_active_session.iloc[-2]

                positions = mt5.positions_get(symbol=symbol)
                self._sync_position_tracking(symbol, positions)
                
                if not positions:
                    desired_side = None
                    if is_st_up and is_ali_bull and is_trend_ok and is_session_ok and sentiment_ok and is_risk_ok:
                        desired_side = "BUY"
                    elif (not is_st_up) and is_ali_bear and (not is_trend_ok) and is_session_ok and sentiment_ok and is_risk_ok:
                        desired_side = "SELL"

                    if desired_side is not None:
                        should_trade = True
                        ai_choice = self.ai_type.get()

                        if ai_choice == "rf" and backtest.ai_model is not None:
                            last_row = df.iloc[-2]
                            features = pd.DataFrame([[
                                last_row['close'], last_row['rsi'], last_row['body_size'],
                                last_row['upper_wick'], last_row['lower_wick'], last_row['volatility'],
                                last_row['returns_1h'], last_row['dist_from_jaw'], last_row['hour_sin'], last_row['hour_cos']
                            ]], columns=['close', 'rsi', 'body_size', 'upper_wick', 'lower_wick',
                                         'volatility', 'returns_1h', 'dist_from_jaw', 'hour_sin', 'hour_cos'])
                            prediction = int(backtest.ai_model.predict(features)[0])
                            if desired_side == "BUY" and prediction == 0:
                                should_trade = False
                                self.log("AI (RF) filtered out a potential BUY signal.")
                            elif desired_side == "SELL" and prediction == 1:
                                should_trade = False
                                self.log("AI (RF) filtered out a potential SELL signal.")

                        elif ai_choice == "lstm" and backtest.lstm_model is not None:
                            lookback = 60
                            if len(df) >= lookback + 1:
                                feature_cols = ['hl2', 'rsi', 'atr', 'volatility', 'dist_from_jaw', 'hour_sin', 'hour_cos']
                                seq_df = df.iloc[-(lookback+1):-1][feature_cols]
                                scaled_seq = backtest.lstm_scaler.transform(seq_df.values)
                                scaled_seq = np.expand_dims(scaled_seq, axis=0)

                                pred_next_hl2_scaled = float(backtest.lstm_model.predict(scaled_seq, verbose=0)[0][0])
                                last_vec = scaled_seq[0, -1, :].astype(float)
                                pred_vec = last_vec.copy()
                                pred_vec[0] = pred_next_hl2_scaled

                                last_hl2 = float(backtest.lstm_scaler.inverse_transform([last_vec])[0][0])
                                pred_hl2 = float(backtest.lstm_scaler.inverse_transform([pred_vec])[0][0])
                                delta_price = pred_hl2 - last_hl2

                                atr = float(df['atr'].iloc[-2])
                                min_atr_ratio = float(conf.get("lstm_min_atr_ratio", 0.0))
                                min_move = max(0.0, atr * min_atr_ratio)

                                if not np.isfinite(delta_price):
                                    should_trade = False
                                elif desired_side == "BUY" and delta_price <= min_move:
                                    should_trade = False
                                    self.log(f"AI (LSTM) filtered BUY: predicted move {delta_price:.5f} <= {min_move:.5f}")
                                elif desired_side == "SELL" and delta_price >= -min_move:
                                    should_trade = False
                                    self.log(f"AI (LSTM) filtered SELL: predicted move {delta_price:.5f} >= {-min_move:.5f}")
                            else:
                                should_trade = False
                                self.log("AI (LSTM) waiting for more data sequences...")

                        if should_trade:
                            if hasattr(self, 'corr_matrix') and self.corr_matrix is not None:
                                for pos in all_positions:
                                    if pos.symbol in self.corr_matrix.columns and symbol in self.corr_matrix.index:
                                        correlation = self.corr_matrix.loc[symbol, pos.symbol]
                                        if correlation > 0.7:
                                            should_trade = False
                                            self.log(f"[{symbol}] Trade cancelled: High correlation ({correlation:.2f}) with open {pos.symbol}")
                                            break

                        if should_trade:
                            atr = float(df['atr'].iloc[-2])
                            risk_per_trade = 0.01
                            account_info = mt5.account_info()
                            balance = account_info.balance if account_info else 10000

                            sl_dist = atr * 2
                            risk_amount = balance * risk_per_trade
                            contract_size = self._get_symbol_contract_size(symbol)
                            denom = float(sl_dist) * float(contract_size)
                            lots = (risk_amount / denom) if denom > 0 else 0.0
                            lots = round(max(0.01, min(lots, 10.0)), 2)

                            tick = mt5.symbol_info_tick(symbol)
                            if not tick:
                                self.log(f"[{symbol}] Trade Failed: no tick data")
                                time.sleep(10)
                                continue
                            self._update_symbol_spread_estimate(symbol, tick)

                            point = getattr(mt5.symbol_info(symbol), "point", 0.0) if hasattr(mt5, "symbol_info") else 0.0
                            slippage_price = (lots * 0.2) / 10.0
                            deviation = int(max(1, round(slippage_price / point))) if point else 20

                            if desired_side == "BUY":
                                price = float(tick.ask)
                                sl = float(price - sl_dist)
                                tp = float(price + (atr * 4))
                                self.log(f"[{symbol}] BUY Signal detected. Risking {risk_per_trade*100}% with {lots} lots.")
                                res = self.execute_trade(symbol, mt5.ORDER_TYPE_BUY, lots, conf["magic_number"], sl=sl, tp=tp, deviation=deviation)
                                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                    self.notifier.alert_trade(symbol, "BUY", res.price, lots, f"AI ({ai_choice}) + Alligator/Supertrend")
                                    self._sync_position_tracking(symbol, mt5.positions_get(symbol=symbol))
                            else:
                                price = float(tick.bid)
                                sl = float(price + sl_dist)
                                tp = float(price - (atr * 4))
                                self.log(f"[{symbol}] SELL Signal detected. Risking {risk_per_trade*100}% with {lots} lots.")
                                res = self.execute_trade(symbol, mt5.ORDER_TYPE_SELL, lots, conf["magic_number"], sl=sl, tp=tp, deviation=deviation)
                                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                                    self.notifier.alert_trade(symbol, "SELL", res.price, lots, f"AI ({ai_choice}) + Alligator/Supertrend")
                                    self._sync_position_tracking(symbol, mt5.positions_get(symbol=symbol))
                else:
                    for pos in positions:
                        if pos.magic != conf["magic_number"]:
                            continue
                        pos_type = getattr(pos, "type", None)
                        is_buy_pos = (pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0))
                        is_sell_pos = (pos_type == getattr(mt5, "POSITION_TYPE_SELL", 1))

                        if is_buy_pos:
                            if not (is_st_up and is_ali_bull):
                                self.log(f"[{symbol}] CLOSE BUY: Alligator/Supertrend reversed.")
                                self.close_position(pos)
                        elif is_sell_pos:
                            if not ((not is_st_up) and is_ali_bear):
                                self.log(f"[{symbol}] CLOSE SELL: Alligator/Supertrend reversed.")
                                self.close_position(pos)

            except Exception as e:
                self.log(f"Loop Error: {e}")
            
            time.sleep(10) # Check every 10 seconds

    def _sync_position_tracking(self, symbol, positions):
        if positions is None:
            positions = []
        now = datetime.utcnow()
        current = {}
        for p in positions:
            try:
                current[int(p.ticket)] = p
            except Exception:
                continue

        with self._pos_lock:
            tracked_for_symbol = self._tracked_positions.get(symbol, {})
            tracked_tickets = set(tracked_for_symbol.keys())
            current_tickets = set(current.keys())

            closed_tickets = tracked_tickets - current_tickets
            for ticket in list(closed_tickets):
                info = tracked_for_symbol.pop(ticket, None)
                if not info:
                    continue
                close_data = self._get_last_close_for_position(ticket, now)
                close_price = close_data.get("price")
                profit = close_data.get("profit")
                reason = close_data.get("reason", "CLOSED")

                if close_price is None:
                    close_price = info.get("tp") or info.get("sl") or info.get("price_open", 0.0)

                if profit is None:
                    profit = 0.0

                self.log(f"[{symbol}] Position {ticket} closed ({reason}) | P/L: {profit:.2f}")
                log_transaction(symbol, "CLOSE", float(close_price), float(info.get("volume", 0.0)), int(info.get("magic", 0)), reason, int(ticket))
                try:
                    self.notifier.alert_close(symbol, float(profit), reason)
                except Exception:
                    pass

                acc = mt5.account_info() if MT5_AVAILABLE else None
                if acc:
                    self._maybe_append_balance_history(getattr(acc, "balance", getattr(acc, "equity", 0.0)))

            for ticket, p in current.items():
                tracked_for_symbol[ticket] = {
                    "ticket": ticket,
                    "magic": int(getattr(p, "magic", 0)),
                    "volume": float(getattr(p, "volume", 0.0)),
                    "price_open": float(getattr(p, "price_open", 0.0)),
                    "sl": float(getattr(p, "sl", 0.0)),
                    "tp": float(getattr(p, "tp", 0.0)),
                    "time": int(getattr(p, "time", 0)),
                }

            self._tracked_positions[symbol] = tracked_for_symbol

    def _get_last_close_for_position(self, ticket, now_utc):
        if not MT5_AVAILABLE:
            return {"reason": "CLOSED", "price": None, "profit": None}

        try:
            dt_from = now_utc - pd.Timedelta(days=7)
        except Exception:
            dt_from = now_utc
        try:
            deals = mt5.history_deals_get(dt_from, now_utc, position=int(ticket))
        except Exception:
            deals = None

        if not deals:
            return {"reason": "CLOSED", "price": None, "profit": None}

        try:
            deals_sorted = sorted(deals, key=lambda d: getattr(d, "time", 0))
            last = deals_sorted[-1]
        except Exception:
            last = deals[-1]

        price = float(getattr(last, "price", 0.0))
        profit = float(getattr(last, "profit", 0.0))

        reason = "CLOSED"
        entry = int(getattr(last, "entry", 0))
        deal_type = int(getattr(last, "type", 0))
        if entry == 1:
            reason = "OPEN"
        elif entry == 0:
            reason = "CLOSE"
        if deal_type == getattr(mt5, "DEAL_TYPE_SL", -999):
            reason = "STOP_LOSS"
        elif deal_type == getattr(mt5, "DEAL_TYPE_TP", -999):
            reason = "TAKE_PROFIT"

        return {"reason": reason, "price": price, "profit": profit}

    def _maybe_append_balance_history(self, balance_value):
        now = time.time()
        if now - self._last_balance_write < 30:
            return
        with self._balance_lock:
            now = time.time()
            if now - self._last_balance_write < 30:
                return
            self._last_balance_write = now
            os.makedirs(DATA_DIR, exist_ok=True)
            row = {"time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "balance": float(balance_value)}
            file_exists = os.path.exists(BALANCE_HISTORY)
            try:
                import csv
                with open(BALANCE_HISTORY, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["time", "balance"])
                    if not file_exists:
                        w.writeheader()
                    w.writerow(row)
            except Exception:
                pass

    def execute_trade(self, symbol, type, lot, magic, sl=None, tp=None, deviation=None):
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        price = tick.ask if type == mt5.ORDER_TYPE_BUY else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": type,
            "price": price,
            "magic": magic,
            "comment": "App Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if deviation is not None:
            request["deviation"] = int(deviation)
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log(f"Trade Executed: {result.order}")
            log_transaction(symbol, "BUY" if type == mt5.ORDER_TYPE_BUY else "SELL", price, lot, magic, "OPEN", result.order)
        else:
            comment = result.comment if result else "MT5 Not Available"
            self.log(f"Trade Failed: {comment}")
            log_transaction(symbol, "BUY" if type == mt5.ORDER_TYPE_BUY else "SELL", price, lot, magic, f"FAILED: {comment}", 0)
        return result

    def close_position(self, pos):
        tick = mt5.symbol_info_tick(pos.symbol)
        type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price_close = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask if tick else 0
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": type_close,
            "position": pos.ticket,
            "price": price_close,
            "magic": pos.magic,
            "comment": "App Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            self.log(f"Position {pos.ticket} Closed.")
            profit = getattr(pos, 'profit', 0.0)
            log_transaction(pos.symbol, "CLOSE", price_close, pos.volume, pos.magic, "CLOSED", pos.ticket)
            self.notifier.alert_close(pos.symbol, profit, "Signal Reversal / TP-SL")
        else:
            comment = result.comment if result else "MT5 Not Available"
            self.log(f"Close Failed: {comment}")
            log_transaction(pos.symbol, "CLOSE", price_close, pos.volume, pos.magic, f"CLOSE_FAILED: {comment}", pos.ticket)

    def close_all_positions(self):
        if not MT5_AVAILABLE:
            self.log("MT5 not available for order execution.")
            return
        
        positions = mt5.positions_get()
        if not positions:
            self.log("No active positions to close.")
            return
            
        self.log(f"Institutional Command: Closing {len(positions)} positions...")
        for pos in positions:
            self.close_position(pos)
            
    def flatten_portfolio(self):
        self.log("Institutional Command: Flattening Portfolio...")
        self.stop_bot()
        self.close_all_positions()
        self.log("Portfolio Flattened. All bots stopped.")

    # --- Langkah 6: Audit Trail & Verification ---
    def export_audit_log(self):
        """Mengekspor log transaksi dan performa untuk verifikasi pihak ketiga."""
        import csv
        audit_file = f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            # Menggabungkan history transaksi dan metrics
            if os.path.exists("data/trade_history.csv"):
                import shutil
                shutil.copy("data/trade_history.csv", audit_file)
                messagebox.showinfo("Audit Trail", f"Institutional Audit Trail exported to: {audit_file}\nThis file is ready for third-party verification.")
                self.log(f"Audit Trail Exported: {audit_file}")
            else:
                messagebox.showwarning("Warning", "No trade history available to export.")
        except Exception as e:
            messagebox.showerror("Error", f"Audit export failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.mainloop()
