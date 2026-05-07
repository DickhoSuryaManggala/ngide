import os
import time
import pandas as pd
from core import ai_trainer
from core import lstm_trainer
from core import backtest
from datetime import datetime, timedelta

class WalkForwardOptimizer:
    """
    Sistem Otomatisasi Institusi untuk melatih ulang AI (Re-training).
    Menjamin AI selalu adaptif dengan kondisi pasar terbaru.
    """
    def __init__(self, data_file="data/data_xauusd.csv", interval_days=7):
        self.data_file = data_file
        self.interval_days = interval_days
        self.last_train_file = "data/last_train_info.txt"

    def needs_retraining(self):
        """Cek apakah sudah waktunya melatih ulang (misal: seminggu sekali)."""
        if not os.path.exists(self.last_train_file):
            return True
            
        with open(self.last_train_file, "r") as f:
            last_train_str = f.read().strip()
            last_train_date = datetime.strptime(last_train_str, "%Y-%m-%d")
            
        if datetime.now() - last_train_date > timedelta(days=self.interval_days):
            return True
        return False

    def run_optimization_cycle(self, progress_callback=None):
        """Menjalankan siklus lengkap: Re-training RF -> Re-training LSTM -> Validation."""
        print(f"\n--- Memulai Siklus Walk-Forward Optimization: {datetime.now()} ---")
        
        try:
            def report(percent, message):
                if progress_callback is None:
                    return
                try:
                    progress_callback(int(percent), str(message))
                except Exception:
                    pass

            # 1. Re-train Random Forest
            print("\n[1/3] Re-training Random Forest Classifier...")
            report(10, "Re-training Random Forest...")
            ai_trainer.train_advanced_ai()
            
            # 2. Re-train LSTM Deep Learning
            print("\n[2/3] Re-training LSTM Neural Network...")
            report(45, "Re-training LSTM...")
            lstm_trainer.train_lstm_ai()
            
            # 3. Validation Backtest
            print("\n[3/3] Running Validation Backtest on latest data...")
            report(75, "Running validation backtest...")
            # Load fresh data
            df = pd.read_csv(self.data_file)
            df.columns = [c.lower() for c in df.columns]
            
            # Ambil 30% data terakhir untuk validasi
            val_size = int(len(df) * 0.3)
            val_df = df.tail(val_size)
            
            # Load models to backtest
            backtest.load_ai_models()
            
            with open('data/config.json', 'r') as f:
                import json
                conf = json.load(f)
                
            print("\nValidation Results (LSTM):")
            backtest.event_driven_backtest(val_df, conf, use_ai=True, ai_type='lstm')
            
            # Update last training date
            with open(self.last_train_file, "w") as f:
                f.write(str(datetime.now().date()))
                
            print("\n--- Walk-Forward Cycle Completed Successfully ---")
            report(100, "Completed.")
            return True
            
        except Exception as e:
            print(f"Error during Walk-Forward Optimization: {e}")
            try:
                if progress_callback is not None:
                    progress_callback(0, f"Failed: {e}")
            except Exception:
                pass
            return False

if __name__ == "__main__":
    wfo = WalkForwardOptimizer(interval_days=0) # Set 0 to force run for test
    wfo.run_optimization_cycle()
