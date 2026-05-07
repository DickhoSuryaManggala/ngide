import json
import os
from datetime import datetime

class RiskManager:
    """
    Sistem 'Circuit Breaker' untuk Institusi AI.
    Memantau kerugian harian dan total drawdown untuk mencegah kehancuran modal.
    """
    def __init__(self, config_file="data/risk_config.json"):
        self.config_file = config_file
        self.default_config = {
            "max_daily_loss_pct": 2.0,    # Berhenti jika rugi > 2% dalam sehari
            "max_total_drawdown_pct": 10.0, # Berhenti jika total drawdown > 10%
            "max_trades_per_day": 10,     # Batasi jumlah trade harian
            "circuit_breaker_active": False,
            "last_reset_date": str(datetime.now().date())
        }
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                return json.load(f)
        return self.default_config

    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def check_daily_reset(self):
        current_date = str(datetime.now().date())
        if self.config["last_reset_date"] != current_date:
            self.config["last_reset_date"] = current_date
            self.config["circuit_breaker_active"] = False
            self.save_config()
            return True
        return False

    def is_trading_allowed(self, current_daily_loss_pct, current_drawdown_pct, trades_today):
        self.check_daily_reset()
        
        if self.config["circuit_breaker_active"]:
            return False, "Circuit Breaker is ACTIVE (Manual reset required or next day)"

        if current_daily_loss_pct >= self.config["max_daily_loss_pct"]:
            self.config["circuit_breaker_active"] = True
            self.save_config()
            return False, f"Daily Loss Limit Exceeded ({current_daily_loss_pct:.2f}%)"

        if current_drawdown_pct >= self.config["max_total_drawdown_pct"]:
            self.config["circuit_breaker_active"] = True
            self.save_config()
            return False, f"Total Drawdown Limit Exceeded ({current_drawdown_pct:.2f}%)"

        if trades_today >= self.config["max_trades_per_day"]:
            return False, f"Max Daily Trades Reached ({trades_today})"

        return True, "Trading Allowed"

if __name__ == "__main__":
    rm = RiskManager()
    print("Risk Manager Initialized.")
    allowed, msg = rm.is_trading_allowed(0.5, 2.0, 3)
    print(f"Status: {allowed}, Message: {msg}")
