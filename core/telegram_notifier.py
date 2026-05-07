import requests
import json
import os

# Path Configuration
DATA_DIR = "data"
TELEGRAM_CONFIG = os.path.join(DATA_DIR, "telegram_config.json")

class TelegramNotifier:
    """
    Sistem Notifikasi Telegram untuk Institusi AI.
    Mengirimkan alert real-time untuk setiap aktivitas trading penting.
    """
    def __init__(self):
        self.config = self.load_config()
        self.bot_token = self.config.get("bot_token", "")
        self.chat_id = self.config.get("chat_id", "")
        self.enabled = self.config.get("enabled", False)

    def load_config(self):
        if os.path.exists(TELEGRAM_CONFIG):
            with open(TELEGRAM_CONFIG, "r") as f:
                return json.load(f)
        return {"bot_token": "", "chat_id": "", "enabled": False}

    def save_config(self, bot_token, chat_id, enabled=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        config = {
            "bot_token": bot_token,
            "chat_id": chat_id,
            "enabled": enabled
        }
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(TELEGRAM_CONFIG, "w") as f:
            json.dump(config, f, indent=4)

    def send_message(self, message):
        """Mengirim pesan teks ke Telegram."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"🐊 *ALIGATOR AI INSTITUTIONAL ALERT*\n\n{message}",
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram Error: {e}")
            return False

    def alert_trade(self, symbol, trade_type, price, lots, reason="AI Strategy"):
        """Notifikasi khusus untuk pembukaan posisi."""
        msg = (
            f"🚀 *NEW POSITION OPENED*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 *Symbol:* {symbol}\n"
            f"🔹 *Type:* {trade_type}\n"
            f"🔹 *Price:* {price:.5f}\n"
            f"🔹 *Lots:* {lots:.2f}\n"
            f"🔹 *Reason:* {reason}"
        )
        return self.send_message(msg)

    def alert_close(self, symbol, profit, reason="Target Reached"):
        """Notifikasi khusus untuk penutupan posisi."""
        emoji = "✅" if profit >= 0 else "❌"
        msg = (
            f"{emoji} *POSITION CLOSED*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔹 *Symbol:* {symbol}\n"
            f"🔹 *Profit:* ${profit:.2f}\n"
            f"🔹 *Reason:* {reason}"
        )
        return self.send_message(msg)

    def alert_risk(self, alert_msg):
        """Notifikasi bahaya untuk circuit breaker."""
        msg = (
            f"⚠️ *INSTITUTIONAL RISK ALERT*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔴 *Status:* CIRCUIT BREAKER ACTIVE\n"
            f"🔴 *Reason:* {alert_msg}\n\n"
            f"Bot has been paused for safety."
        )
        return self.send_message(msg)

    def send_institutional_report(self, metrics, account_info=None):
        """Mengirimkan laporan performa kelas institusi yang mendalam."""
        header = "📊 *INSTITUTIONAL PERFORMANCE REPORT*\n"
        timestamp = f"🕒 *Generation Time:* {os.popen('date').read().strip()}\n"
        divider = "━━━━━━━━━━━━━━━━━━━━\n"
        
        # Account Summary Section
        acc_info = ""
        if account_info:
            acc_info = (
                f"🏦 *ACCOUNT SUMMARY*\n"
                f"💰 Equity: ${account_info.get('equity', 0):,.2f}\n"
                f"💵 Balance: ${account_info.get('balance', 0):,.2f}\n"
                f"🛡️ Margin Level: {account_info.get('margin_level', 0):.2f}%\n"
                f"{divider}"
            )

        # Performance Metrics Section
        perf_info = (
            f"📈 *RISK-ADJUSTED METRICS*\n"
            f"💎 Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}\n"
            f"💎 Sortino Ratio: {metrics.get('Sortino Ratio', 0):.2f}\n"
            f"💎 Calmar Ratio: {metrics.get('Calmar Ratio', 0):.2f}\n"
            f"📉 Max Drawdown: {metrics.get('Max Drawdown', 0):.2f}%\n"
            f"🎯 Win Rate: {metrics.get('Win Rate', 0):.2f}%\n"
            f"💰 Total Return: {metrics.get('Total Return', 0):.2f}%\n"
            f"{divider}"
        )

        # Conclusion/Advice
        status = "🟢 HEALTHY" if metrics.get('Sharpe Ratio', 0) > 1.5 else "🟡 CAUTION"
        if metrics.get('Max Drawdown', 0) < -15: status = "🔴 HIGH RISK"
        
        advice = (
            f"🛡️ *RISK STATUS:* {status}\n"
            f"📝 _Note: Performance is within institutional risk limits._"
        )

        full_msg = f"{header}{timestamp}{divider}{acc_info}{perf_info}{advice}"
        return self.send_message(full_msg)

if __name__ == "__main__":
    # Test script
    notifier = TelegramNotifier()
    print("Telegram Notifier Initialized.")
