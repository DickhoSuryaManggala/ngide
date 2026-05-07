import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Gunakan backend non-GUI agar aman di threading
import matplotlib.pyplot as plt
import os
import json

# Path Configuration
DATA_DIR = "data"
BACKTEST_RESULTS = os.path.join(DATA_DIR, "backtest_results.csv")

def run_monte_carlo(csv_file=BACKTEST_RESULTS, simulations=1000):
    """
    Menjalankan simulasi Monte Carlo berdasarkan hasil backtest terakhir.
    Ini mensimulasikan 1000 skenario berbeda dengan mengacak urutan trade yang ada
    untuk melihat kemungkinan terburuk (Maximum Drawdown) di masa depan.
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} tidak ditemukan. Jalankan backtest terlebih dahulu.")
        return

    if os.path.getsize(csv_file) == 0:
        print(f"Error: {csv_file} kosong. Jalankan backtest terlebih dahulu.")
        return

    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print(f"Error: {csv_file} kosong atau tidak memiliki kolom. Jalankan backtest terlebih dahulu.")
        return
    except Exception as e:
        print(f"Error: Gagal membaca {csv_file}: {e}")
        return

    if 'profit' not in df.columns:
        print(f"Error: Kolom 'profit' tidak ditemukan di {csv_file}. Pastikan file backtest_results.csv valid.")
        return

    df = df.dropna(subset=['profit'])
    
    if len(df) < 5:
        print("Data trade terlalu sedikit untuk simulasi Monte Carlo.")
        return

    profits = df['profit'].values
    initial_balance = 10000
    
    plt.figure(figsize=(12, 6))
    
    final_balances = []
    max_drawdowns = []

    print(f"--- Menjalankan {simulations} Simulasi Monte Carlo ---")

    for i in range(simulations):
        # Acak urutan profit trade (Resampling with replacement)
        sim_profits = np.random.choice(profits, size=len(profits), replace=True)
        
        # Hitung kurva ekuitas simulasi
        equity_curve = initial_balance + np.cumsum(sim_profits)
        final_balances.append(equity_curve[-1])
        
        # Hitung Drawdown simulasi
        cum_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cum_max) / cum_max
        max_drawdowns.append(np.min(drawdown) * 100)
        
        # Plot 50 garis pertama agar tidak terlalu penuh
        if i < 50:
            plt.plot(equity_curve, color='blue', alpha=0.1)

    # Statistik Hasil Simulasi
    final_balances = np.array(final_balances)
    max_drawdowns = np.array(max_drawdowns)
    
    # Institutional Risk Metrics
    mean_final = np.mean(final_balances)
    prob_profit = (final_balances > initial_balance).mean() * 100
    worst_drawdown = np.min(max_drawdowns)
    avg_drawdown = np.mean(max_drawdowns)
    var_95 = initial_balance - np.percentile(final_balances, 5)

    stats = {
        "Mean Final Balance": f"${mean_final:,.2f}",
        "Profit Probability": f"{prob_profit:.1f}%",
        "Worst Case Drawdown": f"{worst_drawdown:.2f}%",
        "Average Drawdown": f"{avg_drawdown:.2f}%",
        "Value at Risk (95%)": f"${var_95:,.2f}"
    }
    
    os.makedirs("assets/reports", exist_ok=True)
    os.makedirs("assets/plots", exist_ok=True)

    with open("assets/reports/monte_carlo_report.json", "w") as f:
        json.dump(stats, f, indent=4)

    plt.axhline(y=initial_balance, color='red', linestyle='--', label='Initial Balance')
    plt.title(f"Monte Carlo Simulation ({simulations} paths)")
    plt.xlabel("Number of Trades")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("assets/plots/monte_carlo_plot.png")
    plt.close()

if __name__ == "__main__":
    run_monte_carlo()
