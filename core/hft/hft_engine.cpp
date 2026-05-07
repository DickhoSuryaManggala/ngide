#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

extern "C" {
    // Menghitung SMA dengan performa C++ (HFT Standard)
    void calculate_sma_cpp(const double* data, int size, int period, double* result) {
        for (int i = 0; i < size; ++i) {
            if (i < period - 1) {
                result[i] = 0.0;
                continue;
            }
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += data[i - j];
            }
            result[i] = sum / period;
        }
    }

    // Menghitung ATR dengan performa C++ untuk pemrosesan ribuan data secara instan
    void calculate_atr_cpp(const double* high, const double* low, const double* close, int size, int period, double* result) {
        std::vector<double> tr(size);
        tr[0] = high[0] - low[0];
        for (int i = 1; i < size; ++i) {
            double hl = high[i] - low[i];
            double hpc = std::abs(high[i] - close[i - 1]);
            double lpc = std::abs(low[i] - close[i - 1]);
            tr[i] = std::max({hl, hpc, lpc});
        }

        // Simple Moving Average dari TR
        for (int i = 0; i < size; ++i) {
            if (i < period - 1) {
                result[i] = 0.0;
                continue;
            }
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += tr[i - j];
            }
            result[i] = sum / period;
        }
    }
}
