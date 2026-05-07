#!/bin/bash

# Nama direktori virtual environment
VENV_NAME="venv_aligator"

echo "Mengecek apakah dependensi sistem (venv & tkinter) sudah terinstal..."
if ! dpkg -l | grep -q python3-venv || ! dpkg -l | grep -q python3-tk; then
    echo "Dependensi sistem belum lengkap. Menginstal python3-venv dan python3-tk..."
    sudo apt update && sudo apt install -y python3-venv python3-tk
fi

echo "Membuat virtual environment: $VENV_NAME..."
python3 -m venv $VENV_NAME

echo "Mengaktifkan virtual environment dan menginstal dependensi..."
source $VENV_NAME/bin/activate
pip install --upgrade pip

# Instal satu per satu agar jika MT5 gagal, yang lain tetap terinstal
for pkg in pandas numpy scikit-learn joblib yfinance MetaTrader5; do
    echo "Menginstal $pkg..."
    pip install $pkg || echo "Gagal menginstal $pkg (mungkin karena OS tidak mendukung)"
done

echo "------------------------------------------------"
echo "Setup selesai! Gunakan ./run_app.sh untuk menjalankan aplikasi."
echo "------------------------------------------------"
