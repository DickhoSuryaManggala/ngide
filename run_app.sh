#!/bin/bash

VENV_NAME="venv_aligator"

if [ ! -d "$VENV_NAME" ]; then
    echo "Virtual environment tidak ditemukan. Menjalankan setup..."
    ./setup_env.sh
fi

echo "Menjalankan Aligator Trading App (Hybrid Engine)..."
source $VENV_NAME/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# Jalankan Ruby Automation di background (Maintenance)
ruby core/automation/automation.rb &

python3 core/trading_app.py
