#!/bin/bash
set -e

# Rusty Eyes Startup Script

echo "=== Rusty Eyes Launcher ==="

# 1. Check Python Environment
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    python3 -m venv venv
    
    echo "[SETUP] Installing dependencies..."
    ./venv/bin/pip install -r scripts/requirements.txt
else
    echo "[SETUP] Virtual environment found."
    # Optional: check if we need to update deps? 
    # For now, let's assume if venv exists, it's good. 
    # If user wants to force update, they can delete venv.
fi

# 2. Build and Run Rust Application
echo "[RUN] Starting Rusty Eyes..."
# We use --release for performance, especially for video processing
cargo run --release
