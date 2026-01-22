#!/bin/bash
set -e

# 1. Config Git (for Stash)
git config user.email 'deploy@bot.local'
git config user.name 'Deploy Bot'

# 2. Update Repo
echo "[SETUP] Updating Repo..."
git stash
git pull

# 3. Setup Environment (Cargo)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# 4. Setup ONNX Runtime (v1.15.1 for Ubuntu 18.04 compat)
ORT_VERSION="1.15.1"
ORT_DIR="$HOME/ort-${ORT_VERSION}"
if [ ! -d "$ORT_DIR" ]; then
    echo "[SETUP] Downloading ONNX Runtime v${ORT_VERSION}..."
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
    tar -xzf onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
    mv onnxruntime-linux-aarch64-${ORT_VERSION} "$ORT_DIR"
    rm onnxruntime-linux-aarch64-${ORT_VERSION}.tgz
fi

# Export for both Build (if needed) and Runtime
export ORT_DYLIB_PATH="$ORT_DIR/lib/libonnxruntime.so"
export LD_LIBRARY_PATH="$ORT_DIR/lib:$LD_LIBRARY_PATH"
echo "[SETUP] ORT_DYLIB_PATH=$ORT_DYLIB_PATH"

# 5. Build
echo "[SETUP] Building..."
# Ensure we clean if switching strategies, or just build
# cargo clean # Optional, might differ
cargo build --release --bin rusty-eyes

# 6. Run
echo "[SETUP] Running..."
export DISPLAY=:0
# Pass environment to run.sh if it handles it, or just run binary directly
# Assuming run.sh launches the binary
bash run.sh
