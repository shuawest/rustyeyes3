#!/bin/bash
set -e

# Detect OS
OS_TYPE=$(uname)
echo "[SETUP] Detected OS: $OS_TYPE"

# Ensure Cargo is in PATH (common issue in non-interactive SSH)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if [ "$OS_TYPE" = "Linux" ]; then
    # Check for GStreamer dev dependencies
    if ! pkg-config --exists gstreamer-1.0; then
        echo "[SETUP] Missing GStreamer dev libraries. Please run:"
        echo "sudo apt-get update && sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio"
        # We don't auto-run sudo for safety/interactivity reasons usually, but user is on Jetson.
        # Let's just warn and exit if critical build deps are missing? Or try to continue?
        # The user reported build failure, so we must fix it.
    fi

    echo "[SETUP] Linux detected. Building Overlay..."
    
    # Set RUSTFLAGS to avoid linker issues on older systems (Ubuntu 18.04)
    # The --gc-sections flag with newer metadata causes issues with binutils 2.30
    export RUSTFLAGS="-C link-arg=-Wl,--no-eh-frame-hdr"
    
    cargo build --release --bin overlay_linux
    
    # Symlink or Copy to ./overlay_app (what main.rs expects)
    cp target/release/overlay_linux ./overlay_app
    chmod +x ./overlay_app
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "[SETUP] macOS detected. using existing overlay logic or prebuilt."
    # If Swift overlay needs building, do it here.
    # For now assume user has it or it's built elsewhere.
fi

# 1. Check Python Environment
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    python3 -m venv venv
    
    echo "[SETUP] Upgrading pip and installing build dependencies..."
    ./venv/bin/pip install --upgrade pip setuptools wheel
    ./venv/bin/pip install scikit-build cmake numpy
    
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

# Check if running on Ubuntu 18.04 (old glibc, ONNX Runtime incompatible)
CARGO_FEATURES=""
if [ "$OS_TYPE" = "Linux" ]; then
    if command -v lsb_release &> /dev/null; then
        UBUNTU_VERSION=$(lsb_release -rs)
        if [ "$UBUNTU_VERSION" = "18.04" ]; then
            echo "[INFO] Ubuntu 18.04 detected - building without ONNX Runtime due to glibc incompatibility"
            CARGO_FEATURES="--no-default-features"
        fi
    fi
fi

cargo run --release $CARGO_FEATURES
