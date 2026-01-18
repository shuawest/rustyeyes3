#!/bin/bash
# deepstream/setup_dgx.sh
# Automated setup script for DeepStream server dependencies on DGX/x86_64 systems

set -e

echo "=== DeepStream Server Setup ==="
echo "This script installs dependencies for DeepStream-based gaze tracking on x86_64 systems"

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

echo ""
echo "Step 1: Updating package lists..."
apt-get update

echo ""
echo "Step 2: Installing GStreamer and plugins..."
apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

echo ""
echo "Step 3: Verifying Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    echo "Docker already installed: $(docker --version)"
fi

echo ""
echo "Step 4: Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-container-toolkit &> /dev/null; then
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    systemctl restart docker
else
    echo "NVIDIA Container Toolkit already installed"
fi

echo ""
echo "Step 5: Pulling DeepStream Docker image..."
docker pull nvcr.io/nvidia/deepstream:7.0-triton-multiarch

echo ""
echo "=== Setup Complete ==="
echo "Installed:"
echo "  - GStreamer 1.0 + plugins (for H.264 RTP reception)"
echo "  - Docker + NVIDIA Container Toolkit (for DeepStream)"
echo "  - DeepStream 7.0 container (for GPU-accelerated inference)"
echo ""
echo "Next steps:"
echo "  1. Ensure firewall allows UDP port 5004 (incoming stream)"
echo "  2. Run: ./run_face_pipeline.sh"
