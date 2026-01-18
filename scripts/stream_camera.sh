#!/bin/bash
# stream_camera.sh
# Streams camera video from Jetson Nano to a remote receiver (DGX) using GStreamer hardware encoding.
# Usage: ./stream_camera.sh [TARGET_IP] [TARGET_PORT] [DEVICE]

TARGET_IP=${1:-"192.168.1.10"} # Default to DGX IP if known, user should override
TARGET_PORT=${2:-5004}
DEVICE=${3:-"/dev/video0"}

echo "Starting Hardware Encoded Stream to ${TARGET_IP}:${TARGET_PORT} from ${DEVICE}..."

# Pipeline Breakdown:
# 1. v4l2src: Capture from USB camera (MJPG required for 720p@30 on this device)
# 2. jpegdec: Software decode MJPG to Raw YUV (I420)
# 3. nvvidconv: Upload Raw YUV to NVMM (NV12)
# 4. nvv4l2h264enc: Hardware H.264 encode
# 5. udpsink: Send via UDP

gst-launch-1.0 -v \
    v4l2src device="${DEVICE}" ! \
    "image/jpeg, width=1280, height=720, framerate=30/1" ! \
    jpegdec ! \
    nvvidconv ! \
    "video/x-raw(memory:NVMM), format=NV12" ! \
    nvv4l2h264enc preset-level=4 insert-sps-pps=true maxperf-enable=1 iframeinterval=30 ! \
    h264parse ! \
    rtph264pay config-interval=1 pt=96 ! \
    udpsink host="${TARGET_IP}" port="${TARGET_PORT}" sync=false async=false
