#!/bin/bash
# scripts/receive_stream_dgx.sh
# Test script to receive and decode UDP H.264 stream on DGX

PORT=${1:-5004}

echo "Receiving H.264 RTP stream on port ${PORT}..."

# For x86_64, use avdec_h264 (software) or omxh264dec if available
# We'll use avdec for compatibility
gst-launch-1.0 -v \
    udpsrc port="${PORT}" caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! \
    rtph264depay ! \
    h264parse ! \
    avdec_h264 ! \
    videoconvert ! \
    fpsdisplaysink sync=false
