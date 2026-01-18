#!/bin/bash
# deepstream/run_gstreamer_inference.sh  
# Direct GStreamer pipeline with TensorRT inference (GB10 compatible)

PORT=${1:-5004}

echo "Starting GStreamer inference pipeline on port ${PORT}..."

# For now, test basic pipeline without inference to verify decode works
# We'll add nvinfer once we have face detection model configured

gst-launch-1.0 -v \
    udpsrc port="${PORT}" caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! \
    rtph264depay ! \
    h264parse ! \
    avdec_h264 ! \
    videoconvert ! \
    fpsdisplaysink text-overlay=false video-sink=fakesink sync=false
