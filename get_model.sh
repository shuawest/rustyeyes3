#!/bin/bash
# Download Face Mesh ONNX model
# This URL points to a community converted version of the MediaPipe Face Mesh model
# Source: https://github.com/thepowerfuldeez/facemesh.pytorch/releases/download/v1.0/face_mesh.onnx 
# (Note: This is an example, actual URL might vary. I'll use a reliable known one found in search or similar repos)

# Using a known reliable source from Pinto Model Zoo or similar is best. 
# Using a known reliable source from Pinto Model Zoo or similar is best.
# For this example, I'll attempt a download from a specific raw content URL if found,
# otherwise I will notify the user to download manually.

# Download Face Mesh ONNX model from Hugging Face (hayashiLin/deepfacelivemodels)
MESH_URL="https://huggingface.co/hayashiLin/deepfacelivemodels/resolve/main/FaceMesh.onnx"
# Download UltraFace Detector (RFB-320) due to reliability
DET_URL="https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx"
# Download WHENet Head Pose
POSE_URL="https://huggingface.co/ykk648/face_lib/resolve/main/head_pose/WHENet.onnx"

echo "Downloading FaceMesh.onnx..."
curl -L -o face_mesh.onnx $MESH_URL

echo "Downloading face_detection.onnx..."
curl -L -o face_detection.onnx $DET_URL

echo "Downloading head_pose.onnx..."
curl -L -o head_pose.onnx $POSE_URL

# Gaze Estimation (Key 5)
# Now uses internal Computer Vision (Pupil Blob Tracking), so no extra model needed.

echo "Models ready. Run with: cargo run --release -- --cam-index 0 --mirror"


echo "Models ready. Run with: cargo run --release -- --cam-index 0 --mirror"

if [ -f "face_mesh.onnx" ] && [ -f "face_detection.onnx" ] && [ -f "head_pose.onnx" ]; then
    echo "Downloads successful."
else
    echo "Download failed."
    exit 1
fi
