#!/bin/bash
set -e

# Ensure models directory exists
mkdir -p models

echo "=========================================="
echo " Rusty Eyes 3: Setup Models"
echo "=========================================="

echo "Downloading Face Tracking Models..."

# 1. Face Mesh (MediaPipe)
if [ -f "models/face_mesh.onnx" ]; then
    echo "[SKIP] models/face_mesh.onnx already exists."
else
    echo "Downloading Face Mesh..."
    curl -L -o models/face_mesh.onnx "https://huggingface.co/hayashiLin/deepfacelivemodels/resolve/main/FaceMesh.onnx"
fi

# 2. Face Detection (UltraFace RFB-320)
if [ -f "models/face_detection.onnx" ]; then
    echo "[SKIP] models/face_detection.onnx already exists."
else
    echo "Downloading Face Detection..."
    curl -L -o models/face_detection.onnx "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/raw/master/models/onnx/version-RFB-320.onnx"
fi

# 3. Head Pose (WHENet)
if [ -f "models/head_pose.onnx" ]; then
    echo "[SKIP] models/head_pose.onnx already exists."
else
    echo "Downloading Head Pose..."
    curl -L -o models/head_pose.onnx "https://huggingface.co/ykk648/face_lib/resolve/main/head_pose/WHENet.onnx"
fi

echo ""
echo "Downloading Gaze Estimation Models..."

# 4. L2CS-Net (ResNet50)
if [ -f "models/l2cs_net.onnx" ]; then
    echo "[SKIP] models/l2cs_net.onnx already exists."
else
    echo "Downloading L2CS-Net (ResNet50)..."
    curl -L -o models/l2cs_net.onnx "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet50_gaze.onnx"
fi

# 5. MobileGaze (MobileNetV2)
if [ -f "models/mobile_gaze.onnx" ]; then
    echo "[SKIP] models/mobile_gaze.onnx already exists."
else
    echo "Downloading MobileGaze (MobileNetV2)..."
    curl -L -o models/mobile_gaze.onnx "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx"
fi

echo "=========================================="
echo " All models present in 'models/' directory."
echo " You are ready to run:"
echo " cargo run --release"
echo "=========================================="
