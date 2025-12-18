# Gaze Estimation Models

This directory contains the ONNX models required for the Rusty Eyes gaze estimation pipeline.

## Required Models

You must download the following models and place them in this directory (`models/`).

### 1. L2CS-Net (ResNet50)

- **Filename**: `l2cs_net.onnx`
- **Source**: [L2CS-Net Official Repo / HuggingFace](https://github.com/Ahmednull/L2CS-Net) or [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- **Description**: High accuracy gaze estimation using a ResNet50 backbone. Input is 224x224 cropped face.

### 2. MobileGaze (MobileNetV2)

- **Filename**: `mobile_gaze.onnx`
- **Source**: Varied (search for `L2CS MobileNet` or similar lightweight gaze models).
- **Description**: Lightweight version suitable for lower latency.

### 3. Face Detection & Mesh (Existing)

Already included in the repo:

- `face_detection.onnx` (UltraFace)
- `face_mesh.onnx` (MediaPipe Face Mesh)
- `head_pose.onnx` (WHENet)

## Usage

The application will automatically look for these files in the `models/` directory. If a file is missing, that specific gaze mode will be disabled.
