# DeepStream Gaze Tracking Server

GPU-accelerated gaze tracking and face mesh inference using NVIDIA DeepStream.

## Architecture

```
Jetson Nano (Camera Edge)
    ↓ H.264 RTP/UDP (Port 5004)
DGX Server (Inference)
    ↓ MQTT/Redis (Metadata)
Rust Client (Visualization)
```

## Quick Start

### 1. Server Setup (DGX/x86_64)

```bash
cd deepstream
sudo ./setup_dgx.sh
```

This installs:
- GStreamer 1.0 + H.264 plugins
- Docker + NVIDIA Container Toolkit
- DeepStream 7.0 container

### 2. Start Camera Stream (Jetson)

```bash
# On jetsone
cd ~/dev/repos/rustyeyes3
./scripts/stream_camera.sh jowestdgxe 5004 /dev/video0
```

### 3. Run DeepStream Inference (DGX)

```bash
# On jowestdgxe
cd ~/deepstream
./run_face_pipeline.sh
```

## Configuration Files

- `config_face_mesh.txt` - DeepStream pipeline config
- `setup_dgx.sh` - Dependency installation script
- `run_face_pipeline.sh` - Container launch script

## Dependencies

**Server (DGX):**
- Ubuntu 20.04/22.04
- NVIDIA GPU with CUDA 12.0+
- Docker + NVIDIA Container Toolkit
- GStreamer 1.0

**Edge (Jetson):**
- JetPack 4.6+ (Ubuntu 18.04)
- GStreamer 1.0 with NVENC

## Troubleshooting

**Stream not received:**
```bash
# Verify UDP port open
sudo ufw allow 5004/udp

# Test reception manually
~/receive_stream_dgx.sh 5004
```

**DeepStream errors:**
```bash
# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
