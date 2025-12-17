# AGENTS.md

> [!NOTE]
> This file acts as a briefing packet for AI coding agents. It contains project specifications, build instructions, and architectural context.

## 1. Project Overview

**RustyEyes3** is a high-performance, Privacy-First Computer Vision application written in Rust. It performs real-time face tracking, mesh generation, head pose estimation, and gaze tracking using lightweight ONNX models. The application is designed to run locally on macOS (and other platforms) without cloud dependencies.

## 2. Quick Start for Agents

### Build

```bash
# Ensure models are downloaded
./get_model.sh

# Build with release profile (compiles swift sidecar automatically on macOS)
cargo build --release
```

### Run

```bash
# List cameras
cargo run --release --bin rusty-eyes -- --list

# Run with camera 0 and mirror mode
cargo run --release --bin rusty-eyes -- --cam-index 0 --mirror
```

### Tech Stack

- **Language**: Rust 2021
- **Computer Vision**: ONNX Runtime (`ort`) with CoreML support
- **Camera**: `nokhwa`
- **Windowing**: `minifb`
- **Overlay**: Swift (macOS Sidecar)

## 3. Functional Requirements

### 3.1. Camera Input

- **Enumeration**: The application shall list all available video input devices connected to the system.
- **Selection**: The user shall be able to select a specific camera via CLI argument (e.g., index 0, 1).
- **Format**: The system shall inspect and utilize the highest available framerate/resolution (preference for 60fps+).
- **Mirroring**: The application shall support a "Mirror Mode" (Horizontal Flip) for intuitive self-view interaction.

### 3.2. Computer Vision Pipelines

The system shall support runtime-switchable pipelines:

#### 3.2.1. Face Detection (Key 2)

- **Model**: UltraFace (ONNX).
- **Function**: Detect faces in the frame and draw bounding boxes.
- **Output**: Green bounding box around detected faces.

#### 3.2.2. Face Mesh (Key 1)

- **Model**: Face Mesh (468 landmarks, ONNX, based on MediaPipe).
- **Function**: Track 468 facial landmarks in 3D space.
- **Output**: Real-time overlay of 468 red points on the face.
- **Performance goal**: >30 FPS (Optimized: >120 FPS achieved).

#### 3.2.3. Head Pose estimation (Key 3)

- **Model**: WHENet (ONNX).
- **Function**: Estimate Yaw, Pitch, and Roll angles of the head.
- **Output**: Center HUD visualization showing orientation via a directional pointer.

#### 3.2.4. Head Gaze (Key 4)

- **Type**: Geometric Simulation.
- **Function**: Estimate gaze direction using Head Pose orientation combined with geometrical eye center positions from the Face Mesh.
- **Output**: Blue Eye Centers + Cyan Gaze Rays projected from eyes.

#### 3.2.5. Pupil Gaze (Key 5)

- **Type**: Computer Vision / Hybrid.
- **Function**: Track pupil position _within_ the eye using pixel-intensity blob tracking (finding the darkest region) combined with Head Pose.
- **Benefit**: Provides higher precision tracking of eye movements independent of head rotation, without requiring heavy/unavailable external gaze models.
- **Fallback**: Robust inputs (Mesh + CV) ensure availability without complex dependencies.

### 3.3. User Interface (UI)

- **Windowing**: A native window displaying the video feed with overlaid graphics.
- **Controls**:
  - `1-5`: Switch Pipelines instantaneously.
  - `6`: Toggle Overlay (macOS Only).
  - `ESC`: Quit application.
- **Visuals**:
  - Clear, high-contrast overlays (Red for Mesh, Green for Box, Cyan for Gaze).
  - Minimal latency rendering.

## 4. Technical Requirements

### 4.1. Performance

- **Language**: Rust (Safe, Fast).
- **Inference Engine**: `ort` (ONNX Runtime) with CoreML/CPU Execution Providers for hardware acceleration on Apple Silicon.
- **Framerate**: The pipeline overhead shall not reduce the camera's native framerate significantly (Target: 60+ FPS processing).

### 4.2. Dependency Management

- **Model Management**: A shell script (`get_model.sh`) shall automatically fetch required ONNX models from reliable public sources (e.g., PINTO Model Zoo) to keep the repository size small.
- **Crate Dependencies**:
  - `nokhwa`: Cross-platform Camera capture.
  - `ort`: ONNX Runtime bindings.
  - `minifb`: Minimal windowing.
  - `image`: Image processing.

## 5. Constraints & Assumptions

- **OS**: Primary target macOS (Apple Silicon), but code should be cross-platform compatible (Linux/Windows).
- **Models**: Must be available in ONNX format.
- **Tooling**: `swiftc` required for macOS overlay integration.
- **Privacy**: All processing must happen locally; no video data sent to cloud.
