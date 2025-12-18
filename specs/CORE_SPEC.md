# Core Specification: RustyEyes3

## 1. Functional Requirements

### 1.1. Camera Input

- **Enumeration**: The application shall list all available video input devices connected to the system.
- **Selection**: The user shall be able to select a specific camera via CLI argument (e.g., index 0, 1).
- **Format**: The system shall inspect and utilize the highest available framerate/resolution (preference for 60fps+).
- **Mirroring**: The application shall support a "Mirror Mode" (Horizontal Flip) for intuitive self-view interaction.

### 1.2. Computer Vision Pipelines

The system shall support runtime-switchable pipelines:

#### 1.2.1. Face Detection (Key 2)

- **Model**: UltraFace (ONNX).
- **Function**: Detect faces in the frame and draw bounding boxes.
- **Output**: Green bounding box around detected faces.

#### 1.2.2. Face Mesh (Key 1)

- **Model**: Face Mesh (468 landmarks, ONNX, based on MediaPipe).
- **Function**: Track 468 facial landmarks in 3D space.
- **Output**: Real-time overlay of 468 red points on the face.
- **Performance goal**: >30 FPS (Optimized: >120 FPS achieved).

#### 1.2.3. Head Pose estimation (Key 3)

- **Model**: WHENet (ONNX).
- **Function**: Estimate Yaw, Pitch, and Roll angles of the head.
- **Output**: Center HUD visualization showing orientation via a directional pointer.

#### 1.2.4. Head Gaze (Key 4)

- **Type**: Geometric Simulation.
- **Function**: Estimate gaze direction using Head Pose orientation combined with geometrical eye center positions from the Face Mesh.
- **Output**: Blue Eye Centers + Cyan Gaze Rays projected from eyes.

#### 1.2.5. Pupil Gaze (Key 5)

- **Type**: Computer Vision / Hybrid.
- **Function**: Track pupil position _within_ the eye using pixel-intensity blob tracking (finding the darkest region) combined with Head Pose.
- **Benefit**: Provides higher precision tracking of eye movements independent of head rotation, without requiring heavy/unavailable external gaze models.
- **Fallback**: Robust inputs (Mesh + CV) ensure availability without complex dependencies.

#### 1.2.6. Moondream Gaze (Key 7)

- **Type**: Calibration / Validation.
- **Function**: Periodically analyzes the video frame using a Vision-Language Model (VLM) to verify gaze direction against the ONNX pipeline.
- **Output Flow**:
  1. **Capture (Start)**: Frame and Gaze captured. Status: "MOON: WATCHING...".
  2. **Pending**: Displays **Green Dot with Red Center** (Captured ONNX Gaze) to confirm capture timing.
  3. **Completion**:
     - **Cyan Dot**: Moondream's predicted gaze point.
     - **Green Dot with Yellow Center**: The original captured ONNX gaze point for comparison.

### 1.3. User Interface (UI)

- **Windowing**: A native window displaying the video feed with overlaid graphics.
- **Controls**:
  - `1`: Toggle Face Mesh.
  - `2`: Toggle Head Pose.
  - `3`: Toggle Eye Gaze.
  - `5`: Toggle Mirror Mode.
  - `6`: **Toggle** Overlay (Show/Hide macOS Sidecar).
  - `7`: **Toggle** Moondream Mode.
  - `9`: **Toggle** Calibration Mode.
  - `Space`: Capture Calibration Point (if Mode 9 active).
  - `ESC`: Quit application.
- **Visuals**:
  - **Red**: Face Mesh (Configurable).
  - **Blue Dot**: Real-time ONNX Gaze (Current Frame).
- **Green Dot (Red Center)**: _PENDING_ Capture. Represents the ONNX gaze position at the exact moment the frame was sent to Moondream.
- **Green Dot (Yellow Center)**: _COMPLETED_ Capture. Represents the same ONNX gaze position, updated to "Approved" state when Moondream returns.
- **Cyan Dot (Yellow Center)**: _RESULT_. The gaze position returned by Moondream.
- **Visual Logic**:
  - When **[7]** is active:
    - Every ~30 frames, a capture occurs.
    - **IMMEDIATELY**: A Green/Red dot appears at the current Blue dot location.
    - **LATENCY**: Processing...
    - **RESULT**: The Green/Red dot turns Green/Yellow. A Cyan/Yellow dot appears nearby.
    - **REGRESSION PROTECTION**: The Green dot MUST correspond to where the Blue dot _was_ at the start of the cycle.

## 2. Technical Requirements

### 2.1. Performance

- **Language**: Rust (Safe, Fast).
- **Inference Engine**: `ort` (ONNX Runtime) with CoreML/CPU Execution Providers for hardware acceleration on Apple Silicon.
- **Framerate**: The pipeline overhead shall not reduce the camera's native framerate significantly (Target: 60+ FPS processing).

### 2.2. Dependency Management

- **Model Management**: A shell script (`get_model.sh`) shall automatically fetch required ONNX models from reliable public sources (e.g., PINTO Model Zoo) to keep the repository size small.
- **Crate Dependencies**:
  - `nokhwa`: Cross-platform Camera capture.
  - `ort`: ONNX Runtime bindings.
  - `minifb`: Minimal windowing.
  - `image`: Image processing.

### 2.3. Project Maintenance

- **Changelog**: All functional changes, bug fixes, and architectural decisions MUST be documented in `CHANGELOG.md` at the root of the repository.
- **Format**: The changelog shall follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standards and include the reasoning ("Why") behind complex changes.
- **Reference**: High-level design intent should be traced back to this Core Spec or specific Feature Specs.

## 3. Constraints & Assumptions

- **OS**: Primary target macOS (Apple Silicon), but code should be cross-platform compatible (Linux/Windows).
- **Models**: Must be available in ONNX format.
- **Tooling**: `swiftc` required for macOS overlay integration.
- **Privacy**: All processing must happen locally; no video data sent to cloud.
