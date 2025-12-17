# Calibration Specification: Moondream2 Gaze Assist

## 1. Overview

This feature introduces a "Gaze Calibration" system using the multimodal **Moondream2** model. While too slow for real-time cursor control (8-10s latency), Moondream2 serves as a "Ground Truth" oracle to correct the drift and offset of the high-speed (100FPS+) geometric/CV gaze tracking.

## 2. Architecture

The system utilizes a **Hybrid Architecture**:

1.  **Fast Path (Primary)**: Existing FaceMesh + Eye Blob tracking (Rust/ONNX). Runs at ~120FPS. Controls the cursor.
2.  **Slow Path (Oracle)**: Moondream2 (Rust/Candle). Runs at ~0.1FPS. Corrects the Fast Path.

## 3. Milestones

### Milestone 1: Visual Verification (Option 7)

**Goal**: Visually confirm Moondream2's accuracy compared to the current geometric solution.

- **Interaction**: User presses `7`.
- **Behavior**:
  1.  App **pauses** the video feed (captures a "Snapshot").
  2.  Display shows the Snapshot.
  3.  Runs Moondream2 inference on the snapshot.
  4.  **Overlays**:
      - 🔴 **Red**: Face Mesh (Geometric Gaze).
      - 🔵 **Blue**: Pupil Gaze (CV).
      - 🟡 **Gold**: Moondream Gaze (Oracle).
- **Success Metric**: Gold vector aligns better with true eye direction than Red/Blue.

### Milestone 2: Asynchronous Picture-in-Picture

**Goal**: Run Moondream2 in parallel without blocking the main UI.

- **Architecture**:
  - Spawns a background `std::thread`.
  - Main thread sends a `Clone` of the current frame to background channel.
  - Background thread processes frame (taking ~10s).
  - Result sent back via `std::sync::mpsc`.
- **UI**:
  - Main view continues showing real-time feed (60FPS).
  - Small **PiP (Picture-in-Picture)** window appears in corner showing the _delayed_ frame that Moondream analyzed, overlaid with the Moondream result.

### Milestone 3: Closed-Loop Auto-Calibration

**Goal**: Automatically correct the cursor offset using Moondream data.

- **Logic**:
  1.  **Time Buffering**: Store a "History Buffer" of recent Real-time Gaze vectors ($V_{rt}$) with timestamps.
  2.  **Comparison**: When Moondream returns a result ($V_{md}$) for Frame $T_{snapshot}$:
      - Look up $V_{rt}$ at time $T_{snapshot}$.
      - Calculate Error: $\Delta = V_{md} - V_{rt}$.
  3.  **Correction**:
      - Apply a smoothing factor $\alpha$ (e.g., 0.1).
      - Update global calibration offset: $C_{new} = C_{old} + (\alpha \times \Delta)$.
      - Real-time cursor is now computed as $V_{final} = V_{rt} + C_{new}$.

## 4. Technical Stack

- **Inference**: `candle-rs` (Pure Rust ML framework).
  - Advantages: No Python dependency, integrates directly into `rusty-eyes` binary, supports Apple Silicon (Metal) acceleration.
- **Model**: `vikhyatk/moondream2` (likely quantized for edge performance on Pi/Device).
- **Input**: 512x512 Resized image from Camera.

## 5. Risk Assessment & Mitigations

| Risk           | Impact                                            | Mitigation                                                           |
| :------------- | :------------------------------------------------ | :------------------------------------------------------------------- |
| **Latency**    | Moondream takes >10s on Pi.                       | Run in disjoint background thread; do not block UI.                  |
| **Accuracy**   | Moondream might be _less_ accurate than geometry. | Milestone 1 is critical to verify this assumption before automation. |
| **Heat/Power** | Running both models overheats device.             | Limit Moondream frequency (e.g., once every 30s) or active cooling.  |

## 6. Testing Plan

### 6.1. Unit Tests

- `test_moondream_parsing`: Ensure Moondream output (coords/text) is correctly parsed into `Point3D`.
- `test_history_buffer`: Ensure looking up past gaze vectors by timestamp is accurate.

### 6.2. Integration Tests

- **Thread Safety**: Verify sending frames to background thread doesn't crash or leak memory.
- **Cancellation**: Ensure switching modes kills the heavy background task.

### 6.3. Performance Tests

- **Benchmark**: Measure `candle` inference time on Mac (M1/M2) vs Pi.
