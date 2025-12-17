# Calibration Specification: Moondream2 Gaze Assist

## 1. Overview

This document defines the calibration system for `rustyeyes3`. The goal is to map the raw output from the gaze models (FaceMesh ONNX and Moondream2) to actual screen coordinates.

The system utilizes a **Hybrid Architecture**:

1.  **Fast Path (Primary)**: Existing FaceMesh + Eye Blob tracking (Rust/ONNX). Runs at ~120FPS. Controls the cursor.
2.  **Slow Path (Oracle)**: Moondream2 (Rust/Candle/Python). Runs at ~0.1FPS. Corrects the Fast Path.

## 2. Methodology

We will implement a **Data Collection & Regression** approach.

1.  **Data Collection Mode**: User provides Ground Truth (GT) by clicking on specific points on the screen while looking at them.
2.  **Dataset**: We collect triplets of `{Image, Screen_X_GT, Screen_Y_GT}`.
3.  **Inference**: Run both models on the collected images to get `{Vector_ONNX, Vector_MD}`.
4.  **Calibration**: Compute homography or polynomial regression matrices to map `Vector -> Screen`.

## 3. Workflow

### 3.1. Data Collection Mode (New)

**Activation**: Key `9` toggles "Calibration Mode".

**Interaction**:

1.  User looks at a point on the screen.
2.  User presses `SPACE` (or Mouse Click).
3.  **System Action**:
    - Get current Mouse Cursor Position $(X_{screen}, Y_{screen})$.
    - Capture current Frame from Camera.
    - Save Image to `calibration_data/img_{timestamp}.jpg`.
    - Save Metadata to `calibration_data/img_{timestamp}.json` (JSON serialization of `CalibrationPoint`).
    - Visual Feedback: Flash screen or play sound.

### 3.2. Calibration Computation (Offline/Triggered)

**Activation**: Key `0` (or dedicated key) triggers "Compute Calibration" (if data exists).

**Process**:

1.  Read `calibration_data/dataset.csv`.
2.  For each row:
    - Load `img_{timestamp}.jpg`.
    - **Run ONNX Pipeline**: Extract `Landmarks` -> Compute Raw Gaze Vector $(Yaw, Pitch)$ or Eye Center Vector. Let's call this $V_{onnx}$.
    - **Run Moondream Oracle**: Extract Gaze Point $(X_{md}, Y_{md})$ (Normalized). Let's call this $V_{md}$.
3.  **Compute Mappings**:
    - **ONNX Mapping**: Find matrix $M_{onnx}$ such that $M_{onnx} \times V_{onnx} \approx P_{screen}$.
      - Likely a **Projective Transformation (Homography)** or **2nd Order Polynomial** since head pose/eye rotation is non-linear to screen planar coords.
    - **Moondream Mapping**: Find matrix $M_{md}$ such that $M_{md} \times V_{md} \approx P_{screen}$.
      - Moondream outputs normalized coords $(0..1, 0..1)$, so this might just be a simple Affine transform (Scale + Offset) or Homography to account for camera angle.
4.  **Save Profile**:
    - Save matrices/coefficients to `calibration.json`.

### 3.3. Inference (Runtime)

1.  **Load** `calibration.json` on startup.
2.  **Fast Path**:
    - $P_{raw} = \text{Pipeline}(Frame)$
    - $P_{screen} = M_{onnx}(P_{raw})$
    - Move Mouse to $P_{screen}$.
3.  **Slow Path (Correction)**:
    - Periodically run Moondream.
    - $P_{md\_raw} = \text{Moondream}(Frame)$
    - $P_{md\_screen} = M_{md}(P_{md\_raw})$
    - _Drift Correction_: Compare $P_{md\_screen}$ with current average $P_{screen}$. Adjust offset if consistent drift detected.

## 4. Implementation Details

### 4.1. File Structure

```
rustyeyes3/
├── calibration_data/      # Gitignored directory
│   ├── dataset.csv
│   └── img_123456789.jpg
└── calibration.json       # Generated profile
```

### 4.2. Data Format (JSON per sample)

Instead of a single CSV, we save a JSON file alongside each image for easier atomic management.

**File:** `img_{timestamp}.json`

```json
{
  "timestamp": 1702934821,
  "screen_x": 500.0,
  "screen_y": 500.0
}
```

### 4.3. Math

We will start with a simple **Linear Regression** (Affine Transform) for robustness, then try Homography if accuracy is poor.
$X_{screen} = a X_{raw} + b Y_{raw} + c$
$Y_{screen} = d X_{raw} + e Y_{raw} + f$

This requires at least 3 points (non-collinear). 9 points (3x3 grid) recommended.

## 5. UI Changes

- **Main Window**:
  - Add status text: "Mode: Calibration (9)"
  - When in Calibration mode: "Move mouse & Press SPACE to capture."
  - Show count of captured points: "Captured: 5"

## 6. Future Work

- **Automatic Target Display**: Instead of user choosing points, app displays a red dot sequence. (Out of scope for now).

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
