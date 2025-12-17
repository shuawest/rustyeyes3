# Overlay Specification

This document defines the behavior of the Full-Screen Overlay, specifically the **Triple Cursor** system used for Moondream calibration verification.

## Architecture

The overlay runs as a separate process (`overlay_app`, compiled from `src/overlay_sidecar.swift`) and communicates with the main Rust application via `stdin`.

## Protocol

Commands are new-line delimited strings sent to `stdin`:

| Command       | Format  | Meaning                   | Visual                          |
| :------------ | :------ | :------------------------ | :------------------------------ |
| **Gaze**      | `G x y` | Update Real-time Gaze     | **Blue** Circle with Red Dot    |
| **Moondream** | `M x y` | Update Model Prediction   | **Cyan** Circle with Gold Dot   |
| **Captured**  | `C x y` | Update Captured Benchmark | **Green** Circle with White Dot |

_Legacy Support_: `x y` (without prefix) is treated as `G x y`.

## Cursor Definitions

### 1. Real-time Gaze (Blue)

- **Source**: ONNX Head Pose / Pupil Gaze pipeline.
- **Update Rate**: ~30-60 FPS (Camera Native).
- **Behavior**: Smoothly tracks the user's current eye position. "Flies" around the screen.

### 2. Moondream Gaze (Cyan)

- **Source**: Moondream2 Vision-Language Model (Simulated in v0.1).
- **Update Rate**: ~0.5 FPS (High Latency).
- **Behavior**: Updates discretely every few seconds. Represents the model's _inference result_ for a past frame.

## Technical Implementation

### HUD Display

The overlay renders real-time coordinate data in **five locations** for maximum visibility:

- **Top-Left Corner** (20px from edges, 120px from top)
- **Top-Right Corner** (250px from right edge, 120px from top)
- **Bottom-Left Corner** (20px from edges, 40px from bottom)
- **Bottom-Right Corner** (250px from right, 40px from bottom)
- **Center** (centered horizontally, slightly above center vertically)

Each HUD displays three lines:

```
REALTIME:  0960, 0540  (Blue/Red cursor - 60 FPS)
CAPTURED:  0800, 0600  (Green/White - snapshot at model trigger)
MOONDREAM: 0720, 0480  (Cyan/Gold - model prediction, ~2 Hz simulated)
```

### Communication Protocol

Standard input commands:

- `G x y` - Update real-time gaze cursor (Blue/Red)
- `C x y` - Update captured gaze cursor (Green/White)
- `M x y` - Update Moondream prediction cursor (Cyan/Gold)

### Performance Optimization

**Channel Draining Strategy:**

- Frame channel uses unbounded queue with worker-side draining
- Result channel drains on main thread to display only latest prediction
- Prevents lag accumulation from 60 FPS producer → 2 Hz consumer mismatch

**Gaze Smoothing:**

- Low-pass filter (alpha=0.3) applied to real-time gaze
- Reduces jitter while maintaining responsiveness
- Smoothed coordinates used for both Blue cursor display and Green snapshot capture

**Update Frequencies:**

- Blue (Real-time): 60 FPS
- Green (Captured): ~2 Hz (every 0.5s when Moondream mode active)
- Cyan (Moondream): ~2 Hz (0.5s simulated inference delay)

## Current Limitations

**Moondream Integration Status (v0.1):**

- Currently using **simulated gaze prediction** (hardcoded scanning pattern)
- Real Moondream2 model loading blocked by `candle-transformers` architecture incompatibility
- Projection layer shape mismatch: expected `[1152, 1152]`, actual `[8192, 2304]`
- Simulation validates entire pipeline architecture (UI, threading, channels, overlay)
- Future: Real model integration pending library compatibility or alternative VLM

## Future Enhancements

- Load actual Moondream2/compatible VLM for real gaze prediction
- Adaptive smoothing based on head movement velocity
- Configurable HUD positions and visibility toggles
- Performance metrics overlay (FPS, latency)

### 3. Captured ONNX Gaze (Green)

- **Source**: Snapshot of the Real-time Gaze (Blue).
- **Trigger**: Captured at the exact millisecond the frame was sent to Moondream.
- **Behavior**: Updates in sync with the Cyan cursor.
- **Purpose**: Provides a **Ground Truth Reference**. By comparing the Green Dot (where you _were_ looking) with the Cyan Dot (where the model _thinks_ you looked), we can visually verify accuracy.

## Coordinate System

- **Input**: (0,0) = Top-Left, (Width, Height) = Bottom-Right.
- **Rendering**: macOS Cocoa uses Bottom-Left origin. The sidecar automatically inverts `y` (`ScreenHeight - InputY`).
