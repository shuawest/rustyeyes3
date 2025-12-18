# Offline Calibration System Specification

## Overview

A data-driven calibration system to map gaze model outputs (Yaw/Pitch) to accurate screen coordinates. It processes a dataset of images and ground-truth screen coordinates to optimize calibration parameters (Gain, Offset) which are then saved and loaded by the main application.

## Data Format

Input data resides in `calibration_data/`. Pairs of files:

- `img_{timestamp}.jpg`: Captured webcam frame.
- `img_{timestamp}.json`: Metadata containing:
  ```json
  {
      "screen_x": f32,
      "screen_y": f32,
      ...
  }
  ```

## Calibration Model

The gaze projection logic is defined as:

```rust
PredictedX = EyeCenter.x + sin((Yaw - OffsetYaw).to_radians()) * GainX * ScreenWidth
PredictedY = EyeCenter.y - sin((Pitch - OffsetPitch).to_radians()) * GainY * ScreenHeight
```

_Note: This matches the verification logic in `main.rs`, parameterized for calibration._

The solver optimizes **4 parameters**:

1. `offset_yaw` (degrees)
2. `offset_pitch` (degrees)
3. `gain_yaw` (scalar)
4. `gain_pitch` (scalar)

## Components

### 1. `calibration.json`

A new configuration file stored in the root (or `calibration_data/`) containing:

```json
{
  "l2cs": {
    "yaw_offset": 0.0,
    "pitch_offset": 12.0,
    "yaw_gain": 5.0,
    "pitch_gain": 5.0
  },
  "mobile": { ... }
}
```

### 2. `calibration_history.json`

Stores a history of all calibration runs to enable benchmarking and best-selection.

```json
[
  {
    "run_id": "2024-12-17T23:45:00Z",
    "model": "l2cs",
    "params": { ... },
    "metrics": {
      "mean_error": 45.2,
      "std_dev": 12.5,
      "max_error": 120.0
    },
    "best": true
  }
]
```

### 3. `calibration_report_{run_id}.json`

Detailed per-run report.

```json
{
  "run_id": "...",
  "timestamp": "...",
  "summary": {
    "mean_error": 45.2,
    "std_dev": 12.5,
    "histogram": [
      { "bin": "0-10px", "count": 15 },
      { "bin": "10-50px", "count": 5 }
    ]
  },
  "entries": [
    {
      "filename": "img_123.jpg",
      "target_x": 100,
      "target_y": 100,
      "calibrated_x": 105,
      "calibrated_y": 95,
      "delta": 7.07,
      "percent_error": 0.5
    }
  ]
}
```

### 4. Gaze Pipeline Refactor

The `L2CSPipeline` and `MobileGazePipeline` structs must be updated to:

- Accept calibration parameters on initialization or update.
- Apply these parameters during the `process()` loop instead of hardcoded values.

### 5. CLI Tool (`src/bin/calibrate.rs`)

A new binary that:

1. Reads `calibration_data/`.
2. Instantiates the Gaze Pipelines (L2CS and Mobile).
3. Iterates through all valid `.jpg` + `.json` pairs.
4. Runs inference to get **Raw** Yaw/Pitch and Eye Centers.
5. Performs a Gradient Descent (or similar optimization) to minimize Euclidean distance between `(PredictedX, PredictedY)` and `(TargetX, TargetY)`.
6. **Benchmarking**: compares the current run's mean error against historical bests in `calibration_history.json`.
7. **Selection**: If the current run is better, updates `calibration.json`.
8. **Reporting**: Generates a detailed JSON report with histograms and per-file metrics.

## Usage

```bash
# Run calibration
cargo run --bin calibrate

# Run app (automatically loads calibration.json)
cargo run --release
```
