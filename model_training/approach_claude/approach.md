I'll research the current state of gaze estimation models and calibration techniques, then create a comprehensive spec for you. Let me start by investigating the best approaches.Now I have comprehensive research. Let me create a detailed agentic coding specification document for your gaze calibration project.Now let me create a practical starter script that you can run immediately with your existing calibration data:I've created a comprehensive specification and starter implementation for your gaze calibration system. Here's the executive summary:

## Key Research Findings

**Why your current model is jittery and inaccurate:**
Your ONNX model returns **head pose** (yaw/pitch/roll of the face), not **eye gaze direction**. The `vector: {x: 0, y: 0, z: 1}` field being static confirms no actual gaze computation is happening. True gaze = head pose + eye-in-head rotation (pupil direction relative to head).

## Recommended Phased Approach

| Phase | Approach               | Effort    | Expected Improvement   |
| ----- | ---------------------- | --------- | ---------------------- |
| **1** | SVR Calibration Layer  | 1-2 days  | 30-50% error reduction |
| **2** | L2CS-Net Integration   | 3-5 days  | Additional 20-40%      |
| **3** | Kalman Smoothing       | 1-2 days  | Eliminates jitter      |
| **4** | Custom Pupil Detection | 1-2 weeks | Maximum accuracy       |

## Immediate Action: Run the Starter Script

```bash
# With your existing data structure
python gaze_calibration_starter.py --calibration-dir ./calibration_data --visualize

# This will:
# 1. Analyze your calibration data coverage
# 2. Train SVR calibration models
# 3. Report leave-one-out cross-validation accuracy
# 4. Generate visualization of results
```

## Critical Insight from Research

The most successful webcam gaze systems use a two-stage approach:

1. **Appearance-based gaze model** (like L2CS-Net) that outputs pitch/yaw angles from the eye appearance
2. **Calibration layer** (SVR or polynomial regression) that maps those angles to screen coordinates

Your 15 samples are sufficient for SVR calibration (research shows 3-9 samples can work with few-shot methods like MAML), but collecting 25 points in a 5x5 grid would significantly improve robustness.

Excellent clarifications. Let me revise the approach with these specific constraints in mind. The key architectural insight is:

1. **DGX Spark (GB10)** = Training powerhouse (can train real models from scratch)
2. **MacBook Pro (M-series)** = Inference target (needs <16ms for 60Hz)
3. **Personal calibration first** → generalize to multi-user later
4. **Current stack**: PINTO ONNX (real-time) + Moondream2 (async, uncalibrated)

Let me create a refined architecture and training plan:Now let me create the practical training pipeline for DGX Spark:Now let me create the MacBook inference module that integrates with your existing PINTO ONNX pipeline:I've created a complete agentic coding specification with four deliverables:

## Summary of the System Architecture

### The Core Insight (Why Your Current Model is Jittery)

Your PINTO ONNX model outputs **head pose** (yaw/pitch/roll of the face), not **eye gaze**. The `vector: {0,0,1}` being static confirms this. True gaze = head pose + **pupil direction relative to head** (which is missing).

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 TRAINING (DGX Spark GB10)               │
├─────────────────────────────────────────────────────────┤
│  ETH-XGaze (1M+ images) + Gaze360 (238K images)        │
│              ↓                                          │
│  MobileNetV2 (0.5x width) → Gaze Angles (pitch, yaw)   │
│              ↓                                          │
│  + Your calibration data → Calibration MLP             │
│              ↓                                          │
│  Export both to ONNX                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           INFERENCE (MacBook Pro M-series)              │
├─────────────────────────────────────────────────────────┤
│  PINTO Face Detector → Gaze Model → Calibration MLP    │
│              ↓              ↓              ↓            │
│           (5ms)          (6ms)          (<1ms)         │
│                          ↓                              │
│               Kalman Smoothing → Screen (X, Y)         │
│                                                         │
│  Total: ~12ms = 80+ Hz on M1/M2/M3 Pro                 │
│                                                         │
│  Async (every 30th frame):                             │
│     Moondream2 → Drift detection / Recalibration flag  │
└─────────────────────────────────────────────────────────┘
```

### Screen-Size Adaptive Design

All outputs are **normalized [0,1]**, then scaled to current screen:

```python
pixel_x = normalized_x * screen_width  # Works on any resolution
pixel_y = normalized_y * screen_height
```

### What Each File Does

| File                          | Purpose                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `gaze_system_architecture.md` | Full spec with architecture diagrams, model definitions, training strategy     |
| `train_gaze_dgx.py`           | DGX Spark training script (Phase 1: base model, Phase 2: personal calibration) |
| `inference_macbook.py`        | MacBook Pro inference module with Kalman smoothing, CoreML support, 60Hz+      |
| `gaze_calibration_starter.py` | Quick SVR calibration on your existing data (immediate testing)                |

### Recommended Execution Order

1. **Today:** Run `gaze_calibration_starter.py` on your 15 samples to establish baseline error
2. **This week:** Download ETH-XGaze dataset, train base model on DGX Spark
3. **Then:** Fine-tune with your calibration data, export ONNX
4. **Finally:** Integrate `inference_macbook.py` with your existing Rust app

### Moondream2's Role (Clarified)

Moondream2 runs **asynchronously** (every ~30 frames) for:

- Drift detection (has calibration degraded?)
- Sanity checking the real-time estimate
- Triggering recalibration when needed

It does NOT run in the hot path—the real-time loop is pure ONNX for speed.
