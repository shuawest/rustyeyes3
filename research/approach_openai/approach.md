# Gaze Estimation Model Calibration and Training Plan

## Introduction and Objectives

Current eye-gaze tracking is limited by using face orientation instead of true pupil direction, resulting in jittery and inaccurate on-screen gaze points. The goal is to develop a calibrated gaze estimation model that takes a webcam frame and reliably outputs the point on the screen the user is looking at (`screen_x`, `screen_y`). This model should remain accurate even if the user’s face is turned away (eyes looking out of face center), and it must maintain high precision (minimal jitter) at real-time speeds (about 60 Hz).

We will leverage existing models (for example, Pinto’s Model Zoo ONNX models or Moondream2) by fine-tuning or calibrating them with our collected data, rather than starting from scratch. The solution must be adaptable to different screen sizes and configurations, meaning the gaze outputs can be mapped to any display resolution after calibration. Training will run on NVIDIA DGX Spark GB10; inference will run on a MacBook Pro (M3/M4 GPU/NPU) in your current harness (ONNX real-time, Moondream2 async).

### Key Objectives

- **Accuracy:** Minimize error between predicted and true `screen_x`, `screen_y`.
- **Precision:** Reduce jitter and noise in gaze output, stable points for micro-movements.
- **Robustness to head pose:** Track eye gaze independent of face direction.
- **Calibration and generalization:** Support per-user and per-device recalibration with few samples.
- **Performance:** Lightweight model capable of 60 FPS or more on Apple Silicon.

## Calibration Data Collection and Preparation

You already have a calibration dataset: webcam images (`img_*.jpg`) and matching JSON files (`img_*.json`) with ground truth `screen_x`, `screen_y`, plus current model inference data (eye positions, head pose, etc.). The calibration includes cases where the face is pointed away while the eyes remain fixated on the target, which is exactly what we need to break face-direction bias.

Example JSON:

```json
{
  "timestamp": 1765997461159,
  "screen_x": 1076.6445,
  "screen_y": 973.9961,
  "inference": {
    "Gaze": {
      "left_eye": { "x": 900.979, "y": 359.03485, "z": 0.0 },
      "right_eye": { "x": 1095.01, "y": 354.9803, "z": 0.0 },
      "yaw": -23.216734,
      "pitch": 15.424934,
      "roll": 0.69636035,
      "vector": { "x": 0.0, "y": 0.0, "z": 1.0 },
      "landmarks": null
    }
  },
  "moondream_result": { "x": 0.0, "y": 0.0, "z": 0.0 }
}

Data preprocessing steps
	1.	Label normalization
	•	Convert pixel labels to normalized coordinates:
	•	x_norm = screen_x / screen_width
	•	y_norm = screen_y / screen_height
	•	For your capture: screen_width = 1920, screen_height = 1080.
	•	This makes the model screen-size agnostic; multiply by current resolution at inference to recover pixels.
	2.	Face and eye detection
	•	Use existing eye center values in JSON if reliable, or compute via MediaPipe FaceMesh.
	•	Crop fixed-size patches around eyes (for example, 60×60 or 96×96).
	•	Option A: Use two eye crops (left/right) as model inputs.
	•	Option B: Use a single face crop focusing on eye region.
	3.	Head pose input
	•	Use yaw, pitch, roll as explicit inputs when possible.
	•	This helps the model learn to disentangle eye movement from head movement.
	4.	Augmentation
	•	Photometric only: brightness/contrast, blur, noise.
	•	Horizontal flip only if you also swap left/right eye crops and transform label x_norm = 1 - x_norm.

Model Selection and Design

The priority is accurate pupil/eye gaze at real-time speed, and ONNX deployment. Prefer fine-tuning a specialized gaze model rather than forcing Moondream2 into numeric regression.

Candidate model families
	•	Pinto Model Zoo gaze models (ONNX)
	•	L2CS-Net family and mobile variants
	•	MobileGaze-style MobileNetV2 or MobileOne backbones

Moondream2 can remain async for higher-level checks, but should not be the core 60 FPS numeric regressor.

Output representation choices
	1.	Direct 2D screen regression
	•	Model predicts (x_norm, y_norm) directly.
	•	Fast and practical for screen gaze tracking.
	2.	Gaze angles or gaze vector
	•	Predict yaw/pitch or 3D vector then map to screen via calibration geometry.
	•	More modular, but requires distance/screen-plane assumptions.

Recommended: predict (x_norm, y_norm) directly, using a pretrained gaze backbone, optionally fused with head pose.

Architecture recommendation
	•	Backbone: MobileNetV2 (or MobileOne S0 if you need max FPS).
	•	Inputs:
	•	Left eye crop
	•	Right eye crop
	•	Head pose (yaw, pitch, roll)
	•	Fusion:
	•	Concatenate feature vectors + head pose embedding.
	•	Head:
	•	Small MLP that outputs (x_norm, y_norm).

Training Strategy on NVIDIA DGX Spark (Fine-Tuning)

We want transfer learning, minimal overfitting, and fast iteration.

Training pipeline
	1.	Dataset loader reads img_*.jpg, parses JSON, yields:
	•	Eye crops (and/or face crop)
	•	Head pose
	•	Target (x_norm, y_norm)
	2.	Loss:
	•	Primary: MSE or Smooth L1 on (x_norm, y_norm).
	•	Optional: Add a consistency loss for “same target different head pose” pairs.
	3.	Optimization:
	•	Adam or AdamW.
	•	Use discriminative learning rates:
	•	Backbone LR: 1e-5 to 1e-4
	•	New head layers LR: 1e-4 to 1e-3
	•	Early stopping due to small dataset.
	4.	Freezing strategy:
	•	Start by freezing backbone and training only the head.
	•	If needed, unfreeze last N layers with very low LR.
	5.	Validation:
	•	Hold out a subset of points, or do K-fold cross validation.
	•	Evaluate in pixels by de-normalizing.

Metrics
	•	Mean absolute pixel error:
	•	err_px = sqrt((x_pred_px - x_true_px)^2 + (y_pred_px - y_true_px)^2)
	•	Optional angular error approximation if you assume a viewing distance.
	•	Stability metric:
	•	Standard deviation of predicted gaze when user holds still on one target.

Calibration Layer and Mapping to Screen Dimensions

Even a strong model benefits from a lightweight calibration layer.

Per-user/per-device calibration

Collect 5–20 calibration points on the new device or for new user, then fit:
	•	Affine mapping
	•	x_cal = a*x + b*y + c
	•	y_cal = d*x + e*y + f
	•	Or simpler:
	•	x_cal = a*x + b
	•	y_cal = c*y + d

Fit via least squares using calibration samples.

Resolution scaling

Because the model outputs normalized coordinates, for any screen:
	•	x_px = x_cal * screen_width
	•	y_px = y_cal * screen_height

Real-Time Inference and Jitter Reduction

Runtime loop (60 Hz target)
	1.	Capture frame
	2.	Detect face + eyes (reuse your current landmarks pipeline if available)
	3.	Crop eyes, normalize input tensors
	4.	Run ONNX model
	5.	Apply calibration mapping
	6.	Convert to pixels for current screen
	7.	Optional smoothing filter

Smoothing options
	•	Exponential moving average:
	•	g_t = alpha*g_{t-1} + (1-alpha)*g_raw
	•	2D Kalman filter (more complex, better for jitter without lag)
	•	Adaptive smoothing:
	•	Lower smoothing (more responsive) when velocity high
	•	Higher smoothing (more stable) when velocity low

Deployment: ONNX and Apple Silicon Acceleration

Export and runtime
	•	Export fine-tuned model to ONNX.
	•	Run with ONNX Runtime on macOS.
	•	Consider CoreML conversion for Apple Neural Engine if needed:
	•	ONNX -> CoreML with coremltools
	•	Or ONNX Runtime CoreML EP (Execution Provider), depending on your stack.

Quantization
	•	FP16 is usually safe and fast on Apple GPU/NPU.
	•	INT8 can be tested if accuracy holds.

Moondream2 Role (Async)

Keep Moondream2 as an asynchronous assistant:
	•	Debug verification: classify region (top-left, center, etc.)
	•	Optional fallback when face landmarks fail
	•	Not the primary numeric tracker due to latency and cost

Future Expansion and Multi-User Strategy

Two-phase approach
	1.	Build a strong base model with your data.
	2.	Add new users incrementally:
	•	Collect small calibration sets per user (5–20 points).
	•	Either fit a user-specific calibration mapping or fine-tune only the final head layers.

Few-shot personalization (recommended direction)

Treat the backbone as universal, and train a tiny per-user head or calibration layer using a few samples. This gives fast onboarding for new users and avoids retraining the full model.

Deliverables as Agentic Coding Spec

Phase 1: Dataset and tooling
	•	Build dataset loader:
	•	Parse paired JPG/JSON
	•	Normalize labels
	•	Crop eyes reliably
	•	Emit tensors + labels

Phase 2: Baseline evaluation
	•	Measure current ONNX gaze error vs labels
	•	Measure jitter stats on held-still points
	•	Produce baseline metrics report

Phase 3: Fine-tune model on DGX
	•	Start with pretrained gaze backbone
	•	Train head for (x_norm, y_norm)
	•	Evaluate and iterate

Phase 4: Calibration module
	•	Implement least squares affine mapping
	•	Persist calibration per user/device
	•	Apply mapping at inference

Phase 5: Deployment + perf
	•	Export to ONNX
	•	Integrate into macOS harness
	•	Benchmark FPS and latency
	•	Add smoothing filter

Phase 6: Generalization
	•	Add new user data
	•	Validate few-shot calibration approach
	•	Improve robustness (lighting, glasses, head pose range)


```
