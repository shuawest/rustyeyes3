# Strategic Review: Path to Super-Accurate Gaze

## Current Status & Limitations

We have pushed the **General Purpose RGB Webcam** approach near its limit with static configuration.

- **Method**: L2CS-Net (General Model) -> Polynomial Regression (Calibration) -> Simple Smoothing.
- **Ceiling**: L2CS has a native error of ~4-5 degrees. At 60cm distance, 1 degree = ~1cm. 5 degrees = **5cm error circle**.
- **The "Gain" Trap**: To cover the screen with small eye movements, we set `Gain ~ 2.5`. This amplifies the 5cm native error into a **12.5cm error**, causing the jitter/instability you feel.
- **Conclusion**: We cannot "tune" away the base noise of the model. We must either **filter it intelligently** or **replace the input source**.

---

## Tier 1: Software Improvements (No Cost, High Effort)

_Enhancing the signal we have._

### 1. Advanced Jitter Filtering (The "Smoothing Buffer")

Current smoothing is a simple "Moving Average". It causes lag ("swimming") and doesn't stop jitter when still.

- **Solution: One-Euro Filter**: An adaptive filter that jitters _less_ when moving slowly (reading) and reacts _fast_ when moving quickly (saccades).
- **Solution: Fixation Latching**: A logic layer that "locks" the cursor to a location if the eye velocity drops below a threshold, eliminating 100% of static jitter.
- **Impact**: High. Essential for usability.

### 2. Personalized "Neural" Calibration

Polynomial regression (ax² + bx + c) is too rigid. It assumes the error is a smooth curve.

- **Solution**: Train a tiny, personalized Neural Network (MLP) or Support Vector Regressor (SVR) on the calibration data.
- **Impact**: Medium. Handles non-linear lens distortion better than polynomials.

### 3. Moondream "Grounding" (Hybrid approach)

Use the VLM (Moondream) not just for "queries" but as a periodic "Ground Truth" corrector.

- **Solution**: Every 5 seconds, capture a "Gold Standard" frame. Use the VLM's high accuracy to re-center the real-time tracker's drift.
- **Impact**: High (for drift), but slow update rate.

---

## Tier 2: Model Improvements (Medium Cost/Effort)

_Replacing the brain._

### 1. Fine-Tune L2CS

Collect 1000s of frames of _your_ face and fine-tune the L2CS ONNX model weights.

- **Pros**: The model learns _your_ specific eye shape and lighting conditions.
- **Cons**: Requires a rigorous data collection rig and GPU training pipeline (Python/PyTorch).
- **Impact**: High. Can reduce native error from 5° to ~2-3°.

### 2. Switch Architectures

Switch from L2CS (ResNet) to something newer like **GazeTR** (Transformer) or **MPIIGaze** variants.

- **Pros**: Potentially higher state-of-the-art accuracy.
- **Cons**: higher CPU/Latency cost.

---

## Tier 3: Hardware Upgrades (Financial Cost, Best Result)

_Replacing the eyes._

### 1. Infrared (IR) Camera ($20 - $100)

**This is the "cheat code" for eye tracking.**
RGB tracking attempts to "guess" pose from pixel texture. IR tracking uses **Physics** (Corneal Reflections / Glint).

- **Method**: IR LED illuminator + IR Camera (e.g., modified PS3 Eye or dedicated module).
- **Algorithm**: Pupil Center Corneal Reflection (PCCR).
- **Accuracy**: Sub-millimeter / < 1 degree.
- **Jitter**: Almost non-existent.
- **Pros**: Solves lighting issues, solves texture noise.
- **Cons**: Need to buy hardware.

### 2. High FPS Camera (60/120Hz)

Standard webcams are 30fps with motion blur.

- **Pros**: 4x the data points for smoothing. Filters work 4x better.
- **Cons**: CPU usage.

### 3. Two-Camera System

Stereo vision to solve depth/head pose perfectly.

- **Pros**: Decouples head movement from eye movement mathematically.
- **Cons**: Software complexity is extreme (calibration between cameras).

---

## Recommendation / Roadmap

**Phase 1: Maximize Software (This Week)**
We are here. We haven't tried smart filtering yet.

1.  Implement **One-Euro Filter** (Solve the Jitter/Lag tradeoff).
2.  Implement **Fixation Latching** (Make it feel "solid" when reading).
3.  _Verdict_: If this isn't enough, we hit the RGB hardware limit.

**Phase 2: Data-Driven Calibration (Next Week)**
Replace the "Polynomial" math with a "Personalized Regressor" (SVR/MLP) in `calibrate.rs`.

**Phase 3: Hardware (The "Nuclear" Option)**
If Phase 1/2 fail, buy a generic IR Camera + IR LED (~$30 on Amazon). We write a new pipeline (`IRGazePipeline`) that tracks the glint. This guarantees success but requires hardware.
