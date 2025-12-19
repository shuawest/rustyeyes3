# Custom Gaze Model Training Specification

## Objective

Train a **personalized gaze estimation model** using user-specific calibration data to overcome the limitations of the generalized L2CS-Net model.
**Target Hardware**: `jowestdgxe` (NVIDIA DGX Spark).

---

## 1. Methodology: Transfer Learning via Fine-Tuning

Training from scratch requires millions of images (`Gaze360`, `MPIIFaceGaze`).
We will **Fine-Tune** the existing L2CS-Net (ResNet50 backbone) on the user's calibration data.

- **Backbone**: ResNet50 (Pre-trained on ImageNet + Gaze360).
- **Head**: Regressor (Linear Layer -> Yaw, Pitch).
- **Strategy**: Freeze the first 3/4 layers of ResNet50. Train the final block and the regressor head.

---

## 2. Data Preparation Pipeline

### Source

- Locally: `calibration_data/*.jpg` (Raw Webcam Frames).
- Locally: `calibration_data/*.json` (Screen Coordinates).

### Preprocessing (Script: `scripts/prepare_dataset.py`)

1.  **Detection**: Run `face_detection.onnx` to find the face.
2.  **Crop**: Extract face with 1.5x margin (matching run-time inference).
3.  **Resize**: 448x448 pixels.
4.  **Labeling**:
    - Convert `Target Screen X/Y` -> `Gaze Yaw/Pitch` (Degrees).
    - **Formula**: Use the geometry verified in `analyze_calibration.rs`:
      `yaw = (target_x - screen_center_x) * deg_per_pixel`
      `pitch = (target_y - screen_center_y) * deg_per_pixel`
      _Use `deg_per_pixel = 0.0307` and `center = 864` (Standard Macbook)._
5.  **Output**: `dataset_clean/` folder containing cropped JPGs and `labels.csv` (`filename, yaw, pitch`).

### Augmentation (Crucial for Small Data)

Since we have < 100 samples, we must augment 50x:

- **Color Jitter**: Brightness, Contrast, Saturation (Simulate lighting changes).
- **Blur**: Gaussian Blur (Simulate motion blur).
- **Noise**: ISO Noise.
- **Shift**: Small pixel shifts (Simulate detector jitter).

---

## 3. Remote Execution (`jowestdgxe`)

### Environment

- Python 3.9+
- PyTorch, torchvision
- ONNX, onnxruntime

### Deployment Script (`scripts/deploy_train.sh`)

1.  **Pack**: Zip `dataset_clean/`.
2.  **Upload**: `scp dataset.zip jowestdgxe:~/rustyeyes_train/`.
3.  **Sync**: `scp scripts/train_remote.py jowestdgxe:~/rustyeyes_train/`.

### Training Logic (`train_remote.py`)

1.  **Load**: `L2CS-Net` weights (Download from official repo or upload our ONNX converted to PyTorch).
2.  **Dataset**: Custom `Dataset` class applying on-the-fly augmentation.
3.  **Loop**:
    - Loss: `MSELoss` (Mean Squared Error) between Predicted Angles and Target Angles.
    - Optimizer: Adam, LR=1e-4.
    - Epochs: 50 (Early Stopping if val_loss increases).
4.  **Export**:
    - Convert best PyTorch model -> ONNX (`custom_gaze.onnx`).
    - Input shape: `(1, 3, 448, 448)`.
    - Output shape: `(1, 2)` (Yaw, Pitch).

---

## 4. Integration

1.  **Download**: `scp jowestdgxe:~/rustyeyes_train/custom_gaze.onnx .`.
2.  **Deploy**: Move to `models/custom_gaze.onnx`.
3.  **Config**: Update `config.json` to point `l2cs_path` to the new model.
4.  **Reset**: Set `yaw_gain=1.0`, `offset=0.0`. The model is now native!
