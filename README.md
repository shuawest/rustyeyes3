# Rusty Eyes 3

## Overview

Rusty Eyes 3 is a high-performance webcam-based eye tracker written in Rust for MacOS. It uses `nokhwa` for camera capture, `ort` (ONNX Runtime) for real-time neural inference, and `moondream` (Python VLM) for asynchronous verification.

## Quick Start

1.  **Install Dependencies**
    Ensure you have Rust installed.
    
    **System Dependencies (Linux)**:
    Use the included Makefile to install required libraries (GStreamer, X11, OpenSSL).
    
    ```bash
    # For Ubuntu/Debian
    make setup-apt
    
    # For Fedora
    make setup-dnf
    ```

2.  **Download Models**
    Run the setup script to download all required ONNX models to the `models/` directory:

    ```bash
    ./scripts/setup_models.sh
    ```

3.  **Run Application**
    ```bash
    cargo run --release
    ```

## Controls

- `[1]` **Face Mesh**: Show/Hide the 468-point face mesh.
- `[2]` **Head Pose**: Show/Hide head orientation axes.
- `[3]` **Eye Gaze**: Show/Hide the Gaze Ray.
- `[4]` **Cycle Model**: Switch Gaze Models (Simulated, Pupil, L2CS, MobileGaze).
- `[5]` **Mirror Mode**: Flip the camera feed horizontally.
- `[6]` **Overlay**: Toggle the detached transparent overlay window.
- `[7]` **Moondream**: Enable VLM verification (requires PyTorch server).
- `[9]` **Calibration**: Enter active calibration mode.
- `[Space]` **Capture**: Take a calibration snapshot.

## Visual Feedback Guide

The overlay provides real-time feedback on tracking status:

## Visual Feedback Guide

The overlay provides real-time feedback on tracking status:

- **Blue Dot**: Your current real-time gaze.
- **Green Dot (Red Center)**: "Processing". This marks where you were looking when the system captured a frame for Moondream.
- **Green Dot (Yellow Center)**: "Allocated". The Moondream result has arrived for this green spot.
- **Cyan Dot (Yellow Center)**: "Moondream Opinion". Where the VLM thinks you are looking.

**Goal**: The Green and Cyan dots should be close to each other!

### Calibration

To improve gaze accuracy for your specific setup, use the offline calibration tool:

1.  **Collect Data**:
    - Toggle **Calibration Mode** by pressing `[9]`.
    - Move your mouse cursor to a target on the screen.
    - Look at the cursor with both eyes.
    - Press `[Space]` to capture the point.
    - Repeat for 9-16 points covering the screen edges and center.
2.  **Run Calibration**:

    ```bash
    cargo run --release --bin calibrate
    ```

    This will:

    - Process all captured images.
    - Optimize parameters for both models.
    - Generate a report `calibration_report_{model}_{timestamp}.json`.
    - Update `calibration_history.json`.
    - Automatically select the best historical parameters and save them to `config.json`.

3.  **Inspect Results**: Check the generated JSON reports for accuracy metrics (Mean Error, Precision).

### Key Bindings

- **Mirror Mode**: Toggle with `[5]`. Mirrors video and gaze direction.
- **Calibration**: Toggle with `[9]`. Use Spacebar to capture points.
- **Moondream**: Toggle with `[7]`. Periodically validates gaze against a VLM.
- **HUD**: Toggle with `[0]`. Shows FPS, Yaw/Pitch, and status.

## Custom Model Training (Advanced)

To achieve higher accuracy than the standard L2CS model, you can fine-tune a model on your specific calibration data.

1.  **Prepare Data (Local)**:

    ```bash
    pip3 install -r scripts/requirements.txt
    python3 scripts/prepare_dataset.py
    ```

    This generates `dataset_clean/` with augmented training images derived from your `calibration_data`.

2.  **Train (Remote/DGX)**:
    Upload the `dataset_clean/` folder and `scripts/train_remote.py` to your GPU server.

    ```bash
    python3 train_remote.py --data_dir dataset_clean --output custom_gaze.onnx
    ```

3.  **Deploy**:
    Copy the resulting `custom_gaze.onnx` file to the `models/` directory.
    The application will automatically detect the regression model and switch to it.

## Dev Notes

- Edit `config.json` for camera settings.
- See `specs/` for architectural details.
- Logs: `cargo run` prints standard logs. Enable debug logging in `main.rs` for IPC traces.
