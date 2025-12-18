# Rusty Eyes 3

## Overview

Rusty Eyes 3 is a high-performance webcam-based eye tracker written in Rust for MacOS. It uses `nokhwa` for camera capture, `ort` (ONNX Runtime) for real-time neural inference, and `moondream` (Python VLM) for asynchronous verification.

## Quick Start

1.  **Install Dependencies**
    Ensure you have Rust installed.

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

## Key Features

- **Mirror Mode**: Toggle with `[5]`. Mirrors video and gaze direction.
- **Calibration**: Toggle with `[9]`. Use Spacebar to capture points.
- **Moondream**: Toggle with `[7]`. Periodically validates gaze against a VLM.
- **HUD**: Toggle with `[0]`. Shows FPS, Yaw/Pitch, and status.

## Dev Notes

- Edit `config.json` for camera settings.
- See `specs/` for architectural details.
- Logs: `cargo run` prints standard logs. Enable debug logging in `main.rs` for IPC traces.
