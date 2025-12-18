# Rusty Eyes 3

## Overview

Rusty Eyes 3 is a high-performance webcam-based eye tracker written in Rust for MacOS. It uses `nokhwa` for camera capture, `tract` (ONNX) for real-time inference, and `moondream` (Python VLM) for asynchronous verification.

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
