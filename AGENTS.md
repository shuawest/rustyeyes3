# AGENTS.md

> [!NOTE]
> This file acts as a briefing packet for AI coding agents. It contains project specifications, build instructions, and architectural context.

## 1. Project Overview

**RustyEyes3** is a high-performance, Privacy-First Computer Vision application written in Rust. It performs real-time face tracking, mesh generation, head pose estimation, and gaze tracking using lightweight ONNX models. The application is designed to run locally on macOS (and other platforms) without cloud dependencies.

## 2. Quick Start for Agents

### Build

```bash
# Ensure models are downloaded
./get_model.sh

# Build with release profile (compiles swift sidecar automatically on macOS)
cargo build --release
```

### Run

```bash
# List cameras
cargo run --release --bin rusty-eyes -- --list

# Run with camera 0 and mirror mode
cargo run --release --bin rusty-eyes -- --cam-index 0 --mirror
```

### Tech Stack

- **Language**: Rust 2021
- **Computer Vision**: ONNX Runtime (`ort`) with CoreML support
- **Camera**: `nokhwa`
- **Windowing**: `minifb`
- **Overlay**: Swift (macOS Sidecar)

## 3. Specifications

Detailed project specifications are organized in the `specs/` directory:

- [Core Specification](specs/CORE_SPEC.md) - System architecture, pipelines, and **UI Controls (including Moondream/Overlay Toggles)**.
- [Calibration Specification](specs/CALIBRATION_SPEC.md) - Moondream2 calibration strategy.
- [Overlay Specification](specs/OVERLAY_SPEC.md) - Triple Cursor Logic and HUD.

## 4. History & Changelog

See [CHANGELOG.md](CHANGELOG.md) for a historical journal of changes, bug fixes, and design decisions. This file provides the "Why" behind the current state of the codebase.

See these documents for in-depth behaviors and architecture.
