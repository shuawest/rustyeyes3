# RustyEyes3 🦀👁️

**RustyEyes3** is a prototype for controlling a mouse cursor using eye gaze detection. The prototype uses rust as the harness for low latency and efficient usage of system resources. A screen overlay is provided to display position of the eye gaze cursor.

The mission is to make a cursor track the position of eye gaze to accurately align with the desired position on the screen with low amounts of jitter and latency. High precision tracking is a bonus.

## Design Goals and Constraints

Dual or single camera - eye gaze tracking typically requires multiple cameras for triangulation and depth information. Dual cameras should be used if that is the best approach. Being able to function with a single camera is a bonus.

Infrared camera - low light impacts tracking performance, so being able to work with low cost infrared cmos camera is benefitial.

Target Devices - the target devices are low cost and low power devices such as a Raspberry Pi or NVIDIA Jetson Nano with a touch screen display. Working on macos or other laptops will help with development and testing.

Cross platform - the target platform is macOS, and linux (fedora and ubuntu). Working on windows is benefitial but not required.

Disconnected First - All inference runs locally. The application should not require an internet connection to function.

AI acceleration - Use any accelerated runtime to use GPUs/NPUs/etcs on the local device. It currently uses the ONNX Runtime to provide Face Mesh (468 pts), Head Pose Estimation, and Hybrid Gaze Tracking locally on the machine.

---

## Features

- **⚡ Blazing Fast**: Processes video streams at native camera framerates (120FPS+ tested on M1 Max).
- **🔒 Privacy First**: All inference runs locally. No data leaves your device.
- **🧩 Modular Architecture**: Switch between pipelines instantly:
  - `[1]` **Face Mesh**: 468-point 3D facial landmarks (MediaPipe compatible).
  - `[2]` **Face Detection**: Fast bounding box detection (UltraFace).
  - **[3] Head Pose**: 6DOF orientation (Yaw, Pitch, Roll) using WHENet.
  - **[4] Head Gaze**: Geometric gaze estimation based on head orientation.
  - **[5] Pupil Gaze**: Precision gaze tracking using Computer Vision pupil blob detection + Head Pose.
- **📹 Camera Support**: Automatic enumeration and selection of USB/Built-in cameras.
- **🪞 Digital Mirror**: Camera output is flipped horizontally by default, creating a natural "mirror" experience (looking left appears left on screen).
- **🔮 VLM Gaze Verification**: Moondream2 integration via Python sidecar for experimental AI-based gaze detection comparison.

## Quick Start

### Prerequisites

1.  **Rust** (1.70+): Install from [rustup.rs](https://rustup.rs)
2.  **Python 3.9+**: For Moondream2 VLM integration
3.  **macOS**: Currently overlay requires macOS (Swift sidecar)

### Setup

**1. Clone and Build Rust Components:**

```bash
git clone https://github.com/shuawest/rustyeyes3.git
cd rustyeyes3
cargo build --release
```

**2. Setup Python Environment (for Moondream2):**

```bash
# Create virtual environment
python3 -m venv venv

# Install dependencies (~100MB download)
./venv/bin/pip install -r scripts/requirements.txt
```

**3. Download Moondream2 Model (Optional but Recommended):**

```bash
# Pre-download model (~3.7GB, requires internet)
./venv/bin/python3 scripts/download_moondream.py
```

**Why pre-download?**

- Avoids 30-60 second delay on first run
- Verify download succeeded before testing
- Can download during setup, run app offline later

**Note**: If skipped, model auto-downloads on first Moondream activation (press `7`).

**4. Test Python Server (Optional):**

```bash
./scripts/test_server.sh
```

Expected output: JSON response with gaze analysis.

### Running the Application

**Basic Usage:**

```bash
cargo run --release
```

**With Custom Camera:**

```bash
cargo run --release -- --cam-index 1
```

### Operating the Application

**Real-Time Gaze Tracking:**

1.  Launch application (see above)
2.  Press `5` - Activate **Pupil Gaze** (best accuracy)
3.  Press `6` - Toggle **Overlay** (shows gaze cursors on screen)
4.  Move your eyes - watch the Blue/Red cursor track your gaze!

**Moondream VLM Comparison:**

1.  Ensure overlay is active (press `6`)
2.  Press `7` - Toggle **Moondream Mode**
3.  Wait 2-5 seconds per inference
4.  Watch three cursors:
    - **Blue/Red**: Real-time ONNX gaze (60 FPS)
    - **Green/White**: Captured snapshot at Moondream trigger
    - **Cyan/Gold**: Moondream2 VLM prediction
5.  Compare accuracy in console output

**Overlay HUD:**

- Shows coordinates in 5 locations (corners + center)
- Real-time, Captured, and Moondream positions
- Updated automatically as you look around

## Controls

| Key     | Function                                |
| :------ | :-------------------------------------- |
| **0**   | Switch to **Combined Mode** (Default)   |
| **1**   | Switch to **Face Mesh**                 |
| **2**   | Switch to **Face Detection**            |
| **3**   | Switch to **Head Pose**                 |
| **4**   | Switch to **Head Gaze** (Geometric)     |
| **5**   | Switch to **Pupil Gaze** (CV Precision) |
| **6**   | Toggle **Overlay** (macOS Sidecar)      |
| **7**   | Toggle **Moondream Mode** (Calibration) |
| **ESC** | Quit Application                        |

## Architecture

**Core Components:**

1. **Rust Application** (`src/main.rs`): Camera capture, ONNX inference, overlay management
2. **Python Server** (`scripts/moondream_server.py`): Moondream2 VLM for gaze prediction
3. **Swift Overlay** (`src/overlay_sidecar.swift`): macOS transparent window with triple-cursor display

**Communication:**

- Rust ↔ Python: JSON over stdin/stdout (subprocess)
- Rust ↔ Overlay: JSON over stdin (Swift subprocess)
- All local, no network required

## Troubleshooting

**"Failed to spawn Python server":**

- Ensure venv is created: `ls venv/bin/python3`
- Reinstall dependencies: `./venv/bin/pip install -r scripts/requirements.txt`

**Model download stuck:**

- Check internet connection
- Check disk space (~4GB free needed)
- Manually download: Visit https://huggingface.co/vikhyatk/moondream2

**Slow Moondream inference:**

- Expected on CPU: 2-5 seconds per frame
- GPU acceleration: Install PyTorch with CUDA support
- Reduce frequency: Moondream is experimental, not real-time

**Overlay not appearing:**

- Ensure macOS permissions for Accessibility
- Try toggling: Press `6` twice
- Check console for Swift errors

## Documentation

- **[AGENTS.md](AGENTS.md)**: Complete specification and architecture
- **[specs/CORE_SPEC.md](specs/CORE_SPEC.md)**: Core functionality specification
- **[specs/OVERLAY_SPEC.md](specs/OVERLAY_SPEC.md)**: Overlay and HUD details
- **[scripts/README.md](scripts/README.md)**: Python server documentation

## License

MIT License.
