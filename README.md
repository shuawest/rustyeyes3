# RustyEyes3 👁️🦀

**RustyEyes3** is a prototype for controlling a mouse cursor using eye gaze detection. The prototype uses rust as the harness for low latency and efficient usage of system resources. It uses the ONNX Runtime to provide Face Mesh (468 pts), Head Pose Estimation, and Hybrid Gaze Tracking locally on the machine. A screen overlay is provided for macOS to display position of the eye gaze cursor.

The mission is to make a cursor track the position of eye gaze to accurately align with the desired position on the screen with low amounts of jitter and latency. High precision tracking is a bonus.

## Design Goals and Constraints

Dual or single camera - eye gaze tracking typically requires multiple cameras for triangulation and depth information. Dual cameras should be used if that is the best approach. Being able to function with a single camera is a bonus.

Infrared camera - low light impacts tracking performance, so being able to work with low cost infrared cmos camera is benefitial.

Target Devices - the target devices are low cost and low power devices such as a Raspberry Pi or NVIDIA Jetson Nano with a touch screen display. Working on macos or other laptops will help with development and testing.

Cross platform - the target platform is macOS, and linux (fedora and ubuntu). Working on windows is benefitial but not required.

Disconnected First - All inference runs locally. The application should not require an internet connection to function.

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
- **🪞 Mirror Mode**: Built-in horizontal flip for self-cam usage.

## Installation

### Prerequisites

- [Rust Toolchain](https://rustup.rs/) (latest stable)
- `curl` (for model downloading)

### Setting Up

0.  **Update Rust**:

    ```bash
    rustup update
    cargo --version
    rustc --version
    ```

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/shuawest/rustyeyes3.git
    cd rustyeyes3
    ```

2.  **Download Models**:
    Run the included script to fetch the required ONNX models (FaceMesh, Detect, HeadPose) from public model zoos.

    ```bash
    chmod +x get_model.sh
    ./get_model.sh
    ```

3.  **Run**:
    List available cameras:

    ```bash
    cargo run --release --bin rusty-eyes -- --list
    ```

    Start the app (Camera index 0, with Mirroring):

    ```bash
    cargo run --release --bin rusty-eyes -- --cam-index 0 --mirror
    ```

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
| **ESC** | Quit Application                        |

## Requirements

See [AGENTS.md](AGENTS.md) for detailed specification.

## License

MIT License.
