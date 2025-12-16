# RustyEyes3 👁️🦀

> **RustyEyes3** is a high-performance, real-time Facial Intelligence toolkit written in Rust. It utilizes ONNX Runtime to provide Face Mesh (468 pts), Head Pose Estimation, and Hybrid Gaze Tracking locally on your machine.

---

## Features

-   **⚡ Blazing Fast**: Processes video streams at native camera framerates (120FPS+ tested on M1 Max).
-   **🔒 Privacy First**: All inference runs locally. No data leaves your device.
-   **🧩 Modular Architecture**: Switch between pipelines instantly:
    -   `[1]` **Face Mesh**: 468-point 3D facial landmarks (MediaPipe compatible).
    -   `[2]` **Face Detection**: Fast bounding box detection (UltraFace).
    -   **[3] Head Pose**: 6DOF orientation (Yaw, Pitch, Roll) using WHENet.
    -   **[4] Head Gaze**: Geometric gaze estimation based on head orientation.
    -   **[5] Pupil Gaze**: Precision gaze tracking using Computer Vision pupil blob detection + Head Pose.
-   **📹 Camera Support**: Automatic enumeration and selection of USB/Built-in cameras.
-   **🪞 Mirror Mode**: Built-in horizontal flip for self-cam usage.

## Installation

### Prerequisites
-   [Rust Toolchain](https://rustup.rs/) (latest stable)
-   `curl` (for model downloading)

### Setting Up

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
    cargo run --release -- --list
    ```

    Start the app (Camera index 0, with Mirroring):
    ```bash
    cargo run --release -- --cam-index 0 --mirror
    ```

## Controls

| Key | Function |
| :--- | :--- |
| **1** | Switch to **Face Mesh** |
| **2** | Switch to **Face Detection** |
| **3** | Switch to **Head Pose** |
| **4** | Switch to **Head Gaze** (Geometric) |
| **5** | Switch to **Pupil Gaze** (CV Precision) |
| **ESC** | Quit Application |

## Requirements
See [REQUIREMENTS.md](REQUIREMENTS.md) for detailed specification.

## License
MIT License.
