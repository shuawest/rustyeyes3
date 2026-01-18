# Distributed Gaze Processing Implementation

We have successfully migrated `rusty-eyes` to a distributed architecture using gRPC, enabling high-performance remote processing on DGX Spark while maintaining robust local capabilities on the Jetson Nano.

## Key Features

### 1. Hybrid Processing Pipeline
- **Default**: Runs local ONNX inference on Jetson.
- **Remote**: When enabled (`[0]` key or config), streams video to DGX.
- **Fallback**: Automatically falls back to local processing if the remote server is unreachable or healthy checks fail.

### 2. gRPC Streaming Architecture
- **Protocol**: Bidirectional gRPC streaming (`StreamGaze`).
- **Data**: Video frames (JPEG encoded) $\rightarrow$ Server $\rightarrow$ FaceMesh/Gaze Results.
- **Latency**: Minimized using `tokio` async streaming and connection pooling.

### 3. Dual-Protocol Health Checks
The remote server now exposes health status via two protocols for maximum observability:
- **gRPC**: Standard `grpc.health.v1` service (Port 50051).
- **REST**: Lightweight JSON endpoint `GET /health` (Port 8080).

### 4. Developer Experience
- **Makefile**: Automation for system dependency setup (`make setup-apt`, `make setup-dnf`).
- **Configuration**: Support for `remote_dgx_url` in `config.json`.
- **Deployment**: `deploy.sh` script for zero-friction updates to the DGX host.

---

## Verification Results

### Build Validation
- **Local (MacOS)**: `cargo build --release` ✅
- **Jetson (Linux)**: `cargo build --release --no-default-features` (Verified `pkg-config` fix for GStreamer).
- **Server (Python)**: `deploy.sh` verified, dependencies installed.

### Interface Testing
- **Overlay**: Menu shows `[0] Remote`. Toggling enables/disables streaming.
- **Health Check**: `curl http://jowestdgxe:8080/health` returns valid JSON status.

## Usage

### How to Test (End-to-End)

1.  **Verify Server is Running (DGX)**
    From your Jetson (or any machine on the network), check the server health:
    ```bash
    curl http://jowestdgxe:8080/health
    # Expected: {"status":"SERVING", ...}
    ```

2.  **Start Client (Jetson)**
    Run the application with the remote URL:
    ```bash
    ./target/release/rusty-eyes --remote-dgx http://jowestdgxe:50051
    ```

3.  **Verify Hybrid Operation**
    - **Visual Feedback**: The overlay should appear.
    - **Toggle Remote**: Press `[0]` on your keyboard.
        - **OFF**: Status line should say "Local".
        - **ON**: Status line should say "Remote" (and potentially show lower latency/higher quality if MediaPipe is superior).
    - **Fallback Test**:
        - While running, kill the server (`ssh jowestdgxe 'tmux kill-session -t gaze_server'`).
        - The client should automatically revert to "Local" (or just continue working using local inference) without crashing.

**On DGX (Server):**
```bash
cd ~/gaze-streaming-server
python3 server.py
# Logs at server.log
```
