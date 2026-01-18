# Remote Gaze Server

A Python-based gRPC server for offloading heavy face mesh and gaze estimation tasks from edge devices.

## Features
- **High Performance**: Uses MediaPipe for optimized CPU/GPU inference.
- **Streaming**: Bidirectional gRPC streaming for low-latency video processing.
- **Health Monitoring**: Dual-protocol health checks (gRPC standard + REST API).

## Setup

### Prerequisites
- Python 3.8+
- `pip`

### Installation
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
python3 server.py
# gRPC listening on :50051
# REST listening on :8080
```

### Firewall Configuration (Server)
Ensure the following ports are open on the firewall (e.g., using `ufw`):

```bash
sudo ufw allow 50051/tcp  # gRPC (Streaming + Health)
sudo ufw allow 8080/tcp   # REST API (Health + Metadata)
```

## Deployment
Use the included `deploy.sh` script to deploy to a remote host (e.g., DGX):

```bash
./deploy.sh
```
This will:
1. Copy files to the remote host.
2. Install dependencies (user-space).
3. Generate protobuf bindings.
4. Restart the server in the background.

## API

### Health Check (REST)
`GET /health`
```json
{
  "service": "GazeStreamService",
  "status": "SERVING",
  "backend": "MediaPipe FaceMesh",
  "stats": { ... }
}
```

### Protocol
See `gaze_stream.proto` for the gRPC service definition.
