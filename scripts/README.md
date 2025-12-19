# Scripts Documentation

This directory contains utilities for model management, data preparation, and the Python VLM server.

## Script Index

- `setup_models.sh`: Downloads required ONNX models (L2CS, FaceMesh, etc).
- `prepare_dataset.py`: Processes `calibration_data/` into a dataset for custom model training.
- `train_remote.py`: PyTorch training script for fine-tuning L2CS on a GPU server.
- `moondream_server.py`: The VLM verification server.

## Python Moondream2 Server Setup

## Installation

1. Install Python dependencies:

```bash
pip3 install -r scripts/requirements.txt
```

2. **Pre-download Moondream2 model** (Recommended):

```bash
./venv/bin/python3 scripts/download_moondream.py
```

This downloads ~3.7GB to `~/.cache/huggingface/`. If skipped, model auto-downloads on first use.

## Testing the Server Standalone

Test the Python server before running the full Rust application:

```bash
# Start the server
python3 scripts/moondream_server.py

# In another terminal, send a test request:
echo '{"image_path": "/path/to/test_image.jpg", "timestamp": 0}' | python3 scripts/moondream_server.py
```

Expected output:

```json
{
  "status": "success",
  "response": "looking at the center of the screen",
  "timestamp": 0,
  "prompt_used": "Where is the person looking? Give me screen coordinates."
}
```

## Running with Rust

```bash
cargo run --release
```

Press `7` to toggle Moondream mode. The Rust app will automatically:

1. Launch the Python server subprocess
2. Send frames for gaze detection
3. Display results in the overlay

## Troubleshooting

**"Failed to spawn Python server"**

- Ensure Python 3.8+ is installed: `python3 --version`
- Ensure dependencies are installed: `pip3 list | grep transformers`

**"Moondream error: ..."**

- Check stderr output (Python debug logs are shown)
- Ensure sufficient disk space for model (~4GB)
- Check internet connection (first run downloads model)

**Slow inference**

- Expected: ~2-5 seconds per frame on CPU
- GPU acceleration: Install `torch` with CUDA support if available
- Model runs in FP16 on GPU, FP32 on CPU

## How It Works

1. **Rust → Python**: JSON request via stdin with image path and timestamp
2. **Python Processing**: Load image, run Moondream2 inference with gaze prompt
3. **Python → Rust**: JSON response via stdout with natural language gaze description
4. **Rust Parsing**: Extract coordinates or direction from text response
