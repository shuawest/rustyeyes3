#!/bin/bash
# deepstream/run_face_pipeline.sh
# Runs DeepStream face detection pipeline receiving from Jetson stream

CONFIG_FILE=${1:-config_face_mesh.txt}

echo "Starting DeepStream Face Pipeline..."
echo "Config: ${CONFIG_FILE}"

# Run DeepStream app in Docker container
docker run --rm -it \
    --gpus all \
    --net=host \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/deepstream:7.0-triton-multiarch \
    deepstream-app -c /workspace/${CONFIG_FILE}
