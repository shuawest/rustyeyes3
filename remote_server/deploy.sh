#!/bin/bash
# Deploy gRPC Gaze Streaming Server to DGX Spark (jowestdgxe) - User Space Version

set -e

HOST="jowestdgxe"
DEPLOY_DIR="~/gaze-streaming-server"

echo "=== Deploying to $HOST (User Space) ==="

# 1. Copy files
echo "[1/5] Copying files..."
ssh $HOST "mkdir -p $DEPLOY_DIR"
scp remote_server/gaze_stream.proto remote_server/requirements.txt remote_server/server.py $HOST:$DEPLOY_DIR/

# 2. Install dependencies
echo "[2/5] Installing Python dependencies..."
ssh $HOST "cd $DEPLOY_DIR && pip3 install --user -r requirements.txt --break-system-packages || pip3 install --user -r requirements.txt"

# 3. Generate protobuf code
echo "[3/5] Generating protobuf code..."
ssh $HOST "cd $DEPLOY_DIR && python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gaze_stream.proto"

# 4. Check firewall (informational)
echo "[4/5] Note: Ensure ports 50051 (gRPC) and 8080 (REST) are open. You may need to run:"
echo "  ssh $HOST 'sudo ufw allow 50051/tcp'"
echo "  ssh $HOST 'sudo ufw allow 8080/tcp'"

# 5. Start server
echo "[5/5] Starting server in background..."
# Start new instance via tmux for persistence
echo "Starting tmux session 'gaze_server'..."
ssh $HOST "tmux kill-session -t gaze_server || true"
ssh $HOST "cd $DEPLOY_DIR && tmux new-session -d -s gaze_server 'python3 server.py > server.log 2>&1'"

echo ""
echo "=== Deployment Complete ==="
echo "Server running on: jowestdgxe:50051 (in tmux session 'gaze_server')"
echo ""
echo "View logs:"
echo "  ssh $HOST 'tail -f $DEPLOY_DIR/server.log'"
echo "Attach to console:"
echo "  ssh $HOST 'tmux attach -t gaze_server'"
