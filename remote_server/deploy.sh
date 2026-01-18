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
echo "[4/5] Note: Ensure port 50051 is open. You may need to run:"
echo "  ssh $HOST 'sudo ufw allow 50051/tcp'"

# 5. Start server
echo "[5/5] Starting server in background..."
# Kill existing instance if any
ssh $HOST "pkill -f 'python3 server.py' || true"
# Start new instance
ssh $HOST "cd $DEPLOY_DIR && nohup python3 server.py > server.log 2>&1 &"

echo ""
echo "=== Deployment Complete ==="
echo "Server running on: jowestdgxe:50051"
echo ""
echo "View logs:"
echo "  ssh $HOST 'tail -f $DEPLOY_DIR/server.log'"
