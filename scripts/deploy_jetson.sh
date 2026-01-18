#!/bin/bash
set -e

REMOTE_HOST="jetsone"
REMOTE_DIR="rustyeyes3"

# Check if we are in the root of the repo
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Please run this script from the root of the repository."
    exit 1
fi

echo "[DEPLOY] Syncing code to $REMOTE_HOST:~/$REMOTE_DIR..."
# Using rsync to sync files. 
# --delete: delete extraneous files from dest dirs (optional, but good for clean sync)
# --exclude: skip large/unnecessary artifacts
rsync -avz --exclude 'target' --exclude 'venv' --exclude '.git' --exclude 'models' --exclude '__pycache__' ./ $REMOTE_HOST:~/$REMOTE_DIR/

echo "[DEPLOY] Running build on remote host..."
# We explicitly call bash to run the script
ssh -t $REMOTE_HOST "cd $REMOTE_DIR && bash run.sh"
