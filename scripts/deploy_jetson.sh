#!/bin/bash
set -e

REMOTE_HOST="jetsone"
REMOTE_DIR="dev/repos/rustyeyes3"

# Check if we are in the root of the repo
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Please run this script from the root of the repository."
    exit 1
fi

echo "[DEPLOY] Syncing code to $REMOTE_HOST:~/$REMOTE_DIR..."
# Using rsync to sync files. 
# --delete: delete extraneous files from dest dirs (optional, but good for clean sync)
# --exclude: skip large/unnecessary artifacts
echo "[DEPLOY] Triggering Pull, Build, and Restart on $REMOTE_HOST..."
ssh -t $REMOTE_HOST "cd $REMOTE_DIR && git config user.email 'deploy@bot.local' && git config user.name 'Deploy Bot' && git stash && git pull && cargo build --release --bin rusty-eyes && export DISPLAY=:0 && ./run.sh"
