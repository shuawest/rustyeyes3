#!/bin/bash
set -e

# Configuration
CARGO_TOML="Cargo.toml"
SERVER_PY="remote_server/server.py"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [major|minor|patch] \"Release message\""
    exit 1
fi

BUMP_TYPE=$1
MSG=$2

# 1. Get Current Version from Cargo.toml
CURRENT_VERSION=$(grep '^version =' $CARGO_TOML | sed 's/version = "\(.*\)"/\1/')
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# 2. Calculate New Version
if [ "$BUMP_TYPE" == "major" ]; then
    MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0
elif [ "$BUMP_TYPE" == "minor" ]; then
    MINOR=$((MINOR + 1)); PATCH=0
elif [ "$BUMP_TYPE" == "patch" ]; then
    PATCH=$((PATCH + 1))
else
    echo "Invalid bump type: $BUMP_TYPE"
    exit 1
fi

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "Bumping version: $CURRENT_VERSION -> $NEW_VERSION"

# 3. Update Files
# Rust
sed -i '' "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" $CARGO_TOML
# Python - Update distinct from Rust version in case of drift
sed -i '' "s/VERSION = \".*\"/VERSION = \"$NEW_VERSION\"/" $SERVER_PY

echo "Files updated."

# 4. Git Operations
# 4. Git Operations
git add -u  # Add all tracked files (including src changes)
git commit -m "Release v$NEW_VERSION: $MSG"
git tag "v$NEW_VERSION"
git push origin main --tags

echo "Git push complete."

# 5. Deploy Server
echo "Deploying to Server (jowestdgxe)..."
./remote_server/deploy.sh

# 6. Trigger Client Build
echo "Triggering Client Build (jetsone)..."
# Use client.sh to ensure environment variables (like DISPLAY) and logging are consistent
ssh jetsone "cd ~/dev/repos/rustyeyes3 && git pull && ./client.sh build"

echo "Release v$NEW_VERSION completed successfully!"
