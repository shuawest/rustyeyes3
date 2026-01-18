#!/bin/bash
set -e

# Usage: ./scripts/bump_version.sh [major|minor|patch]
# Default is patch.

TYPE=${1:-patch}

CARGO_FILE="Cargo.toml"
SERVER_FILE="remote_server/server.py"

# 1. Read current version from Cargo.toml
CURRENT_VERSION=$(grep '^version =' $CARGO_FILE | sed -E 's/version = "([0-9]+\.[0-9]+\.[0-9]+)"/\1/')

if [ -z "$CURRENT_VERSION" ]; then
    echo "Error: Could not find version in $CARGO_FILE"
    exit 1
fi

echo "Current Version: $CURRENT_VERSION"

# 2. Split into parts
IFS='.' read -r -a PARTS <<< "$CURRENT_VERSION"
MAJOR="${PARTS[0]}"
MINOR="${PARTS[1]}"
PATCH="${PARTS[2]}"

# 3. Increment
if [ "$TYPE" == "major" ]; then
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
elif [ "$TYPE" == "minor" ]; then
    MINOR=$((MINOR + 1))
    PATCH=0
else
    PATCH=$((PATCH + 1))
fi

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "New Version:     $NEW_VERSION"

# 4. Update Cargo.toml (Client)
# Use perl for in-place edit to avoid sed differences between macOS/Linux
perl -pi -e "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" $CARGO_FILE

# 5. Update remote_server/server.py (Server)
perl -pi -e "s/VERSION = \"$CURRENT_VERSION\"/VERSION = \"$NEW_VERSION\"/" $SERVER_FILE

echo "Updated $CARGO_FILE and $SERVER_FILE to $NEW_VERSION"

# 6. Git commands (Optional, user might want to do this manually)
# echo "Don't forget to commit: git commit -am \"chore: bump version to $NEW_VERSION\""
