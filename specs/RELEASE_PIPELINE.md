# Release Pipeline Specification

## Objective
Enforce strict Semantic Versioning (SemVer) synchronization between the Client (Rust) and Server (Python) components. Ensure that every release is atomically committed, pushed, and deployed to prevent "version drift" and "forgotten push" scenarios.

## Versioning Policy
1.  **Unified Versioning**: Both Client and Server MUST share the exact same version string.
2.  **Atomic Bump**: A version bump must update:
    - `Cargo.toml` (package.version)
    - `remote_server/server.py` (VERSION constant)
3.  **Strict SemVer**:
    - `PATCH`: Backward-compatible bug fixes.
    - `MINOR`: New features (backward-compatible).
    - `MAJOR`: Breaking API changes (e.g., Protobuf schema changes).

## Release Workflow
The `scripts/release.sh` utility MUST be used for all releases. Manual edits to version files are discouraged.

### 1. Verification Phase
- Check git status (working directory must be clean).
- Verify successful local compilation (optional but recommended).

### 2. Version Bump Phase
- Update `Cargo.toml`.
- Update `server.py`.
- Update `CHANGELOG.md` (optional prompt).

### 3. Commit & Push Phase
- Git add modified files.
- Git commit with standardized message: `Release vX.Y.Z: [Summary]`.
- Git tag `vX.Y.Z`.
- Git push origin main --tags.

### 4. Deployment Phase (Authorized)
- **Server**: Automatically `rsync` code to `jowestdgxe`, regenerate protos, and restart service.
- **Client**: SSH into `jetsone`, `git pull`, and trigger `cargo build --release`.

## Automation Script (`scripts/release.sh`)
This script serves as the "source of truth" for the release process.

```bash
./scripts/release.sh [major|minor|patch] "Release message"
```

### Script Responsibilities:
1.  **Validation**: Ensure git cleanliness.
2.  **Bumping**: Parse current version, increment, write files.
3.  **Git Ops**: Commit, Tag, Push.
4.  **Deploy Server**: Execute `remote_server/deploy.sh` (or equivalent rsync logic).
5.  **Trigger Client**: SSH to Jetson to pull & build.

## Failure Recovery
- If Server deploy fails: The git tag exists, but deployment is partial. Re-run `deploy.sh`.
- If Client build fails: The git tag exists. Fix bug, create new PATCH release.
