# Changelog

All notable changes to the `rustyeyes3` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Changelog**: Restored and backfilled `CHANGELOG.md` from git history parameters (`e040e71`).

## [0.2.1] - 2025-12-17 (Gaze Synchronization & Fonts)

### Added

- **Configurable Fonts (`d9df9a1`, `a36f3b3`)**: Added `font_family` to `config.json` and implemented `ttf.rs` rendering.
  - _Intent_: Allow users to customize the HUD appearance (e.g., matching OS sans-serif aesthetics) rather than being stuck with the pixelated bitmap font.
- **Captured Gaze Visualization (`4b9a574`)**: Added "Green Dot" indicator for `Captured` gaze.
  - _Intent_: Provide a stable "Ground Truth" visual that shows exactly where the user was looking when a Moondream frame was captured, allowing valid comparison with Moondream's "Gold" prediction.
- **Screen Coordinates in HUD**: Added `SCREEN: x, y` stats.
  - _Intent_: Enable direct numerical comparison between the main window's internal state and the Overlay's displayed coordinates.
- **Pupil-Based Gaze Vectors (`fbb707f`)**: Implemented dual gaze vectors originating from pupil centers.
  - _Intent_: Increase visualization accuracy by drawing rays from the actual eyes positions rather than a generic head center.

### Changed

- **Moondream Synchronization (`e040e71`)**: Moved pipeline processing _before_ Moondream dispatch.
  - _Intent_: Eliminated 1-frame lag. Previously, "Captured" gaze was from frame N-1 while the image was frame N, causing spatial drift during head movement.
- **Moondream Worker Channel (`a36f3b3`)**: Switched to `sync_channel(1)` with `try_send`.
  - _Intent_: Prevented UI freezes (infinite queue memory pressure) when the Python worker was slower than the video feed.
- **Visual Parameters (`3a8f65b`)**: Made mesh/gaze colors and sizes configurable in `config.json`.
  - _Intent_: Allow visual customization without recompilation.

### Fixed

- **Overlay Blank Data (`e040e71`)**: Added missing `update_moondream()` call.
  - _Intent_: Fixed bug where Overlay showed `----` despite valid inference results.
- **Config Persistence (`b3ce716`)**: Ensured `config.json` is created/updated with defaults on load.
  - _Intent_: robustness against schema changes; ensures users always have a valid config file.

## [0.2.0] - 2025-12-17 (UI & Configuration Overhaul)

### Added

- **Configuration System (`61b573f`)**: Implemented `config.rs` with `serde` serialization.
  - _Intent_: Move hardcoded constants (colors, toggles, timeouts) into a user-editable file.
- **Modular Toggles (`7600c20`)**: Replaced unified "Modes" with granular toggles (Mesh, Pose, Gaze, Mirror).
  - _Intent_: Give users finer control over what is displayed (e.g., just Gaze, no Mesh).
- **Visual Menu (`56fe999`)**: Added on-screen list of active toggles.
  - _Intent_: Improve UX by showing current state without requiring rote memorization of keys.

### Changed

- **Menu Styling (`c106dc4`, `37ad271`, `947be60`)**: Increased font size and contrast.
  - _Intent_: User feedback indicated headers were too small to read on high-res screens.
- **Mirror Mode Logic (`e68b9ac`, `ae8c542`)**: Implemented Horizontal Flip and inverted Yaw for Gaze Ray.
  - _Intent_: "Mirror Mode" is more intuitive for webcam usage. Inverting Yaw ensures the gaze ray points "into" the mirror correctly.

## [0.1.1] - 2025-12-17 (Calibration & Overlay Polish)

### Added

- **Calibration HUD (`5f1ce59`)**: Added status text for calibration capture.
  - _Intent_: Provide feedback when a data point is collected.
- **Moondream Visual Indicator (`c98e7fd`)**: Restored Gold Crosshair for Moondream results.
  - _Intent_: Regression fix; the indicator was lost during refactors. Added `REGRESSION_TESTS.md` to prevent recurrence.

### Fixed

- **HUD Scaling (`8663235`)**: Scaled crosshair coordinates by window/buffer ratio.
  - _Intent_: Fixed misalignment where drawing used raw camera coords (1080p) on a smaller window (900p).
- **Calibration Data (`551aea6`)**: Stripped heavy landmarks from saved JSON.
  - _Intent_: Reduce file size of calibration datasets.

## [0.1.0] - 2025-12-16 (Initial Release)

### Added

- **Core Pipelines (`9cf024d`, `23b9037`)**:
  - `FaceDetectionPipeline` (UltraFace).
  - `FaceMeshPipeline` (468 landmarks).
  - `PupilGazePipeline` (Darkest-pixel centroid).
  - `MoondreamOracle` (Async VLM).
- **Overlay Sidecar (`e66aaa6`, `489d85d`)**:
  - Swift-based transparent window for cross-application cursors.
  - Triple Cursor logic (Realtime, Moondream, Mouse).
- **Control Script (`9519f68`)**: Added `run.sh` and `get_model.sh`.
  - _Intent_: Simplified setup transparency.

### Specs

- **Specification Hierarchy (`c432c98`, `b765623`)**:
  - `CORE_SPEC.md`, `CALIBRATION_SPEC.md`, `OVERLAY_SPEC.md`.
  - _Intent_: Formalized requirements and architecture.
