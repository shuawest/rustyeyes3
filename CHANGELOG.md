# Changelog

All notable changes to the `rustyeyes3` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation

- **`1c11d2c` - Formalize comprehensive changelog**:
  - _Change_: Created detailed CHANGELOG.md and updated AGENTS.md.
  - _Intent_: Establish a source of truth for project history to aid future AI agents in understanding the "why" behind architectural decisions.

## [0.2.2] - 2025-12-17 (Performance & Sync)

### Fixed

- **`e040e71` - Fix Moondream overlay and gaze synchronization**:
  - _Change_: Reordered `main.rs` loop to run `pipeline.process()` _before_ dispatching the frame to the Moondream worker. Added missing `update_moondream()` call.
  - _Intent_: Eliminated the 1-frame synchronization lag (33ms) between the camera frame and the recorded gaze coordinates. Previously, the "Captured" gaze (Green Dot) represented the _previous_ frame's gaze, causing visual drift during head movement.
- **`a36f3b3` - Fix Moondream Freeze**:
  - _Change_: Switched `tx_frame` channel from unbounded to `sync_channel(1)` with `try_send`.
  - _Intent_: Prevented the main UI thread from freezing or stuttering. Previously, if the Python worker was slow, the unbounded channel would fill up with frames, causing memory pressure and potential blocking during allocation/GC.

### Added

- **`4b9a574` - Buffer and display captured gaze**:
  - _Change_: Added "Green Dot" indicator corresponding to `Captured` gaze coords.
  - _Intent_: Provide a stable "Ground Truth" visual for the user to compare against the "Gold Crosshair" (Moondream). This proves whether the drift is in the model or just temporal lag.

## [0.2.1] - 2025-12-17 (Configuration & Visuals)

### Added

- **`d9df9a1` - Add `font_family` to configuration**:
  - _Change_: Added `ui.font_family` field to `config.json` and logic to load it.
  - _Intent_: Allow users to match the application's aesthetic to their OS (e.g., using "Calibri" or "Arial" instead of the default bitmap font).
- **`fbb707f` - Implement pupil-based gaze vectors**:
  - _Change_: Added secondary "Blue" gaze rays originating from pupil variance centers.
  - _Intent_: Improve visual debugging of the geometric solver.
- **`3a8f65b` - Configurable visual parameters**:
  - _Change_: Exposed colors (Hex) and sizes (Pt) for mesh, gaze, and HUD in `config.json`.
  - _Intent_: Enable users to visually tweak the HUD for visibility against different backgrounds (e.g., dark vs light rooms) without recompiling.

### Fixed

- **`b3ce716` - Ensure config file persistence**:
  - _Change_: Forced `config.json` to be instantiated with defaults if missing.
  - _Intent_: Improving First-Run Experience (FRE) and ensuring stability.

## [0.2.0] - 2025-12-17 (UI Overhaul)

### Added

- **`61b573f` - Implement config.json support**:
  - _Change_: Created `config.rs` with Serde and lazy_static generic patterns.
  - _Intent_: Decouple hardcoded "magic numbers" from the codebase, preparing for a user-distributable binary.
- **`7600c20` - Modular toggles**:
  - _Change_: Replaced mutually exclusive "Modes" (1,2,3) with additive boolean toggles.
  - _Intent_: Allow composing visualizations (e.g., "Gaze + Mesh" or "Just Gaze"), giving the user finer control.
- **`56fe999` - Visual Menu**:
  - _Change_: Added a text-based HUD listing active keys `[1] MESH [ON]`.
  - _Intent_: Improve discoverability of controls so users don't need to read the manual to know which keys do what.

### Changed

- **`ae8c542` - Invert yaw in mirror mode**:
  - _Change_: Multiplied Yaw by -1 when Mirror Mode is active.
  - _Intent_: Fixed a geometric bug where the Gaze Ray pointed "out" of the mirror instead of "into" it when the image was flipped.
- **`e68b9ac` - Restore video feed & Mirror Toggle**:
  - _Change_: Added horizontal flip logic.
  - _Intent_: Standard webcam expectation (Mirroring) for easier hand-eye coordination.
- **`947be60`, `37ad271`, `c106dc4` - Menu Styling**:
  - _Change_: Updated font size `5x7` -> `Scale 2`.
  - _Intent_: Fix legibility issues on high-DPI displays.

## [0.1.1] - 2025-12-17 (Calibration & Polish)

### Fixed

- **`c98e7fd` - Restore Moondream visual indicator**:
  - _Change_: Re-implemented the drawing of the Gold Crosshair.
  - _Intent_: Regression fix. The indicator was lost during a previous refactor, making it impossible to verify if Moondream was working.
- **`8663235` - Scale crosshair coordinates**:
  - _Change_: Applied `buffer_width / window_width` scaling factor.
  - _Intent_: Fixed misalignment where coordinates were accurate to the _camera_ (1080p) but drawn at wrong offsets on a _window_ (e.g., 900p).
- **`551aea6` - Strip landmarks from saved data**:
  - _Change_: Removed the 468-point mesh from the JSON saved to disk.
  - _Intent_: Reduce disk usage per sample (from ~15KB to ~200B) for large datasets.

### Added

- **`5f1ce59` - Calibration Feedback HUD**:
  - _Change_: Added "LAST CAPTURED: (x,y)" text to HUD.
  - _Intent_: Provide confirmation to the user that their "Spacebar" press actually registered a data point.

## [0.1.0] - 2025-12-16 (Initial Release)

### Added

- **`489d85d` - Triple-cursor overlay**:
  - _Change_: Implemented the communication protocol between Rust and Swift.
  - _Intent_: Allow the "Blue" (Realtime) and "Gold" (Moondream) cursors to exist over the OS desktop, enabling actual cursor control testing.
- **`e66aaa6` - Gaze Mouse Overlay (Swift)**:
  - _Change_: Initial commit of `overlay_sidecar.swift`.
  - _Intent_: Create a transparent, click-through window for drawing cursors on macOS.
- **`124673a` - Async Moondream Integration**:
  - _Change_: Added `std::thread` spawning for the VLM model.
  - _Intent_: Prevent the heavy (2-3s) inference of Moondream from blocking the 60FPS video loop.
- **`23b9037` - Pupil Gaze Pipeline**:
  - _Change_: Implemented blob tracking for pupils.
  - _Intent_: The primary algorithmic innovation—using geometry rather than just ML for speed.
- **`9cf024d` - Initial Commit**:
  - _Change_: Scaffolding, `nokhwa` camera setup, `ort` bindings.
  - _Intent_: Foundation of the project.
