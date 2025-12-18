# Changelog

All notable changes to the `rustyeyes3` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Unreleased - Refactor Research Directory**:
  - _Change_: Moved contents of `model_training` to `research/` directory.
  - _Intent_: Organize exploratory work and training scripts into a dedicated research folder.
- **Unreleased - Fix Calibration Toggle**:
  - _Change_: Implemented the logic for `Key9` to toggle `calibration_mode` boolean and added it to the HUD Menu.
  - _Intent_: The handler was previously empty, preventing the user from entering calibration mode.
  - _Approach_: Added state toggle `val = !val` in the input loop and added `"9", "Calibration"` to the `menu_items` array in the drawing loop.
- **Unreleased - Visual Cleanup**:
  - _Intent_: Declutter the main video feed and consolidate information on the overlay as requested.
  - _Approach_: Implemented `S <text>` protocol command in Swift overlay. Modified `main.rs` to construct the menu string and send it via IPC instead of drawing locally. Removed the redundant 5-corner debug loop in Swift.
- **Unreleased - HUD Refinement**:
  - _Change_: Restored 5-point HUD stats (Corners + Center) in Overlay. Refined positioning: Vertically centered Menu, shifted right-side stats closer to edge. Updated Menu style to UPPERCASE labels to match HUD font aesthetic.
  - _Intent_: Restore debug visibility, center controls, and ensure a unified visual design.
- **Unreleased - Moondream API return format fix**:
  - _Change_: Modified `moondream_server.py` to robustly handle `model.point()` return values (supporting both list-of-lists and list-of-dicts). Added detailed exception logging.
  - _Intent_: Fix `Moondream API error: 0` caused by `KeyError: 0` when accessing dictionary results as lists.
- **Unreleased - Dual Dot Visuals**:
  - _Change_: Implemented simultaneous "Pending" (Green/Red) and "Verified" (Green/Yellow) dots for Moondream capture history. Updated `overlay_sidecar.swift` sizes to match Blue Dot (Radius 50).
  - _Intent_: Provide clearer visual history of VLM requests vs results.
- **Unreleased - Coordinate Clamping**:

  - _Change_: Clamped gaze coordinates in `main.rs` to prevent visual anomalies (e.g. 9000+ Y-coord).
  - _Intent_: Prevent "missing dot" issues caused by transient ONNX glitches.

- **Unreleased - Unified Tracking & Cleanup**:
  - _Change_: Refactored `main.rs` to use a single coordinate calculation for both Realtime and Pending gaze, ensuring perfect synchronization. Removed legacy framebuffer drawing of dots to fix "little green ghost dot" artifact.
  - _Intent_: Fix "Green-Red dot not tracking" issue and remove visual clutter.
- **Unreleased - Debug Logging**:
  - _Change_: Enabled verbose request/response logging in `moondream.rs` (stdout) and `moondream_server.py` (stderr).
  - _Intent_: Provide visibility into the exact JSON payload being exchanged for debugging.
- **Unreleased - Moondream Stabilization**:
  - _Change_: Modified `moondream_server.py` to iterate through all logic prompts to prevent double-inference penalties. Updated visualization flow: Immediate Green/Red dot on capture, Green/Yellow dot upon completion + Cyan Moondream dot. Added "MOON: WATCHING..." status.
  - _Intent_: Fix "no coordinates" errors, reduce latency, and provide immediate visual feedback as requested.
- **Unreleased - Fix Mirror Mode Regression**:
  - _Change_: Restored `image::imageops::flip_horizontal_in_place` in the capture loop. Restored manual yaw inversion Logic (`-yaw`) in both drawing and Moondream dispatch.
  - _Intent_: Correctly mirror the video feed AND invert the gaze X-coordinate calculation so that looking Left (user) moves the dot Left (screen), matching mirror expectations.

### Documentation

- **`1c11d2c` - Formalize comprehensive changelog**:
  - _Change_: Created detailed CHANGELOG.md and updated AGENTS.md.
  - _Intent_: Establish a source of truth for project history to aid future AI agents in understanding the "why" behind architectural decisions.
  - _Approach_: Analyzed `git log` history, categorized commits by semantic impact, and wrote this file. Updated `AGENTS.md` to reference this file as the primary historical record.
  - _Approach_: Analyzed `git log` history, categorized commits by semantic impact, and wrote this file. Updated `AGENTS.md` to reference this file as the primary historical record.

## [0.3.0] - 2025-12-17 (CNN Gaze Models)

### Added

- **L2CS-Net & MobileGaze Integration**:
  - _Change_: Added fully compliant `Pipeline` implementations for L2CS-Net (ResNet50) and MobileGaze (MobileNetV2) ONNX models.
  - _Intent_: Provide robust, ML-based gaze estimation alternatives to the geometric pupil tracker.
- **Model Switching UI**:
  - _Change_: Added `Key4` toggle to cycle between `Pupil`, `L2CS`, `Mobile`, and `HeadPose` pipelines at runtime.
  - _Intent_: Allow rapid comparison of different model behaviors.
- **Model Downloader**:
  - _Change_: Created `scripts/setup_models.sh` to automatically fetch required `.onnx` assets.
  - _Intent_: simplify setup. Replaces `get_model.sh`.

### Changed

- **Gaze Output Units**:
  - _Change_: Standardized all pipelines to output **Degrees** (Yaw/Pitch) to match the rendering logic in `main.rs`.
- **Gaze Smoothing**:
  - _Change_: Integrated Exponential Moving Average (EMA) smoothing for L2CS and MobileGaze to reduce jitter.
- **Sensitivity Gain**:
  - _Change_: Applied a tunable gain factor (5.0x) to L2CS/MobileGaze outputs to map small eye movements to full-screen cursor control.
- **Vertical Calibration**:
  - _Change_: Applied a -12.0 degree pitch offset to compensate for typical webcam placement (top of screen looking down).

## [0.2.2] - 2025-12-17 (Performance & Sync)

### Fixed

- **`e040e71` - Fix Moondream overlay and gaze synchronization**:
  - _Change_: Reordered `main.rs` loop to run `pipeline.process()` _before_ dispatching the frame to the Moondream worker. Added missing `update_moondream()` call.
  - _Intent_: Eliminated the 1-frame synchronization lag (33ms) between the camera frame and the recorded gaze coordinates. Previously, the "Captured" gaze (Green Dot) represented the _previous_ frame's gaze, causing visual drift during head movement.
  - _Approach_: Moved the `pipeline.process()` call from the end of the loop (post-input) to the beginning (pre-dispatch). Captured the resulting `PipelineOutput` into a local variable `output` instead of relying on the stale `last_pipeline_output` state. Passed this fresh `output` directly to the `tx_frame` channel tuple `(img, current_gaze, ...)`. Added `win.update_moondream(mx, my)` inside the async result handler to piping data to the overlay process.
- **`a36f3b3` - Fix Moondream Freeze**:
  - _Change_: Switched `tx_frame` channel from unbounded to `sync_channel(1)` with `try_send`.
  - _Intent_: Prevented the main UI thread from freezing or stuttering. Previously, if the Python worker was slow, the unbounded channel would fill up with frames, causing memory pressure and potential blocking during allocation/GC.
  - _Approach_: Replaced `mpsc::channel()` with `mpsc::sync_channel(1)` in `main.rs`. Changed `tx_frame.send(...)` to `tx_frame.try_send(...)`. If the channel is full (worker busy), the frame is dropped immediately, prioritising main thread responsiveness over exhaustive processing.

### Added

- **`4b9a574` - Buffer and display captured gaze**:
  - _Change_: Added "Green Dot" indicator corresponding to `Captured` gaze coords.
  - _Intent_: Provide a stable "Ground Truth" visual for the user to compare against the "Gold Crosshair" (Moondream). This proves whether the drift is in the model or just temporal lag.
  - _Approach_: Introduced `captured_gaze_result: Option<(f32, f32)>` state variable. When the Moondream worker returns a result, it now passes back the gaze coordinates that were sent with the original frame request. The main loop draws a green circle at these accepted coordinates, ensuring the visual indicator represents the exact frame analyzed by the VLM.

## [0.2.1] - 2025-12-17 (Configuration & Visuals)

### Added

- **`d9df9a1` - Add `font_family` to configuration**:
  - _Change_: Added `ui.font_family` field to `config.json` and logic to load it.
  - _Intent_: Allow users to match the application's aesthetic to their OS (e.g., using "Geneva" or "Arial" instead of the default bitmap font).
  - _Approach_: Added `font_family: String` to `Config` struct in `config.rs`. Updated `ttf.rs` to attempt loading the specified font from `/System/Library/Fonts/` and `/Library/Fonts/`. Implemented a fallback chain to system generics if the specific TTF is not found.
- **`fbb707f` - Implement pupil-based gaze vectors**:
  - _Change_: Added secondary "Blue" gaze rays originating from pupil variance centers.
  - _Intent_: Improve visual debugging of the geometric solver.
  - _Approach_: Modified `PupilGazePipeline` to carry `left_pupil` and `right_pupil` Point2D structs through the `PipelineOutput`. Updated `main.rs` drawing loop to emit two `draw_line` calls originating from these dynamic pupil coordinates rather than the static mesh eye centers.
- **`3a8f65b` - Configurable visual parameters**:
  - _Change_: Exposed colors (Hex) and sizes (Pt) for mesh, gaze, and HUD in `config.json`.
  - _Intent_: Enable users to visually tweak the HUD for visibility against different backgrounds (e.g., dark vs light rooms) without recompiling.
  - _Approach_: Added fields `mesh_color_hex`, `mesh_dot_size`, etc., to `Config` struct. Updated `main.rs` to parse these hex strings (e.g., `#FF0000`) into `(u8, u8, u8)` tuples at runtime using a clearer `parse_hex` helper function.

### Fixed

- **`b3ce716` - Ensure config file persistence**:
  - _Change_: Forced `config.json` to be instantiated with defaults if missing.
  - _Intent_: Improving First-Run Experience (FRE) and ensuring stability.
  - _Approach_: Modified `Config::load()` to check for file existence. If `std::fs::read_to_string` fails, it now creates a `default()` instance, serializes it to JSON, and writes it to disk immediately, ensuring the user has a template to edit.

## [0.2.0] - 2025-12-17 (UI Overhaul)

### Added

- **`61b573f` - Implement config.json support**:
  - _Change_: Created `config.rs` with Serde and lazy_static generic patterns.
  - _Intent_: Decouple hardcoded "magic numbers" from the codebase, preparing for a user-distributable binary.
  - _Approach_: Defined a nested struct hierarchy (`Config` -> `UiConfig`, `Defaults`). Used `serde_json` for serialization. centralized all tunable constants (timeouts, colors, sizes) into this struct, replacing literal values in `main.rs`.
- **`7600c20` - Modular toggles**:
  - _Change_: Replaced mutually exclusive "Modes" (1,2,3) with additive boolean toggles.
  - _Intent_: Allow composing visualizations (e.g., "Gaze + Mesh" or "Just Gaze"), giving the user finer control.
  - _Approach_: Removed the `Mode` enum. Introduced independent `show_mesh`, `show_pose`, `show_gaze` booleans in `main.rs`. Updated key event handlers (`Key1`, `Key2`, `Key3`) to toggle these booleans logic (`val = !val`) instead of setting a state enum.
- **`56fe999` - Visual Menu**:
  - _Change_: Added a text-based HUD listing active keys `[1] MESH [ON]`.
  - _Intent_: Improve discoverability of controls so users don't need to read the manual to know which keys do what.
  - _Approach_: Created a static array of tuples `(Key, Label, &bool)` in `draw_ui`. Iterated over this array to draw formatted text lines `[{key}] {label} [{ON/OFF}]`, dynamically coloring the "ON" state Green and "OFF" state White.

### Changed

- **`ae8c542` - Invert yaw in mirror mode**:
  - _Change_: Multiplied Yaw by -1 when Mirror Mode is active.
  - _Intent_: Fixed a geometric bug where the Gaze Ray pointed "out" of the mirror instead of "into" it when the image was flipped.
  - _Approach_: In the drawing loop, checked `if mirror_mode`. If true, `draw_yaw = -calculated_yaw`. This mirroring is visual-only; the underlying data remains consistent with the head's physical rotation.
- **`e68b9ac` - Restore video feed & Mirror Toggle**:
  - _Change_: Added horizontal flip logic.
  - _Intent_: Standard webcam expectation (Mirroring) for easier hand-eye coordination.
  - _Approach_: Added `image::imageops::flip_horizontal_in_place` call on the raw camera frame buffer immediately after capture, conditioned on the `mirror_mode` boolean.
- **`947be60`, `37ad271`, `c106dc4` - Menu Styling**:
  - _Change_: Updated font size `5x7` -> `Scale 2`.
  - _Intent_: Fix legibility issues on high-DPI displays.
  - _Approach_: Replaced the basic bitmap font renderer with a scaled lookup. Instead of 1:1 pixel mapping, drew 2x2 pixel blocks for each set bit in the font bitmap, effectively doubling the size and increasing contrast.

## [0.1.1] - 2025-12-17 (Calibration & Polish)

### Fixed

- **`c98e7fd` - Restore Moondream visual indicator**:
  - _Change_: Re-implemented the drawing of the Gold Crosshair.
  - _Intent_: Regression fix. The indicator was lost during a previous refactor, making it impossible to verify if Moondream was working.
  - _Approach_: Added a specific drawing block in `main.rs` that checks `moondream_result`. If present, it maps the normalized (0..1) coordinates to screen dimensions (`x * w`, `y * h`) and draws a gold-colored crosshair (RGB `255, 215, 0`) at that location.
- **`8663235` - Scale crosshair coordinates**:
  - _Change_: Applied `buffer_width / window_width` scaling factor.
  - _Intent_: Fixed misalignment where coordinates were accurate to the _camera_ (1080p) but drawn at wrong offsets on a _window_ (e.g., 900p).
  - _Approach_: In `output.rs`, added logic to retrieve current window size from `minifb`. In `main.rs`, computed `scale_x = buf_w / win_w` and multiplied all drawing coordinates by this factor before rasterization.
- **`551aea6` - Strip landmarks from saved data**:
  - _Change_: Removed the 468-point mesh from the JSON saved to disk.
  - _Intent_: Reduce disk usage per sample (from ~15KB to ~200B) for large datasets.
  - _Approach_: Updated the `Serialize` struct for Calibration data to exclude the `landmarks` vector field. Only `yaw`, `pitch`, and `timestamp` are preserved.

### Added

- **`5f1ce59` - Calibration Feedback HUD**:
  - _Change_: Added "LAST CAPTURED: (x,y)" text to HUD.
  - _Intent_: Provide confirmation to the user that their "Spacebar" press actually registered a data point.
  - _Approach_: Added `last_calibration_point` state to `main.rs`. Updated the input handler for `Space` to update this variable. Added a HUD line that renders this coordinate formatted as text if it is `Some`.

## [0.1.0] - 2025-12-16 (Initial Release)

### Added

- **`489d85d` - Triple-cursor overlay**:
  - _Change_: Implemented the communication protocol between Rust and Swift.
  - _Intent_: Allow the "Blue" (Realtime) and "Gold" (Moondream) cursors to exist over the OS desktop, enabling actual cursor control testing.
  - _Approach_: Used standard input/output pipes (`stdin`) to send simple text commands (`G x y`, `M x y`) from the Rust parent process to the Swift sidecar child process. Swift parses these lines and updates the `NSWindow` positions.
- **`e66aaa6` - Gaze Mouse Overlay (Swift)**:
  - _Change_: Initial commit of `overlay_sidecar.swift`.
  - _Intent_: Create a transparent, click-through window for drawing cursors on macOS.
  - _Approach_: Wrote a standalone Swift application using `AppKit`. Used `NSPanel` with `styleMask: .nonactivatingPanel`, `isOpaque = false`, and `backgroundColor = .clear` to achieve a click-through overlay that floats above all other windows.
- **`124673a` - Async Moondream Integration**:
  - _Change_: Added `std::thread` spawning for the VLM model.
  - _Intent_: Prevent the heavy (2-3s) inference of Moondream from blocking the 60FPS video loop.
  - _Approach_: Created a dedicated worker thread using `std::thread::spawn`. Used `crossbeam` channels (or `std::sync::mpsc`) to send images from the main loop to the worker, and receive results back, ensuring the main UI loop never blocks on inference.
- **`23b9037` - Pupil Gaze Pipeline**:
  - _Change_: Implemented blob tracking for pupils.
  - _Intent_: The primary algorithmic innovation—using geometry rather than just ML for speed.
  - _Approach_: Implemented an OpenCV-like algorithm in pure Rust: Extract eye region of interest (ROI) -> Convert to Grayscale -> Apply Threshold -> Find Contour with lowest average brightness -> Calculate Centroid.
- **`9cf024d` - Initial Commit**:
  - _Change_: Scaffolding, `nokhwa` camera setup, `ort` bindings.
  - _Intent_: Foundation of the project.
  - _Approach_: Set up `Cargo.toml` with dependencies. Implemented basic `nokhwa` camera capture loop and `minifb` window display.
