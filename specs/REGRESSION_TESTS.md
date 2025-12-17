# Regression Prevention & Testing Protocol

This document outlines the Critical User Journeys (CUJs) and Visual Indicators that must be preserved during development. Any changes to the `main.rs` loop or `output.rs` must be verified against this list.

## Critical Visual Indicators (HUD)

The application provides real-time feedback that is essential for the user.

| Feature                | Visual Indicator                | Color / Shape         | Verification                                            |
| :--------------------- | :------------------------------ | :-------------------- | :------------------------------------------------------ |
| **Gaze Ray**           | Line projecting from eyes       | **Cyan** Line         | Must track user eye movement.                           |
| **Eye Center**         | Center of pupil/eye             | **Blue** Dot          | Must align with pupil.                                  |
| **Moondream Gaze**     | AI-predicted gaze target        | **Gold** Star/Cross   | Must appear when Key 7 is active. Must NOT be at (0,0). |
| **Face Box**           | Detected face area              | **Green** Rectangle   | Must surround face.                                     |
| **Calibration Target** | Last recorded calibration point | **Green** Crosshair   | Must appear at the exact screen coordinate clicked.     |
| **HUD Text**           | Status Info                     | Green Text (Top-Left) | Must show "LAST: (x,y) AT Time".                        |

## Regression Test Checklist

Before committing changes to `main.rs`, `moondream.rs`, or `output.rs`:

1.  **Startup**: Run `./run.sh`. Ensure window opens.
2.  **Tracking**: Move head. Verify **Cyan Ray** follows.
3.  **Moondream**:
    - Press `7`. Ensure console says "Moondream Continuous Mode: ACTIVE".
    - Verify **Gold Star** appears and moves (laggy is expected, stationary at 0,0 is FAIL).
    - Verify Console logs `[DATA] Moondream: (x, y) ...`.
4.  **Calibration**:
    - Press `9`. Verify HUD says "Calibration Mode: ON".
    - Click or Press Space.
    - Verify **Green Crosshair** appears exactly where mouse was.
    - Verify HUD Text updates with new timestamp.
5.  **Overlay**:
    - Press `6`. Verify Overlay Sidecar (if on macOS) is active/responsive.
