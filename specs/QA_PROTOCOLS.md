# QA Protocols for Rusty Eyes 3

This document outlines manual verification steps to be performed before releasing changes, especially for refactors affecting core pipelines.

## 1. Mirror Mode Verification

**Risk Area**: `main.rs` frame flipping and coordinate logic.

**Protocol**:

1. Launch app (`cargo run`).
2. Ensure you are in default mode (Mesh On).
3. Toggle Mirror Mode [5].
   - [ ] Verify the video feed flips horizontally.
4. Toggle Eye Gaze [3] -> ON.
5. **Look Left** (physically turn head left).
   - [ ] **Dot Movement**: The Blue Gaze dot should move to the **LEFT** side of the screen.
   - [ ] **Reason**: In a mirror, if you look to your left, your reflection looks to its left (which is screen left).
6. **Look Right** (physically turn head right).
   - [ ] **Dot Movement**: The Blue Gaze dot should move to the **RIGHT** side of the screen.
7. Toggle Mirror Mode [5] -> OFF.
   - [ ] **Look Left**: Dot should move **RIGHT** (inverted, camera perspective).

## 2. Moondream Integration

**Risk Area**: Python IPC, Coordinate Parsing, Latency.

**Protocol**:

1. Toggle Moondream [7] -> ON.
   - [ ] Status Menu should show `MOONDREAM: ON`.
2. Wait for a capture (Green Dot appears).
   - [ ] **Immediate Feedback**: A Green Dot with Red Center should appear _instantly_ when status says "MOON: WATCHING...".
3. Wait for completion (~2-5s).
   - [ ] **Result**:
     - Green Dot changes to Yellow Center.
     - Cyan Dot appears.
   - [ ] **Accuracy**: Cyan Dot should be reasonably close to Green Dot.

## 3. Calibration

**Risk Area**: Coordinate mapping, File I/O.

### Calibration Verification

- [ ] Toggle Calibration Mode `[9]`.
- [ ] Press Spacebar while looking at various points.
- [ ] Verify console logs: `[CALIBRATION] Captured Point...`
- [ ] Verify HUD shows: `LAST CAL: x, y`.

### Visual Feedback & Dot Semantics

- [ ] **Blue Dot**: Move head. Dot tracks in real-time. Mirror mode `[5]` flips movement correctly.
- [ ] **Moondream Cycle**:
  - Enable `[7]`.
  - **Capture Start**: A large **Green Dot (Red Center)** appears immediately at Blue Dot location (Pending).
  - **Capture End**: After ~5-10s, the _Pending_ dot disappears. A **Green Dot (Yellow Center)** (Verified) and **Cyan Dot (Yellow Center)** (Result) appear.
  - **Continuous**: If a new cycle starts before the old one fades (if configured?), you may see both Pending and Verified dots simultaneously.
  - **Verification**: Ensure the "Verified" dot matches the location where the "Pending" dot WAS. They represent the same capture.
