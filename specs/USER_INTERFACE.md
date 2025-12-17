# User Interface Specification

## Overview

Rusty Eyes now employs a modular **Toggle-based** interface instead of exclusive modes. This allows users to mix and match visualizations (Face Mesh, Head Pose, Eye Gaze) freely. Additionally, persistent configuration is supported via `config.json`.

## Control System

### Feature Toggles

Instead of selecting a single "Mode", users can toggle individual features on and off.

| Key | Feature         | Description                                                                                                                   | Default |
| :-- | :-------------- | :---------------------------------------------------------------------------------------------------------------------------- | :------ |
| `1` | **Face Mesh**   | Visualizes the 468-point face mesh (Red dots).                                                                                | `ON`    |
| `2` | **Head Pose**   | Visualizes the 3D head orientation (Green lines).                                                                             | `ON`    |
| `3` | **Eye Gaze**    | Visualizes the computed gaze vector (Cyan ray).                                                                               | `OFF`   |
| `5` | **Mirror Mode** | Flips the video feed horizontally. **Crucially, it also inverts the Yaw calculation** so that looking Left visuals move Left. | `ON`    |
| `6` | **Overlay**     | Toggles the secondary overlay window helper.                                                                                  | `ON`    |
| `7` | **Moondream**   | Toggles the Vision Language Model analysis (Async).                                                                           | `OFF`   |

### System Controls

| Key   | Action        | Description                                 |
| :---- | :------------ | :------------------------------------------ |
| `ESC` | **Quit**      | Exits the application.                      |
| `9`   | **Calibrate** | Toggles Data Collection / Calibration mode. |

## Visual HUD

The on-screen menu displays the current status of all toggles:

```
[1] Face Mesh [ON]
[2] Head Pose [ON]
[3] Eye Gaze  [OFF]
[5] Mirror    [ON]

[6] Overlay   [ON]
...
```

- **Active** items are displayed in **Green**.
- **Inactive** items are displayed in **White**.
- The menu is rendered at **2x scale** for readability using a custom 5x7 bitmap font.

## Configuration System (`config.json`)

At startup, `rusty-eyes` looks for `config.json` in the current directory. If not found, it creates one with default values.

### Schema

```json
{
  "defaults": {
    "show_mesh": true,
    "show_pose": true,
    "show_gaze": false,
    "show_overlay": true,
    "mirror_mode": true,
    "moondream_active": false,
    "head_pose_length": 150.0
  },
  "ui": {
    "menu_scale": 2,
    "font_size_pt": 12,
    "font_family": "Monospace",
    "mesh_dot_size": 2,
    "mesh_color_hex": "#FF0000"
  }
}
```

**\*Note**: `font_family` is currently just a configuration placeholder for external tools (Overlay) or future rendering updates. The internal HUD uses a built-in bitmap font.\*

### persistence

Currently, the configuration file is **Read-Only** at startup. Toggling features at runtime does _not_ write back to the file (to avoid overwriting preferred defaults with temporary changes), but this can be changed in future versions if "Save Settings" is requested.
