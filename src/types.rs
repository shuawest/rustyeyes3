use serde::{Deserialize, Serialize};

/// Represents a single 3D point
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    #[allow(dead_code)]
    pub z: f32,
}

/// Represents the result of a face mesh inference
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Landmarks {
    pub points: Vec<Point3D>,
}

impl Landmarks {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationPoint {
    pub timestamp: u64,
    pub screen_x: f32,
    pub screen_y: f32,
    pub inference: Option<PipelineOutput>,
    pub moondream_result: Option<Point3D>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationProfile {
    // Linear Regression Coefficients
    // X_screen = c0 + c1*X_in + c2*Y_in
    // Y_screen = c3 + c4*X_in + c5*Y_in
    pub x_coeffs: Vec<f32>,
    pub y_coeffs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineOutput {
    Landmarks(Landmarks),
    FaceRects(Vec<Rect>),
    HeadPose(f32, f32, f32), // Yaw, Pitch, Roll
    Gaze {
        left_eye: Point3D,
        right_eye: Point3D,
        yaw: f32,
        pitch: f32,
        #[allow(dead_code)]
        roll: f32,
        #[allow(dead_code)]
        vector: Point3D,
        landmarks: Option<Landmarks>,
    },
}
