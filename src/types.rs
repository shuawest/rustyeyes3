/// Represents a single 3D point
#[derive(Debug, Clone, Copy, Default)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    #[allow(dead_code)]
    pub z: f32,
}

/// Represents the result of a face mesh inference
#[derive(Debug, Clone, Default)]
pub struct Landmarks {
    pub points: Vec<Point3D>,
}

impl Landmarks {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }
}

#[derive(Debug, Clone)]
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
    },
}
