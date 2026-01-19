use crate::types::PipelineOutput;
use anyhow::Result;
use image::{ImageBuffer, Rgb};

pub trait Pipeline {
    fn name(&self) -> String;
    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>>;
    // We could move drawing to the OutputSink, but for now the pipeline knows best how to visualize its output?
    // Or simpler: Pipeline returns Output, Main/Output module draws it.
    // Let's stick to returning Output. The Output module (window) will need to know how to draw PipelineOutput.
}

// Dummy pipeline when ONNX is not available
pub struct DummyPipeline {
    frame_count: u32,
}

impl DummyPipeline {
    pub fn new() -> Self {
        Self { frame_count: 0 }
    }
}

impl Pipeline for DummyPipeline {
    fn name(&self) -> String {
        "No ONNX (Simulated Gaze)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        use crate::types::Point3D;

        self.frame_count += 1;

        // Simulate gaze moving in a circular pattern
        let t = (self.frame_count as f32) * 0.05; // Slow rotation
        let yaw = (t.cos() * 20.0).clamp(-30.0, 30.0); // -30 to +30 degrees
        let pitch = (t.sin() * 15.0).clamp(-25.0, 25.0); // -25 to +25 degrees

        // Simulate eye positions (center of frame)
        let width = frame.width() as f32;
        let height = frame.height() as f32;
        let center_x = width / 2.0;
        let center_y = height / 2.0;

        let left_eye = Point3D {
            x: center_x - 30.0,
            y: center_y,
            z: 0.0,
        };

        let right_eye = Point3D {
            x: center_x + 30.0,
            y: center_y,
            z: 0.0,
        };

        Ok(Some(PipelineOutput::Gaze {
            left_eye,
            right_eye,
            yaw,
            pitch,
            roll: 0.0,
            vector: Point3D {
                x: yaw,
                y: pitch,
                z: 0.0,
            },
            landmarks: None,
        }))
    }
}
