use anyhow::Result;
use image::{imageops::FilterType, ImageBuffer, Rgb};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;

use crate::detector::FaceDetector;
use crate::pipeline::Pipeline;
use crate::types::{Landmarks, PipelineOutput, Point3D, Rect};

pub struct FaceMeshPipeline {
    mesh_session: Option<Session>,
    detector: Option<FaceDetector>,
    start_time: std::time::Instant,
}

impl FaceMeshPipeline {
    pub fn new(model_path: &str, detector_path: &str) -> Result<Self> {
        // We expect main.rs to pass the Mesh model path, but we also need the Detector path.
        // For simplicity, we'll hardcode or deduce detector path. "face_detection.onnx"

        let detector = if Path::new(detector_path).exists() {
            println!("Loading Face Detector...");
            Some(FaceDetector::new(detector_path)?)
        } else {
            println!("Face Detector not found. Accuracy will be poor.");
            None
        };

        if Path::new(model_path).exists() {
            println!("Loading Face Mesh from {}...", model_path);
            let mesh_session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(model_path)?;

            Ok(Self {
                mesh_session: Some(mesh_session),
                detector,
                start_time: std::time::Instant::now(),
            })
        } else {
            Ok(Self {
                mesh_session: None,
                detector: None,
                start_time: std::time::Instant::now(),
            })
        }
    }
}

impl Pipeline for FaceMeshPipeline {
    fn name(&self) -> String {
        "Face Mesh (468 pts)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        // 1. Detect Face
        // 1. Detect Face
        let roi: Option<Rect> = if let Some(det) = &mut self.detector {
            det.detect(frame)?
        } else {
            None
        };

        /*
           If detection found a face, crop to it.
           If not, we could fall back to full frame (mock behavior) or return None.
           For now, if detector exists but finds nothing, return None (no face).
           If detector is missing, use full frame (bad accuracy but works).
        */

        let (crop, offset_x, offset_y, scale_x, scale_y) = if let Some(rect) = roi {
            // Expand ROI slightly for better mesh context
            let pad_w = rect.width * 0.25;
            let pad_h = rect.height * 0.25;
            let mut x = rect.x - pad_w / 2.0;
            let mut y = rect.y - pad_h / 2.0;
            let mut w = rect.width + pad_w;
            let mut h = rect.height + pad_h;

            // Clip to frame
            if x < 0.0 {
                x = 0.0;
            }
            if y < 0.0 {
                y = 0.0;
            }
            if x + w > frame.width() as f32 {
                w = frame.width() as f32 - x;
            }
            if y + h > frame.height() as f32 {
                h = frame.height() as f32 - y;
            }

            let crop =
                image::imageops::crop_imm(frame, x as u32, y as u32, w as u32, h as u32).to_image();
            // Scaling factor from Crop 192x192 back to Crop
            let sx = w / 192.0;
            let sy = h / 192.0;
            (crop, x, y, sx, sy)
        } else if self.detector.is_some() {
            // Detector active but no face found
            return Ok(None);
        } else {
            // No detector, use full frame
            (
                frame.clone(),
                0.0,
                0.0,
                frame.width() as f32 / 192.0,
                frame.height() as f32 / 192.0,
            )
        };

        // 2. Mesh Inference
        if let Some(model) = &mut self.mesh_session {
            let resized = image::imageops::resize(&crop, 192, 192, FilterType::Triangle);
            let mut input_data = Vec::with_capacity(1 * 192 * 192 * 3);
            for y in 0..192 {
                for x in 0..192 {
                    let pixel = resized.get_pixel(x, y);
                    let r = (pixel[0] as f32 / 127.5) - 1.0;
                    let g = (pixel[1] as f32 / 127.5) - 1.0;
                    let b = (pixel[2] as f32 / 127.5) - 1.0;
                    input_data.push(r);
                    input_data.push(g);
                    input_data.push(b);
                }
            }

            let shape = vec![1, 192, 192, 3];
            let input = ort::value::Tensor::from_array((shape, input_data))?;
            let outputs = model.run(ort::inputs![input])?;

            let (_output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;
            let vec = output_data;

            if vec.len() >= 1404 {
                let mut points = Vec::with_capacity(468);
                for i in 0..468 {
                    let mx = vec[i * 3];
                    let my = vec[i * 3 + 1];
                    let mz = vec[i * 3 + 2];

                    // Transform: Mesh Local (0..192) -> Crop -> Full Frame
                    // CropX = mx * scale_x
                    // FullX = offset_x + CropX

                    points.push(Point3D {
                        x: offset_x + mx * scale_x,
                        y: offset_y + my * scale_y,
                        z: mz, // Depth relative to Z?
                    });
                }
                return Ok(Some(PipelineOutput::Landmarks(Landmarks { points })));
            }
            Ok(None)
        } else {
            // Mock Inference
            let w = frame.width() as f32;
            let h = frame.height() as f32;
            let cx = w / 2.0;
            let cy = h / 2.0;
            let t = self.start_time.elapsed().as_secs_f32();

            let radius = 100.0 + (t * 2.0).sin() * 20.0;
            let mut points = Vec::new();

            for i in 0..468 {
                let angle = (i as f32 / 468.0) * std::f32::consts::PI * 2.0 + t;
                let x = cx + angle.cos() * radius;
                let y = cy + angle.sin() * radius;
                points.push(Point3D { x, y, z: 0.0 });
            }

            Ok(Some(PipelineOutput::Landmarks(Landmarks { points })))
        }
    }
}

pub struct FaceDetectionPipeline {
    detector: FaceDetector,
}

impl FaceDetectionPipeline {
    pub fn new(model_path: &str) -> Result<Self> {
        let detector = FaceDetector::new(model_path)?;
        Ok(Self { detector })
    }
}

impl Pipeline for FaceDetectionPipeline {
    fn name(&self) -> String {
        "Face Detection (UltraFace)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        if let Some(rect) = self.detector.detect(frame)? {
            Ok(Some(PipelineOutput::FaceRects(vec![rect])))
        } else {
            Ok(None)
        }
    }
}
