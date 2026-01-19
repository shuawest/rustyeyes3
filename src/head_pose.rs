use crate::detector::FaceDetector;
use crate::pipeline::Pipeline;
use crate::types::PipelineOutput;
use anyhow::Result;
use image::{imageops::FilterType, ImageBuffer, Rgb};
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;

pub struct HeadPosePipeline {
    session: Option<Session>,
    detector: Option<FaceDetector>,
    _detector_path: String,
}

impl HeadPosePipeline {
    pub fn new(model_path: &str, detector_path: &str) -> Result<Self> {
        let detector = if Path::new(detector_path).exists() {
            Some(FaceDetector::new(detector_path)?)
        } else {
            None
        };

        if Path::new(model_path).exists() {
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                    ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(model_path)?;

            Ok(Self {
                session: Some(session),
                detector,
                _detector_path: detector_path.to_string(),
            })
        } else {
            Ok(Self {
                session: None,
                detector,
                _detector_path: detector_path.to_string(),
            })
        }
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum_exps).collect()
    }

    fn expectation(probs: &[f32], range_min: f32, step: f32) -> f32 {
        let mut sum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            sum += prob * (range_min + i as f32 * step);
        }
        sum
    }
}

impl Pipeline for HeadPosePipeline {
    fn name(&self) -> String {
        "Head Pose (WHENet)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        // 1. Detect
        // 1. Detect
        let roi: Option<crate::types::Rect> = if let Some(det) = &mut self.detector {
            det.detect(frame)?
        } else {
            None
        };

        let (crop, _, _, _, _) = if let Some(rect) = roi {
            let pad = rect.width.max(rect.height) * 0.20; // 20% padding? WHENet likes loose crops?
                                                          // Actually WHENet is trained on 224x224.
                                                          // Let's pad square.
            let size = rect.width.max(rect.height) + pad;
            let cx = rect.x + rect.width / 2.0;
            let cy = rect.y + rect.height / 2.0;

            let x = cx - size / 2.0;
            let y = cy - size / 2.0;
            // Clip to frame
            let mut x_clean = x;
            let mut y_clean = y;
            let mut w_clean = size;
            let mut h_clean = size;

            if x_clean < 0.0 {
                x_clean = 0.0;
            }
            if y_clean < 0.0 {
                y_clean = 0.0;
            }
            if x_clean + w_clean > frame.width() as f32 {
                w_clean = frame.width() as f32 - x_clean;
            }
            if y_clean + h_clean > frame.height() as f32 {
                h_clean = frame.height() as f32 - y_clean;
            }

            let crop = image::imageops::crop_imm(
                frame,
                x_clean as u32,
                y_clean as u32,
                w_clean as u32,
                h_clean as u32,
            )
            .to_image();
            (crop, 0.0, 0.0, 0.0, 0.0) // We don't need transform for head pose output (just angles)
        } else if self.detector.is_some() {
            return Ok(None);
        } else {
            (frame.clone(), 0.0, 0.0, 0.0, 0.0)
        };

        if let Some(model) = &mut self.session {
            // 224x224 input
            let resized = image::imageops::resize(&crop, 224, 224, FilterType::Triangle);
            let mut input_data = Vec::with_capacity(1 * 3 * 224 * 224);

            // WHENet Norm: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] ? Or simple?
            // Official repo uses: x = (x / 255.0 - mean) / std (ImageNet)
            let mean = [0.485, 0.456, 0.406];
            let std = [0.229, 0.224, 0.225];

            // R
            for y in 0..224 {
                for x in 0..224 {
                    let p = resized.get_pixel(x, y)[0] as f32 / 255.0;
                    input_data.push((p - mean[0]) / std[0]);
                }
            }
            // G
            for y in 0..224 {
                for x in 0..224 {
                    let p = resized.get_pixel(x, y)[1] as f32 / 255.0;
                    input_data.push((p - mean[1]) / std[1]);
                }
            }
            // B
            for y in 0..224 {
                for x in 0..224 {
                    let p = resized.get_pixel(x, y)[2] as f32 / 255.0;
                    input_data.push((p - mean[2]) / std[2]);
                }
            }

            let input = ort::value::Tensor::from_array((vec![1, 3, 224, 224], input_data))?;
            let outputs = model.run(ort::inputs![input])?;

            // Extract logits
            // Guessing: 0=Yaw, 1=Roll, 2=Pitch? User reported Pitch insensitivity.
            // If 1 was interpreted as Pitch but was Roll (near 0), swapping might fix.
            let (_, yaw_logits) = outputs[0].try_extract_tensor::<f32>()?;
            let (_, roll_logits) = outputs[1].try_extract_tensor::<f32>()?;
            let (_, pitch_logits) = outputs[2].try_extract_tensor::<f32>()?;

            let yaw_prob = Self::softmax(yaw_logits);
            let pitch_prob = Self::softmax(pitch_logits);
            let roll_prob = Self::softmax(roll_logits);

            // Yaw: 120 bins, -180 to 180, step 3
            let yaw = Self::expectation(&yaw_prob, -180.0, 3.0);
            // Pitch: 66 bins, -99 to 99, step 3
            let pitch = Self::expectation(&pitch_prob, -99.0, 3.0);
            // Roll: 66 bins, -99 to 99, step 3
            let roll = Self::expectation(&roll_prob, -99.0, 3.0);

            Ok(Some(PipelineOutput::HeadPose(yaw, pitch, roll)))
        } else {
            Ok(None)
        }
    }
}
