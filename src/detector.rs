use crate::types::Rect;
use anyhow::Result;
use image::{imageops::FilterType, ImageBuffer, Rgb};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub struct FaceDetector {
    session: Session,
    anchors: Vec<(f32, f32, f32, f32)>, // cx, cy, w, h
}

impl FaceDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_execution_providers([
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ort::execution_providers::CPUExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        let anchors = generate_anchors(320, 240);
        Ok(Self { session, anchors })
    }

    pub fn detect(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<Rect>> {
        // 1. Preprocess: Resize to 320x240
        let resized = image::imageops::resize(frame, 320, 240, FilterType::Triangle);

        // HWC -> NCHW, Normalize (pixel - 127) / 128
        let mut input_data = Vec::with_capacity(1 * 3 * 240 * 320);

        // Iterate channel first for NCHW? No, standard is usually:
        // Input layout: often NCHW for ONNX.
        // Let's create NCHW [1, 3, 240, 320]

        let width = 320;
        let height = 240;

        // R channel
        for y in 0..height {
            for x in 0..width {
                let p = resized.get_pixel(x, y)[0];
                input_data.push((p as f32 - 127.0) / 128.0);
            }
        }
        // G channel
        for y in 0..height {
            for x in 0..width {
                let p = resized.get_pixel(x, y)[1];
                input_data.push((p as f32 - 127.0) / 128.0);
            }
        }
        // B channel
        for y in 0..height {
            for x in 0..width {
                let p = resized.get_pixel(x, y)[2];
                input_data.push((p as f32 - 127.0) / 128.0);
            }
        }

        let input_tensor = Tensor::from_array((vec![1, 3, 240, 320], input_data))?;
        let outputs = self.session.run(ort::inputs![input_tensor])?;

        let (_scores_shape, scores_data) = outputs["scores"].try_extract_tensor::<f32>()?;
        let (_boxes_shape, boxes_data) = outputs["boxes"].try_extract_tensor::<f32>()?;

        let score_threshold = 0.7;
        let best_box =
            FaceDetector::post_process(&self.anchors, scores_data, boxes_data, score_threshold);

        if let Some(rect) = best_box {
            // Scale back to original frame
            let sx = frame.width() as f32 / 320.0;
            let sy = frame.height() as f32 / 240.0;

            Ok(Some(Rect::new(
                rect.x * sx,
                rect.y * sy,
                rect.width * sx,
                rect.height * sy,
            )))
        } else {
            Ok(None)
        }
    }

    fn post_process(
        anchors: &[(f32, f32, f32, f32)],
        scores_raw: &[f32],
        boxes_raw: &[f32],
        threshold: f32,
    ) -> Option<Rect> {
        // Iterate 4420 anchors
        let num_anchors = anchors.len();

        let mut best_score = 0.0;
        let mut best_rect = None;

        // Variance for UltraFace
        let center_variance = 0.1;
        let size_variance = 0.2;

        for i in 0..num_anchors {
            let score = scores_raw[i * 2 + 1];
            if score > threshold && score > best_score {
                // Decode box
                let cx_enc = boxes_raw[i * 4];
                let cy_enc = boxes_raw[i * 4 + 1];
                let w_enc = boxes_raw[i * 4 + 2];
                let h_enc = boxes_raw[i * 4 + 3];

                let (ax, ay, aw, ah) = anchors[i];

                let cx = cx_enc * center_variance * aw + ax;
                let cy = cy_enc * center_variance * ah + ay;
                let w = (w_enc * size_variance).exp() * aw;
                let h = (h_enc * size_variance).exp() * ah;

                let x = cx - w / 2.0;
                let y = cy - h / 2.0;

                best_score = score;
                best_rect = Some(Rect::new(x * 320.0, y * 240.0, w * 320.0, h * 240.0));
            }
        }

        best_rect
    }
}

fn generate_anchors(width: usize, height: usize) -> Vec<(f32, f32, f32, f32)> {
    // UltraFace configs
    let shrinkage_list = [8, 16, 32, 64];
    // Use Vec<Vec> to avoid mismatched array size error
    let min_boxes = vec![
        vec![10.0, 16.0, 24.0],
        vec![32.0, 48.0],
        vec![64.0, 96.0],
        vec![128.0, 192.0, 256.0],
    ];
    let mut anchors = Vec::new();

    let w = width as f32;
    let h = height as f32;

    for (i, &shrinkage) in shrinkage_list.iter().enumerate() {
        let feature_h = (height as f32 / shrinkage as f32).ceil() as usize;
        let feature_w = (width as f32 / shrinkage as f32).ceil() as usize;

        for v in 0..feature_h {
            for u in 0..feature_w {
                let cx = (u as f32 * shrinkage as f32 + shrinkage as f32 / 2.0) / w;
                let cy = (v as f32 * shrinkage as f32 + shrinkage as f32 / 2.0) / h;

                for &min_box in &min_boxes[i] {
                    anchors.push((cx, cy, min_box / w, min_box / h));
                }
            }
        }
    }
    anchors
}
