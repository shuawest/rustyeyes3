use image::{imageops::FilterType, ImageBuffer, Rgb};
use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use crate::types::{PipelineOutput, Point3D};
use crate::pipeline::Pipeline;
use crate::inference::FaceMeshPipeline;
use crate::head_pose::HeadPosePipeline;
use crate::rectification::CalibrationParams;
use smartcore::svm::svr::SVR;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::fs;

// =========================================================================
// Smoothing Helper (Exponential Moving Average)
// =========================================================================
// ... (Smoothing struct remains same, skipping for brevity in replacement if possible, but replace tool needs context)
// Wait, I can't skip. Let's just do imports first.

// Let's do imports separately.


// =========================================================================
// Smoothing Helper (Exponential Moving Average)
// =========================================================================
pub struct Smoothing {
    yaw: f32,
    pitch: f32,
    alpha: f32,
    initialized: bool,
}

impl Smoothing {
    pub fn new(alpha: f32) -> Self {
        Self { yaw: 0.0, pitch: 0.0, alpha, initialized: false }
    }
    
    pub fn filter(&mut self, yaw: f32, pitch: f32) -> (f32, f32) {
        if !self.initialized {
            self.yaw = yaw;
            self.pitch = pitch;
            self.initialized = true;
            return (yaw, pitch);
        }
        self.yaw = self.alpha * yaw + (1.0 - self.alpha) * self.yaw;
        self.pitch = self.alpha * pitch + (1.0 - self.alpha) * self.pitch;
        (self.yaw, self.pitch)
    }
}

// =========================================================================
// Pipeline 4: Simulated Gaze (Head Pose + Eye Centers)
// Uses geometric approximation.
// =========================================================================
pub struct SimulatedGazePipeline {
    mesh_pipeline: FaceMeshPipeline,
    pose_pipeline: HeadPosePipeline,
}

impl SimulatedGazePipeline {
    pub fn new(mesh_model: &str, pose_model: &str, detector: &str) -> Result<Self> {
        Ok(Self {
            mesh_pipeline: FaceMeshPipeline::new(mesh_model, detector)?,
            pose_pipeline: HeadPosePipeline::new(pose_model, detector)?,
        })
    }
}

impl Pipeline for SimulatedGazePipeline {
    fn name(&self) -> String {
        "Simulated Gaze (Head + Eye Position)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        // Run Mesh and Pose
        let mesh_out = self.mesh_pipeline.process(frame)?;
        let pose_out = self.pose_pipeline.process(frame)?;

        if let (Some(PipelineOutput::Landmarks(l)), Some(PipelineOutput::HeadPose(y, p, r))) = (mesh_out, pose_out) {
             // Calculate eye centers (Geometric)
             let left_indices = [33, 133, 160, 159, 158, 144, 145, 153];
             let right_indices = [362, 263, 387, 386, 385, 373, 374, 380];
             
             let calc_center = |indices: &[usize]| -> Point3D {
                 let mut x = 0.0; let mut y = 0.0;
                 for &i in indices {
                     if i < l.points.len() { x += l.points[i].x; y += l.points[i].y; }
                 }
                 Point3D { x: x / indices.len() as f32, y: y / indices.len() as f32, z: 0.0 }
             };
             
             let lx = calc_center(&left_indices);
             let rx = calc_center(&right_indices);
             
             let (final_yaw, final_pitch) = compute_simulated_gaze(y, p);
             
             return Ok(Some(PipelineOutput::Gaze {
                 left_eye: lx,
                 right_eye: rx,
                 yaw: final_yaw,
                 pitch: final_pitch,
                 roll: r,
                 vector: Point3D { x: 0.0, y: 0.0, z: 1.0 },
                 landmarks: Some(l.clone()),
             }));
        }
        
        Ok(None)
    }
}

// Pure function for regression testing
pub fn compute_simulated_gaze(head_yaw: f32, head_pitch: f32) -> (f32, f32) {
     // Sensitivity Gain
     let yaw_gain = 1.5;
     let pitch_gain = 2.5;
     
     // CRITICAL REGRESSION CHECK:
     // Head Left is Negative Yaw in our coordinate system.
     // Gaze Left should be Negative.
     // Invert input to output?
     // Current working logic: yaw = -y * gain.
     (-head_yaw * yaw_gain, head_pitch * pitch_gain)
}

// =========================================================================
// Pipeline 5: Pupil Gaze (CV Based)
// Uses Head Pose + Computer Vision Pupil Tracking (Darkest Blob)
// =========================================================================
pub struct PupilGazePipeline {
    mesh_pipeline: FaceMeshPipeline,
    pose_pipeline: HeadPosePipeline,
}

impl PupilGazePipeline {
    pub fn new(mesh_model: &str, pose_model: &str, detector: &str) -> Result<Self> {
        Ok(Self {
            mesh_pipeline: FaceMeshPipeline::new(mesh_model, detector)?,
            pose_pipeline: HeadPosePipeline::new(pose_model, detector)?,
        })
    }
    
    // Helper to find pupil offset in an eye crop
    fn detect_pupil_offset(&self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>, center: &Point3D) -> (f32, f32) {
        // Crop 40x40 around eye center
        let radius = 20; 
        let crop_size = 40;
        let cx = center.x as i32;
        let cy = center.y as i32;
        let x = (cx - radius).max(0);
        let y = (cy - radius).max(0);
        
        // Safety check
        if x as u32 + crop_size as u32 > frame.width() || y as u32 + crop_size as u32 > frame.height() {
            return (0.0, 0.0);
        }

        // We process the crop manually to avoid copies if possible, or just loop
        let mut min_val = 255u8;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut count = 0.0;
        
        // Pass 1: Find min brightness (darkest)
        // Simple luminance: 0.299R + 0.587G + 0.114B
        for dy in 0..crop_size {
            for dx in 0..crop_size {
                let px = frame.get_pixel((x + dx) as u32, (y + dy) as u32);
                let luma = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8;
                if luma < min_val { min_val = luma; }
            }
        }
        
        // Threshold: darkest + range
        // If min is 20, threshold maybe 50?
        let threshold = min_val.saturating_add(30); 
        
        // Pass 2: Centroid of dark pixels
        for dy in 0..crop_size {
            for dx in 0..crop_size {
                 let px = frame.get_pixel((x + dx) as u32, (y + dy) as u32);
                 let luma = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8;
                 
                 if luma <= threshold {
                     // Weight by darkness? (threshold - luma)
                     let w = (threshold - luma) as f32;
                     sum_x += dx as f32 * w;
                     sum_y += dy as f32 * w;
                     count += w;
                 }
            }
        }
        
        if count > 0.0 {
            let px = sum_x / count;
            let py = sum_y / count;
            // Offset from center of crop (20, 20)
            let dx = px - 20.0;
            let dy = py - 20.0;
            // Normalize (-1.0 to 1.0) relative to crop radius (20)
            return (dx / 20.0, dy / 20.0);
        }
        
        (0.0, 0.0)
    }
}

impl Pipeline for PupilGazePipeline {
    fn name(&self) -> String {
        "Pupil Gaze (Computer Vision)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        let mesh_out = self.mesh_pipeline.process(frame)?;
        let pose_out = self.pose_pipeline.process(frame)?;
        
        if let (Some(PipelineOutput::Landmarks(l)), Some(PipelineOutput::HeadPose(y, p, r))) = (mesh_out, pose_out) {
             // 1. Get Eye Centers
             let left_indices = [33, 133]; // Simplified for center
             let right_indices = [362, 263];
             
             let calc_center = |indices: &[usize]| -> Point3D {
                 let mut x = 0.0; let mut y = 0.0;
                 for &i in indices {
                     if i < l.points.len() { x += l.points[i].x; y += l.points[i].y; }
                 }
                 Point3D { x: x / indices.len() as f32, y: y / indices.len() as f32, z: 0.0 }
             };
             
             let lx = calc_center(&left_indices);
             let rx = calc_center(&right_indices);
             
             // 2. Detect Pupil Offsets (CV)
             let (ldx, ldy) = self.detect_pupil_offset(frame, &lx);
             let (rdx, rdy) = self.detect_pupil_offset(frame, &rx);
             
             // Avg Offset
             let ax = (ldx + rdx) / 2.0;
             let ay = (ldy + rdy) / 2.0;
             
             // 3. Map to Gaze Angles
             let (final_yaw, final_pitch) = compute_pupil_gaze(y, p, ax, ay);
             
             return Ok(Some(PipelineOutput::Gaze {
                 left_eye: lx,
                 right_eye: rx,
                 yaw: final_yaw, 
                 pitch: final_pitch,
                 roll: r,
                 vector: Point3D { x: 0.0, y: 0.0, z: 1.0 },
                 landmarks: Some(l.clone()),
             }));
        }
        
        Ok(None)
    }
}

pub fn compute_pupil_gaze(
    head_yaw: f32, 
    head_pitch: f32, 
    pupil_offset_x: f32, 
    pupil_offset_y: f32
) -> (f32, f32) {
     // Head Pose Gain
     let head_yaw_gain = 1.5;
     let head_pitch_gain = 2.5; 
     
     // Eye movement Gain (How much pupil shift affects gaze angle)
     // Pupil shift 1.0 (edge of eye) ~= 30 degrees?
     let pupil_gain_x = 40.0; // degrees
     let pupil_gain_y = 30.0; // degrees
     
     let raw_yaw = (head_yaw * head_yaw_gain) + (pupil_offset_x * pupil_gain_x);
     let raw_pitch = (head_pitch * head_pitch_gain) + (pupil_offset_y * pupil_gain_y);
     
     // Invert to match L2CS behavior (verified "Head left -> dot right" with positive)
     (-raw_yaw, raw_pitch)
}

// =========================================================================
// Pipeline 6: L2CS-Net (ResNet50 Backbone)
// High Accuracy, Higher Latency (~20ms on M1)
// =========================================================================
pub struct L2CSPipeline {
    session: Option<Session>,
    mesh_pipeline: FaceMeshPipeline,
    smoothing: Smoothing,
    pub params: CalibrationParams,
    // SVR Models (Optional)
    svr_x: Option<SVR<'static, f64, DenseMatrix<f64>, Vec<f64>>>,
    svr_y: Option<SVR<'static, f64, DenseMatrix<f64>, Vec<f64>>>,
}

impl L2CSPipeline {
    pub fn new(model_path: &str, mesh_path: &str, detector_path: &str) -> Result<Self> {
        let mesh_pipeline = FaceMeshPipeline::new(mesh_path, detector_path)?;
        
        let session = if Path::new(model_path).exists() {
            println!("[L2CS] Loading model from {}...", model_path);
            Some(Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_execution_providers([
                     ort::execution_providers::CoreMLExecutionProvider::default().build(),
                     ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(model_path)?)
        } else {
            println!("[L2CS] Model not found at {}. Gaze will be disabled.", model_path);
            None
        };
        
        // Load SVR if exists
        let load_svr = |path: &str| -> Option<SVR<'static, f64, DenseMatrix<f64>, Vec<f64>>> {
            if Path::new(path).exists() {
                println!("[L2CS] Loading SVR Calibration: {}", path);
                if let Ok(bytes) = fs::read(path) {
                    return bincode::deserialize(&bytes).ok();
                }
            }
            None
        };
        
        let svr_x = load_svr("svr_l2cs_x.model");
        let svr_y = load_svr("svr_l2cs_y.model");

        Ok(Self {
            session,
            mesh_pipeline,
            smoothing: Smoothing::new(0.4),
            params: CalibrationParams::default(),
            svr_x,
            svr_y
        })
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

    pub fn process_raw_values(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        self.smoothing = Smoothing::new(0.4);
        self.process(frame)
    }
}

impl Pipeline for L2CSPipeline {
    fn name(&self) -> String {
        "L2CS-Net Gaze (ResNet50)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        // 1. Get Face Mesh (reusing existing pipeline logic for detection/ROI)
        // We run the mesh pipeline first to populate landmarks and find the face
        let mesh_out = self.mesh_pipeline.process(frame)?;
        
        // Check if we have a model loaded
        let model = match &mut self.session {
            Some(s) => s,
            None => return Ok(None),
        };

        if let Some(PipelineOutput::Landmarks(l)) = &mesh_out {
            // 2. Crop Face
            // Calculate bounding box from landmarks
            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;
            
            for p in &l.points {
                if p.x < min_x { min_x = p.x; }
                if p.y < min_y { min_y = p.y; }
                if p.x > max_x { max_x = p.x; }
                if p.y > max_y { max_y = p.y; }
            }
            
            // Expand crop (L2CS likes some context, maybe 1.5x face size?)
            let w = max_x - min_x;
            let h = max_y - min_y;
            let cx = min_x + w / 2.0;
            let cy = min_y + h / 2.0;
            
            let size = w.max(h) * 1.5; 
            
            let x = cx - size / 2.0;
            let y = cy - size / 2.0;
            
            // Safe Crop
            let mut sx = x;
            let mut sy = y;
            let mut sw = size;
            let mut sh = size;
            
            if sx < 0.0 { sx = 0.0; }
            if sy < 0.0 { sy = 0.0; }
            if sx + sw > frame.width() as f32 { sw = frame.width() as f32 - sx; }
            if sy + sh > frame.height() as f32 { sh = frame.height() as f32 - sy; }
            
            let crop = image::imageops::crop_imm(frame, sx as u32, sy as u32, sw as u32, sh as u32).to_image();
            
            // 3. Resize & Normalize
            // L2CS original is 448x448. The ONNX model apparently expects 448.
            let resized = image::imageops::resize(&crop, 448, 448, FilterType::Triangle);
            
             let mut input_data = Vec::with_capacity(1 * 3 * 448 * 448);
             let mean = [0.485, 0.456, 0.406];
             let std = [0.229, 0.224, 0.225];
             
             // RGB Order (Standard for PyTorch/ONNX)
             // R
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[0] as f32 / 255.0;
                 input_data.push((p - mean[0]) / std[0]);
             }}
             // G
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[1] as f32 / 255.0;
                 input_data.push((p - mean[1]) / std[1]);
             }}
             // B
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[2] as f32 / 255.0;
                 input_data.push((p - mean[2]) / std[2]);
             }}
             
             let input = ort::value::Tensor::from_array((vec![1, 3, 448, 448], input_data))?;
             let outputs = model.run(ort::inputs![input])?;
             
             // 4. Extract Output
             let output_count = outputs.len();
             
             let (yaw_deg, pitch_deg) = if output_count == 1 {
                 // Regression Model (Custom)
                 // Expected shape: [1, 2] -> [yaw, pitch]
                 let (_, out) = outputs[0].try_extract_tensor::<f32>()?;
                 if out.len() == 2 {
                     (out[0], out[1])
                 } else {
                     eprintln!("L2CS Single Output mode expects 2 values (Yaw, Pitch), got {}", out.len());
                     return Ok(None);
                 }
             } else {
                 // Classification Model (Original L2CS)
                 // Expected: Output 0 and 1 are separate (Yaw/Pitch)
                 let (_, out0) = outputs[0].try_extract_tensor::<f32>()?;
                 let (_, out1) = outputs[1].try_extract_tensor::<f32>()?;
                 
                 let prob0 = Self::softmax(out0);
                 let prob1 = Self::softmax(out1);
                 
                 let bins = out0.len(); 
                 let range = 360.0; // -180 to 180
                 let step = range / bins as f32; // step = 4.0
                 let start = -180.0;
                 
                 let val0 = Self::expectation(&prob0, start, step);
                 let val1 = Self::expectation(&prob1, start, step);

                 // Mapping 0->Pitch, 1->Yaw? Based on earlier analysis, out0=Yaw, out1=Pitch was suspected
                 // But logic below assigns val0->yaw, val1->pitch.
                 (val0, val1)
             };

                          // Polynomial Calibration
              // val = a*x^2 + b*x + c
              // x = raw degrees
              // params: curve=a, gain=b, offset=c
              
              let y_raw = yaw_deg;
              let p_raw = pitch_deg;
              
              // SVR Inference (if models loaded)
              let (y_gained, p_gained) = if let (Some(svr_x), Some(svr_y)) = (&self.svr_x, &self.svr_y) {
                    let feat = DenseMatrix::from_2d_vec(&vec![vec![y_raw as f64, p_raw as f64]]);
                    let px = svr_x.predict(&feat).unwrap_or(vec![0.0])[0] as f32;
                    let py = svr_y.predict(&feat).unwrap_or(vec![0.0])[0] as f32;
                    (px, py)
              } else {
                  // Fallback: Polynomial Calibration
                  // Note: Offset is now the constant term 'c', not a subtraction pre-gain.
                  let yg = self.params.yaw_curve * y_raw * y_raw + self.params.yaw_gain * y_raw + self.params.yaw_offset;
                  // Invert pitch result to satisfy main.rs subtraction logic
                  let pg = -(self.params.pitch_curve * p_raw * p_raw + self.params.pitch_gain * p_raw + self.params.pitch_offset);
                  (yg, pg)
              };
             
             // Apply Smoothing
             let (y_smooth, p_smooth) = self.smoothing.filter(y_gained, p_gained);
             
             // Debug
             // println!("L2CS: idx=({:.1}, {:.1}) deg=({:.1}, {:.1}) out=({:.2}, {:.2})", 
             //    pitch_idx, yaw_idx, pitch_deg, yaw_deg, p_smooth, y_smooth);
             
             // Eye Center Calculation (for UI lines)
             // Reuse geometric centers from mesh
              let left_indices = [33, 133]; 
              let right_indices = [362, 263];
              let calc_center = |indices: &[usize]| -> Point3D {
                 let mut x = 0.0; let mut y = 0.0;
                 for &i in indices {
                     if i < l.points.len() { x += l.points[i].x; y += l.points[i].y; }
                 }
                 Point3D { x: x / indices.len() as f32, y: y / indices.len() as f32, z: 0.0 }
             };
             let lx = calc_center(&left_indices);
             let rx = calc_center(&right_indices);
             
            return Ok(Some(PipelineOutput::Gaze {
                 left_eye: lx,
                 right_eye: rx,
                 yaw: y_smooth,
                 pitch: p_smooth,
                 roll: 0.0, // L2CS doesn't do roll
                 vector: Point3D { x: 0.0, y: 0.0, z: 1.0 },
                 landmarks: Some(l.clone()),
             }));
        }
        
        Ok(None)
    }

}

// =========================================================================
// Pipeline 7: MobileGaze (MobileNetV2 Backbone)
// High Speed, Lower Accuracy (~4ms on M1)
// =========================================================================
// #[derive(Clone)] (Cannot derive Clone easily for Session)
pub struct MobileGazePipeline {
    session: Option<Session>,
    mesh_pipeline: FaceMeshPipeline,
    smoothing: Smoothing,
    pub params: CalibrationParams,
}

impl MobileGazePipeline {
    pub fn new(model_path: &str, mesh_path: &str, detector_path: &str) -> Result<Self> {
        let mesh_pipeline = FaceMeshPipeline::new(mesh_path, detector_path)?;
        
        let session = if Path::new(model_path).exists() {
            println!("[MobileGaze] Loading model from {}...", model_path);
            Some(Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_execution_providers([
                     ort::execution_providers::CoreMLExecutionProvider::default().build(),
                     ort::execution_providers::CPUExecutionProvider::default().build(),
                ])?
                .commit_from_file(model_path)?)
        } else {
            println!("[MobileGaze] Model not found at {}. Gaze will be disabled.", model_path);
            None
        };

        Ok(Self {
            session,
            mesh_pipeline,
            smoothing: Smoothing::new(0.4),
            params: CalibrationParams::default(),
        })
    }
    
    // Duplicated helpers for now (could refactor to shared utils later)
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

    pub fn process_raw_values(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
        self.smoothing = Smoothing::new(0.4);
        self.process(frame)
    }
}

impl Pipeline for MobileGazePipeline {
    fn name(&self) -> String {
        "MobileGaze (Lightweight)".to_string()
    }

    fn process(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Option<PipelineOutput>> {
         // 1. Get Face Mesh
        let mesh_out = self.mesh_pipeline.process(frame)?;
        
        let model = match &mut self.session {
            Some(s) => s,
            None => return Ok(None),
        };

        if let Some(PipelineOutput::Landmarks(l)) = &mesh_out {
            // 2. Crop Face (Same logic as L2CS)
            let mut min_x = f32::MAX;
            let mut min_y = f32::MAX;
            let mut max_x = f32::MIN;
            let mut max_y = f32::MIN;
            
            for p in &l.points {
                if p.x < min_x { min_x = p.x; }
                if p.y < min_y { min_y = p.y; }
                if p.x > max_x { max_x = p.x; }
                if p.y > max_y { max_y = p.y; }
            }
            
            let w = max_x - min_x;
            let h = max_y - min_y;
            let cx = min_x + w / 2.0;
            let cy = min_y + h / 2.0;
            let size = w.max(h) * 1.5; 
            let x = cx - size / 2.0;
            let y = cy - size / 2.0;
            
            let mut sx = x;
            let mut sy = y;
            let mut sw = size;
            let mut sh = size;
            
            if sx < 0.0 { sx = 0.0; }
            if sy < 0.0 { sy = 0.0; }
            if sx + sw > frame.width() as f32 { sw = frame.width() as f32 - sx; }
            if sy + sh > frame.height() as f32 { sh = frame.height() as f32 - sy; }
            
            let crop = image::imageops::crop_imm(frame, sx as u32, sy as u32, sw as u32, sh as u32).to_image();
            
            // 3. Resize & Normalize (MobileGaze via L2CS repo also uses 448)
            let _ = crop.save("debug_face_crop.png"); // Save the cropped image
            let resized = image::imageops::resize(&crop, 448, 448, FilterType::Triangle);
            
             let mut input_data = Vec::with_capacity(1 * 3 * 448 * 448);
             let mean = [0.485, 0.456, 0.406];
             let std = [0.229, 0.224, 0.225];
             
             // RGB Order
             // R
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[0] as f32 / 255.0;
                 input_data.push((p - mean[0]) / std[0]);
             }}
             // G
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[1] as f32 / 255.0;
                 input_data.push((p - mean[1]) / std[1]);
             }}
             // B
             for y in 0..448 { for x in 0..448 { 
                 let p = resized.get_pixel(x, y)[2] as f32 / 255.0;
                 input_data.push((p - mean[2]) / std[2]);
             }}
             
             let input = ort::value::Tensor::from_array((vec![1, 3, 448, 448], input_data))?;
             let outputs = model.run(ort::inputs![input])?;
             
             // 4. Extract Output (Pitch, Yaw)
             let (_, out0) = outputs[0].try_extract_tensor::<f32>()?;
             let (_, out1) = outputs[1].try_extract_tensor::<f32>()?;
             
             let prob0 = Self::softmax(out0);
             let prob1 = Self::softmax(out1);
             
             // MobileGaze shared format: 
             // range = 360, start = -180
             let bins = out0.len(); 
             let range = 360.0;
             let step = range / bins as f32;
             let start = -180.0;
             
             let val0 = Self::expectation(&prob0, start, step);
             let val1 = Self::expectation(&prob1, start, step);

             // Calculate indices for debug (0-90)
             let pitch_idx = Self::expectation(&prob0, 0.0, 1.0);
             let yaw_idx = Self::expectation(&prob1, 0.0, 1.0);
             
              // Swapped based on correlation analysis
              // CRITICAL: We invert output here to match L2CS standard (Negative = Left).
              // Original MobileGaze output was inverted relative to L2CS.
              let yaw_deg = -val0; 
              let pitch_deg = val1;
                          // Polynomial Calibration
              let y_raw = yaw_deg;
              let p_raw = pitch_deg;
              
              let y_gained = self.params.yaw_curve * y_raw * y_raw + self.params.yaw_gain * y_raw + self.params.yaw_offset;
              let p_gained = -(self.params.pitch_curve * p_raw * p_raw + self.params.pitch_gain * p_raw + self.params.pitch_offset);
             
             // Smoothing
             let (y_smooth, p_smooth) = self.smoothing.filter(y_gained, p_gained);
             
             // Debug
             // println!("Mobile: idx=({:.1}, {:.1}) deg=({:.1}, {:.1}) out=({:.2}, {:.2})", 
             //    pitch_idx, yaw_idx, pitch_deg, yaw_deg, p_smooth, y_smooth);
             
             // Eye Centers
              let left_indices = [33, 133]; 
              let right_indices = [362, 263];
              let calc_center = |indices: &[usize]| -> Point3D {
                 let mut x = 0.0; let mut y = 0.0;
                 for &i in indices {
                     if i < l.points.len() { x += l.points[i].x; y += l.points[i].y; }
                 }
                 Point3D { x: x / indices.len() as f32, y: y / indices.len() as f32, z: 0.0 }
             };
             let lx = calc_center(&left_indices);
             let rx = calc_center(&right_indices);
             
            return Ok(Some(PipelineOutput::Gaze {
                 left_eye: lx,
                 right_eye: rx,
                 yaw: y_smooth,
                 pitch: p_smooth,
                 roll: 0.0,
                 vector: Point3D { x: 0.0, y: 0.0, z: 1.0 },
                 landmarks: Some(l.clone()),
             }));
        }
        
        Ok(None)
    }


}

