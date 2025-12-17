use image::{ImageBuffer, Rgb};
use anyhow::Result;
use crate::types::{PipelineOutput, Point3D};
use crate::pipeline::Pipeline;
use crate::inference::FaceMeshPipeline;
use crate::head_pose::HeadPosePipeline;

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
             
             // Sensitivity Gain
             let yaw_gain = 1.5;
             let pitch_gain = 2.5;
             
             return Ok(Some(PipelineOutput::Gaze {
                 left_eye: lx,
                 right_eye: rx,
                 yaw: y * yaw_gain,
                 pitch: p * pitch_gain,
                 roll: r,
                 vector: Point3D { x: 0.0, y: 0.0, z: 1.0 },
                 landmarks: Some(l.clone()),
             }));
        }
        
        Ok(None)
    }
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
             // Head Pose Gain
             let head_yaw_gain = 1.5;
             let head_pitch_gain = 2.5; 
             
             // Eye movement Gain (How much pupil shift affects gaze angle)
             // Pupil shift 1.0 (edge of eye) ~= 30 degrees?
             let pupil_gain_x = 40.0; // degrees
             let pupil_gain_y = 30.0; // degrees
             
             let final_yaw = (y * head_yaw_gain) + (ax * pupil_gain_x);
             let final_pitch = (p * head_pitch_gain) + (ay * pupil_gain_y);
             
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
