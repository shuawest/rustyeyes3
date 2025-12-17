use image::{DynamicImage, ImageBuffer, Rgb};
use anyhow::{Result, Context};
use std::fs::{self, File};
use std::path::Path;
use std::io::Write;
use serde::{Deserialize, Serialize};

use crate::types::{CalibrationPoint, CalibrationProfile, PipelineOutput};

pub struct CalibrationManager {
    pub data_dir: String,
    pub profile: Option<CalibrationProfile>,
    data_buffer: Vec<CalibrationPoint>,
}

impl CalibrationManager {
    pub fn new(data_dir: &str) -> Result<Self> {
        if !Path::new(data_dir).exists() {
            fs::create_dir_all(data_dir)?;
        }
        
        // Load existing profile if any
        let profile_path = format!("{}/calibration.json", data_dir);
        let profile = if Path::new(&profile_path).exists() {
            let file = File::open(&profile_path)?;
            let profile: CalibrationProfile = serde_json::from_reader(file).ok().unwrap_or_default();
            println!("Loaded Calibration Profile");
            Some(profile)
        } else {
            None
        };

        Ok(Self {
            data_dir: data_dir.to_string(),
            profile,
            data_buffer: Vec::new(),
        })
    }

    pub fn save_data_point(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>, x: f32, y: f32, inference: Option<PipelineOutput>) -> Result<u64> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as u64;

        let filename = format!("img_{}.jpg", timestamp);
        let file_path = format!("{}/{}", self.data_dir, filename);
        
        let img = DynamicImage::ImageRgb8(frame.clone());
        img.save(&file_path)?;

        let point = CalibrationPoint {
            timestamp,
            screen_x: x,
            screen_y: y,
            inference,
            moondream_result: None,
        };
        
        
        // Save JSON Metadata
        let json_filename = format!("img_{}.json", timestamp);
        let json_path = format!("{}/{}", self.data_dir, json_filename);
        
        let json_file = File::create(&json_path)?;
        serde_json::to_writer_pretty(json_file, &point)?;
        
        self.data_buffer.push(point);
        println!("Saved Calibration Point: ({}, {})", x, y);

        Ok(timestamp)
    }

    pub fn update_point_with_moondream(&mut self, timestamp: u64, result: crate::types::Point3D) -> Result<()> {
        // 1. Find in buffer and update
        if let Some(pt) = self.data_buffer.iter_mut().find(|p| p.timestamp == timestamp) {
            pt.moondream_result = Some(result);
            
            // 2. Overwrite JSON file
            let json_filename = format!("img_{}.json", timestamp);
            let json_path = format!("{}/{}", self.data_dir, json_filename);
            
            let json_file = File::create(&json_path)?;
            serde_json::to_writer_pretty(json_file, &pt)?;
            println!("Updated Calibration Point {} with Moondream Data", timestamp);
        } else {
            // If not in buffer (e.g. restarted), try to load from disk?
            // For now, MVP assumes session continuity.
            // But we can try to blindly load-update-save.
            let json_filename = format!("img_{}.json", timestamp);
            let json_path = format!("{}/{}", self.data_dir, json_filename);
            if Path::new(&json_path).exists() {
                 let file = File::open(&json_path)?;
                 let mut pt: CalibrationPoint = serde_json::from_reader(file)?;
                 pt.moondream_result = Some(result);
                 
                 let json_file = File::create(&json_path)?;
                 serde_json::to_writer_pretty(json_file, &pt)?;
                 println!("Updated Calibration Point {} with Moondream Data (Disk)", timestamp);
            }
        }
        Ok(())
    }
    
    // Very Basic Linear Regression (Affine)
    // Solves X_screen = c0 + c1*X_in + c2*Y_in
    // Using simple Least Squares or just computing from 3 points if small dataset
    // For n > 3 this is OLS.
    // We will assume "raw input" is provided to us as (rx, ry)
    // In real app, we would re-run inference on stored images. 
    // For now, let's assume we implement the 'Offline Compute' later or the user just wants data collection FIRST.
    // The spec says "Compute Calibration" is triggered manually.
    // So we need a function that takes a list of (InputX, InputY, ScreenX, ScreenY) and returns Profile.
    
    pub fn compute_regression(&self, inputs: &[(f32, f32)], targets: &[(f32, f32)]) -> Option<CalibrationProfile> {
        if inputs.len() < 3 {
             println!("Not enough points for calibration (min 3)");
             return None;
        }
        
        // We need to solve A * c = B
        // Where c = [c0, c1, c2]
        // A is [1, x_in, y_in] rows
        // B is [x_screen]
        // We do this twice, once for X_screen, once for Y_screen.
        
        // This requires a matrix library or simple implementation. 
        // For < 10 points, we can write a simple normal equation solver: (A^T A)^-1 A^T B
        
        // Let's implement a very simple solver for 3 coeffs.
        
        let solve = |targets_1d: &Vec<f32>| -> Vec<f32> {
             // Create A matrix flat
             // A [N x 3]
             let n = inputs.len();
             let mut ata = [0.0; 9]; // 3x3
             let mut atb = [0.0; 3]; // 3x1
             
             for i in 0..n {
                 let x_in = inputs[i].0;
                 let y_in = inputs[i].1;
                 let val = targets_1d[i];
                 let row = [1.0, x_in, y_in];
                 
                 // Update ATA (A transpose * A)
                 for r in 0..3 {
                     for c in 0..3 {
                         ata[r*3 + c] += row[r] * row[c];
                     }
                     // Update ATB
                     atb[r] += row[r] * val;
                 }
             }
             
             // Invert ATA (3x3)
             // ... Simple inversion logic ...
             // det
             let det = ata[0] * (ata[4]*ata[8] - ata[5]*ata[7]) -
                       ata[1] * (ata[3]*ata[8] - ata[5]*ata[6]) +
                       ata[2] * (ata[3]*ata[7] - ata[4]*ata[6]);
                       
             if det.abs() < 1e-6 {
                 return vec![0.0, 1.0, 0.0]; // Fallback identity/fail
             }
             
             let inv_det = 1.0 / det;
             let mut inv_ata = [0.0; 9];
             
             inv_ata[0] = (ata[4]*ata[8] - ata[5]*ata[7]) * inv_det;
             inv_ata[1] = (ata[2]*ata[7] - ata[1]*ata[8]) * inv_det;
             inv_ata[2] = (ata[1]*ata[5] - ata[2]*ata[4]) * inv_det;
             inv_ata[3] = (ata[5]*ata[6] - ata[3]*ata[8]) * inv_det;
             inv_ata[4] = (ata[0]*ata[8] - ata[2]*ata[6]) * inv_det;
             inv_ata[5] = (ata[2]*ata[3] - ata[0]*ata[5]) * inv_det;
             inv_ata[6] = (ata[3]*ata[7] - ata[4]*ata[6]) * inv_det;
             inv_ata[7] = (ata[1]*ata[6] - ata[0]*ata[7]) * inv_det;
             inv_ata[8] = (ata[0]*ata[4] - ata[1]*ata[3]) * inv_det;
             
             // Coeffs = inv_ATA * ATB
             let mut coeffs = vec![0.0; 3];
             for r in 0..3 {
                 for c in 0..3 {
                     coeffs[r] += inv_ata[r*3 + c] * atb[c];
                 }
             }
             coeffs
        };
        
        let tx: Vec<f32> = targets.iter().map(|p| p.0).collect();
        let ty: Vec<f32> = targets.iter().map(|p| p.1).collect();
        
        Some(CalibrationProfile {
            x_coeffs: solve(&tx),
            y_coeffs: solve(&ty),
        })
    }
    
    pub fn apply(&self, x_in: f32, y_in: f32) -> (f32, f32) {
        if let Some(prof) = &self.profile {
             if prof.x_coeffs.len() == 3 && prof.y_coeffs.len() == 3 {
                 let x = prof.x_coeffs[0] + prof.x_coeffs[1]*x_in + prof.x_coeffs[2]*y_in;
                 let y = prof.y_coeffs[0] + prof.y_coeffs[1]*x_in + prof.y_coeffs[2]*y_in;
                 return (x, y);
             }
        }
        (x_in, y_in) // Pass through if no calibration
    }
}
