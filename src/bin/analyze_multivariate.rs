use rusty_eyes::config::AppConfig;
use rusty_eyes::gaze::{L2CSPipeline, MobileGazePipeline};
use rusty_eyes::pipeline::Pipeline;
use rusty_eyes::rectification::CalibrationParams;
use image::ImageReader;
use std::path::Path;
use std::fs;
use std::io::Write;

#[derive(serde::Deserialize)]
struct JsonMeta { screen_x: f32, screen_y: f32 }

fn main() -> anyhow::Result<()> {
    let config = AppConfig::load()?;
    let data_dir = Path::new("calibration_data");
    
    println!("=== MULTIVARIATE ANALYSIS ===");
    
    // Using L2CS for stability
    let mut l2cs = L2CSPipeline::new(
        &config.models.l2cs_path,
        &config.models.face_mesh_path,
        &config.models.face_detection_path,
    )?;
    // Identity params
    l2cs.params = CalibrationParams {
        yaw_offset: 0.0, pitch_offset: 0.0, yaw_gain: 1.0, pitch_gain: 1.0,
    };
    
    let mut recs = Vec::new();
    
    let entries = fs::read_dir(data_dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "jpg") {
            let json_path = path.with_extension("json");
            if json_path.exists() {
                 let json_content = fs::read_to_string(&json_path)?;
                 if let Ok(meta) = serde_json::from_str::<JsonMeta>(&json_content) {
                     let img = ImageReader::open(&path)?.decode()?.to_rgb8();
                     
                     // We need face bounding box center. 
                     // L2CS `process_raw_values` runs detection but doesn't expose the bbox center directly in `PipelineOutput`.
                     // BUT, `PipelineOutput::Gaze` has `landmarks`. We can use the mean of landmarks as face center proxy.
                     
                     if let Ok(Some(output)) = l2cs.process_raw_values(&img) {
                         if let rusty_eyes::types::PipelineOutput::Gaze { yaw, pitch, landmarks, .. } = output {
                             let (face_x, face_y) = if let Some(lm) = landmarks {
                                 // Simple centroid of first few landmarks? Or all?
                                 let n = lm.points.len() as f32;
                                 let sum_x: f32 = lm.points.iter().map(|p| p.x).sum();
                                 let sum_y: f32 = lm.points.iter().map(|p| p.y).sum();
                                 (sum_x / n, sum_y / n)
                             } else {
                                 (0.0, 0.0)
                             };
                             
                             recs.push((yaw, pitch, face_x, face_y, meta.screen_x, meta.screen_y));
                             print!(".");
                             std::io::stdout().flush().ok();
                         }
                     }
                 }
            }
        }
    }
    println!("\nProcessed {} items.", recs.len());
    if recs.is_empty() { return Ok(()); }

    // Correlations
    let correlation = |idx_a: usize, idx_b: usize, data: &[(f32,f32,f32,f32,f32,f32)]| -> f32 {
        let n = data.len() as f32;
        let get = |row: &(f32,f32,f32,f32,f32,f32), i| match i {
            0 => row.0, 1 => row.1, 2 => row.2, 3 => row.3, 4 => row.4, 5 => row.5, _ => 0.0
        };
        
        let mean_x : f32 = data.iter().map(|r| get(r, idx_a)).sum::<f32>() / n;
        let mean_y : f32 = data.iter().map(|r| get(r, idx_b)).sum::<f32>() / n;
        
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;
        for r in data {
            let dx = get(r, idx_a) - mean_x;
            let dy = get(r, idx_b) - mean_y;
            num += dx*dy;
            den_x += dx*dx;
            den_y += dy*dy;
        }
        if den_x == 0.0 || den_y == 0.0 { 0.0 } else { num / (den_x.sqrt() * den_y.sqrt()) }
    };
    
    // Indices: 0:Yaw, 1:Pitch, 2:FaceX, 3:FaceY, 4:TgtX, 5:TgtY
    println!("\n--- Correlation Matrix ---");
    println!("Target X vs Yaw: {:.4}", correlation(4, 0, &recs));
    println!("Target X vs FaceX: {:.4}", correlation(4, 2, &recs));
    println!("Target Y vs Pitch: {:.4}", correlation(5, 1, &recs));
    println!("Target Y vs FaceY: {:.4}", correlation(5, 3, &recs));
    
    Ok(())
}
