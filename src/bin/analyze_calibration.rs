use std::fs;
use std::path::Path;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct Report {
    entries: Vec<Entry>,
}

#[derive(Deserialize, Debug)]
struct Entry {
    target_x: f32,
    target_y: f32,
    raw_yaw: f32,
    raw_pitch: f32,
}

fn main() -> anyhow::Result<()> {
    let reports_dir = Path::new("calibration_data/reports");
    if !reports_dir.exists() {
        println!("No reports found.");
        return Ok(());
    }

    for entry in fs::read_dir(reports_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "json") && path.file_name().unwrap().to_string_lossy().contains("report_l2cs") {
            println!("Analyzing {:?}", path);
            let content = fs::read_to_string(&path)?;
            let report: Report = serde_json::from_str(&content)?;
            analyze(&report.entries);
        }
    }
    Ok(())
}

fn analyze(entries: &[Entry]) {
    let n = entries.len() as f32;
    
    // Correlation X vs Yaw
    let mean_x = entries.iter().map(|e| e.target_x).sum::<f32>() / n;
    let mean_yaw = entries.iter().map(|e| e.raw_yaw).sum::<f32>() / n;
    
    let mut num_x = 0.0;
    let mut den_x1 = 0.0;
    let mut den_x2 = 0.0;
    
    for e in entries {
        let dx = e.target_x - mean_x;
        let dy = e.raw_yaw - mean_yaw;
        num_x += dx * dy;
        den_x1 += dx * dx;
        den_x2 += dy * dy;
    }
    let r_x = num_x / (den_x1.sqrt() * den_x2.sqrt());
    
    // Correlation Y vs Pitch
    let mean_y = entries.iter().map(|e| e.target_y).sum::<f32>() / n;
    let mean_pitch = entries.iter().map(|e| e.raw_pitch).sum::<f32>() / n;
    
    let mut num_y = 0.0;
    let mut den_y1 = 0.0;
    let mut den_y2 = 0.0;
    
    for e in entries {
        let dy_target = e.target_y - mean_y;
        let dy_pitch = e.raw_pitch - mean_pitch;
        num_y += dy_target * dy_pitch;
        den_y1 += dy_target * dy_target;
        den_y2 += dy_pitch * dy_pitch;
    }
    let r_y = num_y / (den_y1.sqrt() * den_y2.sqrt());
    
    println!("Correlation TargetX vs RawYaw: {:.4}", r_x);
    println!("Correlation TargetY vs RawPitch: {:.4}", r_y);
    
    if r_x.abs() < 0.5 { println!("Warning: Weak X correlation. Data may be noisy or axes unrelated."); }
    if r_y.abs() < 0.5 { println!("Warning: Weak Y correlation."); }
    
    // Simple 1D Linear Regression to find slope (Gain)
    // x = a + b * yaw -> b = r * (std_x / std_yaw)
    let std_x = (den_x1 / n).sqrt();
    let std_yaw = (den_x2 / n).sqrt();
    let slope_x = r_x * (std_x / std_yaw);
    
    let std_y = (den_y1 / n).sqrt();
    let std_pitch = (den_y2 / n).sqrt();
    let slope_y = r_y * (std_y / std_pitch);
    
    println!("Estimated Gain X (px/deg): {:.2}", slope_x);
    println!("Estimated Gain Y (px/deg): {:.2}", slope_y);
    
    println!("--- Range Analysis ---");
    let min_yaw = entries.iter().map(|e| e.raw_yaw).fold(f32::INFINITY, f32::min);
    let max_yaw = entries.iter().map(|e| e.raw_yaw).fold(f32::NEG_INFINITY, f32::max);
    println!("Yaw Range: [{:.2}, {:.2}] (Delta: {:.2})", min_yaw, max_yaw, max_yaw - min_yaw);

    let min_pitch = entries.iter().map(|e| e.raw_pitch).fold(f32::INFINITY, f32::min);
    let max_pitch = entries.iter().map(|e| e.raw_pitch).fold(f32::NEG_INFINITY, f32::max);
    println!("Pitch Range: [{:.2}, {:.2}] (Delta: {:.2})", min_pitch, max_pitch, max_pitch - min_pitch);
}
