use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize)]
struct DataPoint {
    raw_yaw: f32,
    #[allow(dead_code)]
    raw_pitch: f32,
    target_x: f32,
    #[allow(dead_code)]
    target_y: f32,
}

#[derive(Debug, Deserialize)]
struct Report {
    entries: Vec<DataPoint>,
    model: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: analyze_calibration <report_json>");
        return Ok(());
    }

    let file = File::open(&args[1])?;
    let reader = BufReader::new(file);
    let report: Report = serde_json::from_reader(reader)?;

    println!("Analyzing Report for model: {}", report.model);
    println!("Entries: {}", report.entries.len());

    // 1. Determine Screen Geometry (approx)
    let min_x = report
        .entries
        .iter()
        .map(|e| e.target_x)
        .fold(f32::INFINITY, f32::min);
    let max_x = report
        .entries
        .iter()
        .map(|e| e.target_x)
        .fold(f32::NEG_INFINITY, f32::max);

    // Assume padding, let's say screen is 0..1728 (Macbook Air?) or 1920?
    // Let's deduce width from max_x.
    let width = if max_x > 1800.0 { 1920.0 } else { 1728.0 }; // Heuristic
    let center_x = width / 2.0;

    // 2. Map Pixels -> Degrees
    // Assumption: Screen spans approx 53 degrees width (Macbook at normal distance)
    // +/- 26.5 deg from center.
    let fov_width_deg = 53.0;
    let deg_per_pixel = fov_width_deg / width;

    println!(
        "Geometry: Width={:.0}, Center={:.0}, deg/px={:.4}",
        width, center_x, deg_per_pixel
    );

    let mut x_data = Vec::new(); // raw_yaw
    let mut y_data = Vec::new(); // target_angle

    for e in &report.entries {
        // Convert Target X to Target Angle
        // (x - center) * scale
        // NOTE: Screen X grows Right. Head Yaw grows Left (positive).
        // Standard convention:
        // Screen Right (+X) corresponds to Head Right (-Yaw, usually).
        // But main.rs: "Mirror Mode: -yaw".
        // Let's stick to: Target Angle should follow Head Angle convention.
        // If Head Left = +Yaw.
        // Then Target Left (x < center) should be +Angle.

        let delta_x = e.target_x - center_x;
        // if x < center (Left), delta is negative.
        // We want Positive Angle for Left.
        let target_angle = -delta_x * deg_per_pixel;

        x_data.push(e.raw_yaw as f64);
        y_data.push(target_angle as f64);
    }

    // 3. Linear Regression (Least Squares)
    // y = mx + c
    // target_angle = gain * raw_yaw + offset

    let n = x_data.len() as f64;
    let sum_x: f64 = x_data.iter().sum();
    let sum_y: f64 = y_data.iter().sum();
    let sum_xy: f64 = x_data.iter().zip(&y_data).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = x_data.iter().map(|x| x * x).sum();

    let gain = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    let offset = (sum_y - gain * sum_x) / n;

    println!("--------------------------------------------------");
    println!("Proposed Calibration (Angle Space):");
    println!("yaw_gain: {:.6}", gain);
    println!("yaw_offset: {:.6}", offset);
    println!("--------------------------------------------------");

    // Validation
    let mut total_error = 0.0;
    for (raw, target) in x_data.iter().zip(&y_data) {
        let predicted = raw * gain + offset;
        let error = (predicted - target).abs();
        total_error += error;
        // println!("Raw: {:.2} -> Pred: {:.2} (Tgt: {:.2}) Err: {:.2}", raw, predicted, target, error);
    }
    println!("Mean Angular Error: {:.2} degrees", total_error / n);

    Ok(())
}
