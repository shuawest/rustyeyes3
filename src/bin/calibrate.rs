use rusty_eyes::config::AppConfig;
use rusty_eyes::gaze::{L2CSPipeline, MobileGazePipeline};
use rusty_eyes::pipeline::Pipeline;
use rusty_eyes::rectification::{CalibrationParams, CalibrationConfig};
use std::fs;
use std::path::{Path, PathBuf};
use image::io::Reader as ImageReader;
use image::{DynamicImage, GenericImageView};
use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =========================================================================
// Data Structures
// =========================================================================

#[derive(Debug)]
struct DataPoint {
    filename: String,
    raw_yaw: f32,
    raw_pitch: f32,
    target_x: f32,
    target_y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationMetrics {
    mean_error_px: f32,
    std_dev_px: f32,
    max_error_px: f32,
    histogram: Vec<HistogramBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CalibrationRun {
    run_id: String,
    timestamp: String,
    model: String,
    params: CalibrationParams,
    metrics: CalibrationMetrics,
}

type CalibrationHistory = Vec<CalibrationRun>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistogramBin {
    range: String,
    count: usize,
}

#[derive(Serialize)]
struct ReportSummary {
    mean_error: f32,
    std_dev: f32,
    max_error: f32,
    histogram: Vec<HistogramBin>,
}

#[derive(Serialize)]
struct ReportEntry {
    filename: String,
    target_x: f32,
    target_y: f32,
    raw_yaw: f32,
    raw_pitch: f32,
    calibrated_x: f32,
    calibrated_y: f32,
    delta_pixels: f32,
    error_percent: f32,
}

#[derive(Serialize)]
struct DetailedReport {
    run_id: String,
    timestamp: String,
    model: String,
    summary: ReportSummary,
    entries: Vec<ReportEntry>,
}

#[derive(Deserialize)]
struct JsonMeta {
    screen_x: f32,
    screen_y: f32,
}

// =========================================================================
// Optimization Logic
// =========================================================================

// Simple Pseudo-Random Number Generator to avoid adding `rand` dependency for now
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    fn next_f32(&mut self) -> f32 {
        // LCG constants
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Normalize to 0.0 - 1.0
        (self.state >> 32) as f32 / u32::MAX as f32
    }
    
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

fn optimize_params(entries: &[DataPoint], base_model: &str) -> (CalibrationParams, CalibrationMetrics) {
    println!("Optimizing {} using Analytic Least Squares...", base_model);
    
    let center_x = 1440.0 / 2.0;
    let center_y = 900.0 / 2.0;

    // Helper for Polynomial Regression (Y = ax^2 + bx + c)
    // Returns (a, b, c)
    // Using Cramer's Rule or basic Matrix Inversion for 3x3
    let solve_poly2 = |xs: &[f32], ys: &[f32]| -> (f32, f32, f32) {
        let n = xs.len() as f32;
        if n < 3.0 { return (0.0, 0.0, 0.0); }
        
        let mut s_x = 0.0; let mut s_x2 = 0.0; let mut s_x3 = 0.0; let mut s_x4 = 0.0;
        let mut s_y = 0.0; let mut s_xy = 0.0; let mut s_x2y = 0.0;
        
        for i in 0..xs.len() {
            let x = xs[i];
            let y = ys[i];
            let x2 = x*x;
            s_x += x;
            s_x2 += x2;
            s_x3 += x2*x;
            s_x4 += x2*x2;
            s_y += y;
            s_xy += x*y;
            s_x2y += x2*y;
        }
        
        // Matrix:
        // [ S_x4 S_x3 S_x2 ] [ a ]   [ S_x2y ]
        // [ S_x3 S_x2 S_x  ] [ b ] = [ S_xy  ]
        // [ S_x2 S_x  n    ] [ c ]   [ S_y   ]
        
        let m11 = s_x4; let m12 = s_x3; let m13 = s_x2;
        let m21 = s_x3; let m22 = s_x2; let m23 = s_x;
        let m31 = s_x2; let m32 = s_x;  let m33 = n;
        
        // Determinant
        let det = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
        
        if det.abs() < 1e-9 { return (0.0, 0.0, 0.0); } // Singular
        
        let det_a = s_x2y*(m22*m33 - m23*m32) - m12*(s_xy*m33 - s_y*m23) + m13*(s_xy*m32 - s_y*m22);
        let det_b = m11*(s_xy*m33 - s_y*m23) - s_x2y*(m21*m33 - m23*m31) + m13*(m21*s_y - s_xy*m31);
        let _det_c = m11*(m22*s_y - m32*s_xy) - m12*(m21*s_y - m31*s_xy) + s_x2y*(m21*m32 - m22*m31);
        let _det_c_corr = m11*(m22*s_y - s_xy*m32) - m12*(m21*s_y - s_xy*m31) + m13*(m21*s_xy - s_y*m32); 
        let _dc = m11*(m22*s_y - s_xy*m32) - m12*(m21*s_y - s_xy*m31) + s_x2y*(m21*m32 - m22*m31);
        
        // Correct form:
        let det_c_real = m11*(m22*s_y - s_xy*m32) - m12*(m21*s_y - s_xy*m31) + s_x2y*(m21*m32 - m22*m31);
        
        (det_a / det, det_b / det, det_c_real / det)
    };
    
    // -------------------------------------------------------------
    // RANSAC IMPLEMENTATION (Robust polynomial fit)
    // -------------------------------------------------------------
    // Iterates 1000 times selecting random subsets to find the model
    // that fits the most points (inliers) to exclude outliers.
    
    let iterations = 1000;
    let threshold_px = 150.0; // Inlier threshold
    let min_samples = 4; // Minimum points to fit quadratic (3 needed, 4 for safety)
    
    let n_points = entries.len();
    if n_points < min_samples {
         // Fallback to simple fit if not enough data for RANSAC
         println!("Not enough points for RANSAC ({} < {}), using global fit.", n_points, min_samples);
         return optimize_params_simple(entries, base_model, center_x, center_y);
    }

    let mut rng = SimpleRng::new(12345);
    
    let mut best_inlier_count = 0;
    let mut best_loss = f32::MAX;
    let mut best_params = CalibrationParams {
        yaw_offset: 0.0, pitch_offset: 0.0,
        yaw_gain: 1.0, pitch_gain: 1.0,
        yaw_curve: 0.0, pitch_curve: 0.0,
    };
    
    let magic = 20.0;
    
    println!("Running RANSAC with {} iterations...", iterations);
    
    for _ in 0..iterations {
        // 1. Pick Random Subset
        let mut subset_indices = Vec::with_capacity(min_samples);
        while subset_indices.len() < min_samples {
             let idx = (rng.next_f32() * n_points as f32) as usize;
             if !subset_indices.contains(&idx) {
                 subset_indices.push(idx);
             }
        }
        
        let subset_yaws: Vec<f32> = subset_indices.iter().map(|&i| entries[i].raw_yaw).collect();
        let subset_tgt_xs: Vec<f32> = subset_indices.iter().map(|&i| entries[i].target_x - center_x).collect();
        let subset_pitches: Vec<f32> = subset_indices.iter().map(|&i| entries[i].raw_pitch).collect();
        let subset_tgt_ys: Vec<f32> = subset_indices.iter().map(|&i| entries[i].target_y - center_y).collect();

        // 2. Solve for Subset
        let (a_y, b_y, c_y) = solve_poly2(&subset_yaws, &subset_tgt_xs);
        let (a_p, b_p, c_p) = solve_poly2(&subset_pitches, &subset_tgt_ys);
        
        let params = CalibrationParams {
            yaw_curve: a_y / magic,
            yaw_gain: b_y / magic,
            yaw_offset: c_y / magic,
            pitch_curve: a_p / magic,
            pitch_gain: b_p / magic,
            pitch_offset: c_p / magic,
        };
        
        // 3. Count Inliers
        let mut inliers = 0;
        let mut total_subset_loss = 0.0;
        
        for e in entries {
            let y = e.raw_yaw;
            let p = e.raw_pitch;
            
            let pred_x = center_x + (params.yaw_curve * y*y + params.yaw_gain * y + params.yaw_offset) * magic;
            let pred_y = center_y + (params.pitch_curve * p*p + params.pitch_gain * p + params.pitch_offset) * magic;
            
            let dx = pred_x - e.target_x;
            let dy = pred_y - e.target_y;
            let err = (dx*dx + dy*dy).sqrt();
            
            if err < threshold_px {
                inliers += 1;
                total_subset_loss += err;
            }
        }
        
        // 4. Update Best
        // Prefer more inliers. Tie-break with lower inlier loss.
        if inliers > best_inlier_count || (inliers == best_inlier_count && total_subset_loss < best_loss) {
            best_inlier_count = inliers;
            best_loss = total_subset_loss;
            best_params = params;
        }
    }
    
    // 5. Final Re-fit on ALL Inliers of Best Model (Polishing)
    println!("RANSAC Best: {}/{} inliers. Polishing...", best_inlier_count, n_points);
    
    let mut final_yaws = Vec::new();
    let mut final_tgt_xs = Vec::new();
    let mut final_pitches = Vec::new();
    let mut final_tgt_ys = Vec::new();
    
    for e in entries {
        let y = e.raw_yaw;
        let p = e.raw_pitch;
        
        let pred_x = center_x + (best_params.yaw_curve * y*y + best_params.yaw_gain * y + best_params.yaw_offset) * magic;
        let pred_y = center_y + (best_params.pitch_curve * p*p + best_params.pitch_gain * p + best_params.pitch_offset) * magic;
        
        let dx = pred_x - e.target_x;
        let dy = pred_y - e.target_y;
        let err = (dx*dx + dy*dy).sqrt();
        
        if err < threshold_px {
             final_yaws.push(e.raw_yaw);
             final_tgt_xs.push(e.target_x - center_x);
             final_pitches.push(e.raw_pitch);
             final_tgt_ys.push(e.target_y - center_y);
        }
    }
    
    // Check if we have enough inliers for refit, else keep robust best
    if final_yaws.len() >= min_samples {
         let (a_y, b_y, c_y) = solve_poly2(&final_yaws, &final_tgt_xs);
         let (a_p, b_p, c_p) = solve_poly2(&final_pitches, &final_tgt_ys);
         
         best_params.yaw_curve = a_y / magic;
         best_params.yaw_gain = b_y / magic;
         best_params.yaw_offset = c_y / magic;
         best_params.pitch_curve = a_p / magic;
         best_params.pitch_gain = b_p / magic;
         best_params.pitch_offset = c_p / magic;
    }
    
    // --- End RANSAC ---

    let metrics = calculate_metrics(entries, &best_params);
    (best_params, metrics)
}

fn optimize_params_simple(entries: &[DataPoint], _base_model: &str, center_x: f32, center_y: f32) -> (CalibrationParams, CalibrationMetrics) {
    // Helper for Polynomial Regression (Y = ax^2 + bx + c)
    // Returns (a, b, c)
    let solve_poly2 = |xs: &[f32], ys: &[f32]| -> (f32, f32, f32) {
        let n = xs.len() as f32;
        let s_x: f32 = xs.iter().sum();
        let s_x2: f32 = xs.iter().map(|x| x*x).sum();
        let s_x3: f32 = xs.iter().map(|x| x*x*x).sum();
        let s_x4: f32 = xs.iter().map(|x| x*x*x*x).sum();
        let s_y: f32 = ys.iter().sum();
        let s_xy: f32 = xs.iter().zip(ys.iter()).map(|(x, y)| x*y).sum();
        let s_x2y: f32 = xs.iter().zip(ys.iter()).map(|(x, y)| x*x*y).sum();
        
        // Matrix M = 
        // [ s_x4 s_x3 s_x2 ]
        // [ s_x3 s_x2 s_x  ]
        // [ s_x2 s_x  n    ]
        
        let m11 = s_x4; let m12 = s_x3; let m13 = s_x2;
        let m21 = s_x3; let m22 = s_x2; let m23 = s_x;
        let m31 = s_x2; let m32 = s_x;  let m33 = n;
        
        let det = m11*(m22*m33 - m23*m32) - m12*(m21*m33 - m23*m31) + m13*(m21*m32 - m22*m31);
        
        if det.abs() < 1e-9 { return (0.0, 0.0, 0.0); }
        
        let det_a = s_x2y*(m22*m33 - m23*m32) - m12*(s_xy*m33 - s_y*m23) + m13*(s_xy*m32 - s_y*m22);
        let det_b = m11*(s_xy*m33 - s_y*m23) - s_x2y*(m21*m33 - m23*m31) + m13*(m21*s_y - s_xy*m31);
        let _det_c = m11*(m22*s_y - m32*s_xy) - m12*(m21*s_y - m31*s_xy) + s_x2y*(m21*m32 - m22*m31);
        let _det_c_corr = m11*(m22*s_y - s_xy*m32) - m12*(m21*s_y - s_xy*m31) + m13*(m21*s_xy - s_y*m32);
        let det_c_real = m11*(m22*s_y - s_xy*m32) - m12*(m21*s_y - s_xy*m31) + s_x2y*(m21*m32 - m22*m31);
        
        (det_a / det, det_b / det, det_c_real / det)
    };
    
    let magic = 20.0;
    
    let raw_yaws: Vec<f32> = entries.iter().map(|e| e.raw_yaw).collect();
    let tgt_xs: Vec<f32> = entries.iter().map(|e| e.target_x - center_x).collect();
    let (a_yaw, b_yaw, c_yaw) = solve_poly2(&raw_yaws, &tgt_xs);
    
    let raw_pitches: Vec<f32> = entries.iter().map(|e| e.raw_pitch).collect();
    let tgt_ys: Vec<f32> = entries.iter().map(|e| e.target_y - center_y).collect(); 
    let (a_pitch, b_pitch, c_pitch) = solve_poly2(&raw_pitches, &tgt_ys);
    
    let best_params = CalibrationParams {
        yaw_offset: c_yaw / magic,
        pitch_offset: c_pitch / magic,
        yaw_gain: b_yaw / magic,
        pitch_gain: b_pitch / magic,
        yaw_curve: a_yaw / magic,
        pitch_curve: a_pitch / magic,
    };
    
    let metrics = calculate_metrics(entries, &best_params);
    (best_params, metrics)
}

fn evaluate_loss(entries: &[DataPoint], params: &CalibrationParams) -> f32 {
    let mut total_loss = 0.0;
    let center_x = 1440.0 / 2.0;
    let center_y = 900.0 / 2.0;
    let magic = 20.0;
    
    for e in entries {
        let pred_x = center_x + (e.raw_yaw - params.yaw_offset) * params.yaw_gain * magic;
        let pred_y = center_y + (e.raw_pitch - params.pitch_offset) * params.pitch_gain * magic; // Corrected sign error in GD derivation above?
        // Wait, derivation: sy = cy + ((raw_pitch - po) * pg) * magic
        // In Gaze.rs: sy -= pitch * 20.0
        // pitch = -(raw - off) * gain
        // sy -= (-(raw - off) * gain) * 20.0
        // sy += (raw - off) * gain * 20.0
        // Derivation matches.
        
        let err_x = pred_x - e.target_x;
        let err_y = pred_y - e.target_y;
        total_loss += err_x*err_x + err_y*err_y;
    }
    total_loss / entries.len() as f32
}

fn calculate_metrics(entries: &[DataPoint], params: &CalibrationParams) -> CalibrationMetrics {
    let mut total_error = 0.0;
    let mut errors = Vec::new();
    let center_x = 1440.0 / 2.0;
    let center_y = 900.0 / 2.0;
    let magic = 20.0;

    for e in entries {
         // Polynomial Evaluation
         let y = e.raw_yaw;
         let p = e.raw_pitch;
         
         let pred_x = center_x + (params.yaw_curve * y * y + params.yaw_gain * y + params.yaw_offset) * magic;
         // Pitch logic: p_gained = -(poly(p)). ScreenY = CenterY - p_gained*20 = CenterY + poly(p)*20.
         // Solver solves for TgtY - CenterY.
         let pred_y = center_y + (params.pitch_curve * p * p + params.pitch_gain * p + params.pitch_offset) * magic;
         
         let dx = pred_x - e.target_x;
         let dy = pred_y - e.target_y;
         let dist = (dx*dx + dy*dy).sqrt();
         
         total_error += dist;
         errors.push(dist);
    }
    
    let n = entries.len() as f32;
    let mean = total_error / n;
    let variance = errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    let max_error = errors.iter().cloned().fold(0.0/0.0, f32::max); // NaN safe max
    
    CalibrationMetrics {
        mean_error_px: mean,
        std_dev_px: std_dev,
        max_error_px: max_error,
        histogram: Vec::new(), // Optimizer doesn't need histogram
    }
}

// =========================================================================
// Evaluation & Reporting Logic
// =========================================================================

fn evaluate_and_report(
    run_id: &str,
    model_name: &str,
    dataset: &[DataPoint],
    params: &CalibrationParams,
    cx: f32, 
    cy: f32
) -> (CalibrationMetrics, DetailedReport) {
    let mut total_sq_error = 0.0;
    let mut max_errors = 0.0;
    let mut total_error = 0.0;
    
    // Histogram buckets
    let mut hist_0_50 = 0;
    let mut hist_50_100 = 0;
    let mut hist_100_200 = 0;
    let mut hist_200_500 = 0;
    let mut hist_500_plus = 0;
    
    let magic = 20.0;
    let mut report_entries = Vec::new();
    
    for e in dataset {
        // Polynomial Evaluation
        let y = e.raw_yaw;
        let p = e.raw_pitch;
        
        let pred_x_px = cx + (params.yaw_curve * y * y + params.yaw_gain * y + params.yaw_offset) * magic;
        let pred_y_px = cy + (params.pitch_curve * p * p + params.pitch_gain * p + params.pitch_offset) * magic;
        
        let dx = pred_x_px - e.target_x;
        let dy = pred_y_px - e.target_y;
        let dist = (dx*dx + dy*dy).sqrt(); // L2 euclidean
        
        total_sq_error += dist * dist;
        total_error += dist;
        if dist > max_errors { max_errors = dist; }
        
        if dist < 50.0 { hist_0_50 += 1; }
        else if dist < 100.0 { hist_50_100 += 1; }
        else if dist < 200.0 { hist_100_200 += 1; }
        else if dist < 500.0 { hist_200_500 += 1; }
        else { hist_500_plus += 1; }

        let diag = (cx*2.0 * cx*2.0 + cy*2.0 * cy*2.0).sqrt();
        let pct = (dist / diag) * 100.0;

        report_entries.push(ReportEntry {
            filename: e.filename.clone(),
            target_x: e.target_x,
            target_y: e.target_y,
            raw_yaw: e.raw_yaw,
            raw_pitch: e.raw_pitch,
            calibrated_x: pred_x_px,
            calibrated_y: pred_y_px,
            delta_pixels: dist,
            error_percent: pct,
        });
    }
    
    let n = dataset.len() as f32;
    let mean = total_error / n;
    let variance = (total_sq_error / n) - (mean * mean);
    let std_dev = variance.sqrt();

    let histogram = vec![
        HistogramBin { range: "0-50px".to_string(), count: hist_0_50 },
        HistogramBin { range: "50-100px".to_string(), count: hist_50_100 },
        HistogramBin { range: "100-200px".to_string(), count: hist_100_200 },
        HistogramBin { range: "200-500px".to_string(), count: hist_200_500 },
        HistogramBin { range: "500px+".to_string(), count: hist_500_plus },
    ];
    
    let metrics = CalibrationMetrics {
        mean_error_px: mean,
        std_dev_px: std_dev,
        max_error_px: max_errors,
        histogram: histogram.clone(),
    };
    
    let report = DetailedReport {
        run_id: run_id.to_string(),
        timestamp: Local::now().to_rfc3339(),
        model: model_name.to_string(),
        summary: ReportSummary {
            mean_error: mean,
            std_dev: std_dev,
            max_error: max_errors,
            histogram,
        },
        entries: report_entries,
    };
    
    (metrics, report)
}

// =========================================================================
// Main
// =========================================================================

fn main() -> anyhow::Result<()> {
    let mut config = AppConfig::load()?;
    let data_dir = Path::new("calibration_data");
    let reports_dir = data_dir.join("reports");
    
    // Ensure reports dir exists
    if !reports_dir.exists() {
        fs::create_dir_all(&reports_dir)?;
    }
    
    let history_file = reports_dir.join("calibration_history.json");
    
    let run_id = Local::now().format("%Y%m%d_%H%M%S").to_string();
    println!("Starting Calibration Run: {}", run_id);
    
    // 1. Load History
    let mut history: CalibrationHistory = if history_file.exists() {
        let content = fs::read_to_string(&history_file)?;
        serde_json::from_str(&content).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    };
    
    // 2. Gather Data (same as before)
    // 2. Gather Data (Replaced by Samples Logic below)
    let screen_w = 1440.0;
    let screen_h = 900.0;
    let cx = screen_w / 2.0;
    let cy = screen_h / 2.0;

    // 3. Collect Datasets (L2CS & Mobile)
    let mut dataset_l2cs = Vec::new();
    let mut dataset_mobile = Vec::new();
    
    // Helper to collect inference data
    // We assume data_dir has pairs of .jpg and .json
    let mut samples: Vec<(PathBuf, PathBuf)> = Vec::new();
    if data_dir.exists() {
         for entry in std::fs::read_dir(data_dir)? {
             let entry = entry?;
             let path = entry.path();
             if path.extension().and_then(|s| s.to_str()) == Some("json") {
                  let img_path = path.with_extension("jpg");
                  if img_path.exists() {
                      samples.push((img_path, path));
                  }
             }
         }
    }
    
    // L2CS Inference
    {
        let mut l2cs = L2CSPipeline::new(
            &config.models.l2cs_path,
            &config.models.face_mesh_path,
            &config.models.face_detection_path,
        )?;
        l2cs.params = CalibrationParams {
            yaw_offset: 0.0, pitch_offset: 0.0,
            yaw_gain: 1.0, pitch_gain: 1.0, 
            yaw_curve: 0.0, pitch_curve: 0.0,
        };
        
        println!("Running L2CS Inference (1 pass)...");
        for (img_path, json_path) in &samples {
            let json_str = fs::read_to_string(json_path)?;
            let meta: JsonMeta = serde_json::from_str(&json_str)?;
            let original_img = ImageReader::open(img_path)?.with_guessed_format()?.decode()?.to_rgb8();
            
            // Single Pass (Deterministic)
            if let Ok(Some(output)) = l2cs.process_raw_values(&original_img) {
                 if let rusty_eyes::types::PipelineOutput::Gaze { yaw, pitch, .. } = output {
                     dataset_l2cs.push(DataPoint {
                         filename: img_path.file_name().unwrap().to_string_lossy().to_string(),
                         raw_yaw: yaw, 
                         raw_pitch: -pitch, 
                         target_x: meta.screen_x,
                         target_y: meta.screen_y,
                     });
                 }
            }
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        println!("\nL2CS: Collected {} points.", dataset_l2cs.len());
    }

    // Mobile Inference
    {
        let mut mobile = MobileGazePipeline::new(
            &config.models.mobile_path,
            &config.models.face_mesh_path,
            &config.models.face_detection_path,
        )?;
        mobile.params = CalibrationParams {
            yaw_offset: 0.0, pitch_offset: 0.0,
            yaw_gain: 1.0, pitch_gain: 1.0,
            yaw_curve: 0.0, pitch_curve: 0.0,
        };
        
        println!("Running Mobile Inference (1 pass)...");
        for (img_path, json_path) in &samples {
            let json_str = fs::read_to_string(json_path)?;
            let meta: JsonMeta = serde_json::from_str(&json_str)?;
            let original_img = ImageReader::open(img_path)?.with_guessed_format()?.decode()?.to_rgb8();
            
            if let Ok(Some(output)) = mobile.process_raw_values(&original_img) {
                 if let rusty_eyes::types::PipelineOutput::Gaze { yaw, pitch, .. } = output {
                     // Mobile raw output needs internal inversion? 
                     // Recent fix in gaze.rs inverted it. 
                     // Assuming process_raw_values returns the standardized values.
                     dataset_mobile.push(DataPoint {
                         filename: img_path.file_name().unwrap().to_string_lossy().to_string(),
                         raw_yaw: yaw, 
                         raw_pitch: -pitch, 
                         target_x: meta.screen_x,
                         target_y: meta.screen_y,
                     });
                 }
            }
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        println!("\nMobile: Collected {} points.", dataset_mobile.len());
    }

    // 4. Optimize & Process Reports
    let process_model = |
        name: &str, 
        dataset: &[DataPoint], 
        history: &mut CalibrationHistory, 
        cfg_map: &mut HashMap<String, CalibrationParams>
    | -> anyhow::Result<()> {
        if dataset.is_empty() { return Ok(()); }
        
        println!("Optimizing {}...", name);
        let (mut params, _opt_metrics) = optimize_params(dataset, name);
        
        // STRATEGIC BOOST: Detect Head Turn and Boost Sensitivity for Eye Comfort
        if name == "l2cs" {
            let min_yaw = dataset.iter().map(|p| p.raw_yaw).fold(f32::INFINITY, f32::min);
            let max_yaw = dataset.iter().map(|p| p.raw_yaw).fold(f32::NEG_INFINITY, f32::max);
            let range = max_yaw - min_yaw;
            println!("Detected Yaw Range: {:.1} degrees", range);
            
            if range > 30.0 {
                let comfortable_range = 20.0;
                let boost = range / comfortable_range;
                println!("Head Movement Detected (> 30 deg). Applying Sensitivity Boost {:.2}x (Targeting {} deg eye range)", boost, comfortable_range);
                params.yaw_gain *= boost;
                // Also boost curve? Usually simpler to just boost linear gain.
            }
        }

        let (metrics, report) = evaluate_and_report(&run_id, name, dataset, &params, cx, cy);
        
        println!("{} Results (Boosted): MeanErr={:.2}px, StdDev={:.2}px", name, metrics.mean_error_px, metrics.std_dev_px);
        
        // Save Run to History
        let run = CalibrationRun {
            run_id: run_id.clone(),
            timestamp: Local::now().to_rfc3339(),
            model: name.to_string(),
            params: params.clone(),
            metrics: metrics.clone(),
        };
        history.push(run.clone());
        
        // Save Report File
        let report_name = format!("calibration_report_{}_{}.json", name, run_id);
        let report_path = reports_dir.join(report_name);
        fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
        println!("Saved report to {:?}", report_path);
        
        // Always update best model with this run for now (since we applied manual boost)
        cfg_map.insert(name.to_string(), params.clone());
        
        Ok(())
    };

    process_model("l2cs", &dataset_l2cs, &mut history, &mut config.calibration.models)?;
    process_model("mobile", &dataset_mobile, &mut history, &mut config.calibration.models)?;
    
    // 5. Save State
    fs::write(&history_file, serde_json::to_string_pretty(&history)?)?;
    println!("Updated history at {:?}", history_file);
    
    config.save()?;
    println!("Updated config.json with best parameters.");
    
    Ok(())
}
