use rusty_eyes::config::AppConfig;
use rusty_eyes::gaze::{L2CSPipeline, MobileGazePipeline};
use rusty_eyes::pipeline::Pipeline;
use rusty_eyes::rectification::{CalibrationParams, CalibrationConfig};
use std::fs;
use std::path::{Path, PathBuf};
use image::ImageReader;
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

#[derive(Serialize)]
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

fn optimize_params(
    dataset: &[DataPoint], 
    cx: f32, 
    cy: f32
) -> CalibrationParams {
    let mut y_off = 0.0f32;
    let mut p_off = 0.0f32;
    let mut y_gain = 5.0f32;
    let mut p_gain = 5.0f32;
    
    let lr = 0.00001; 
    let iterations = 5000;
    
    for _ in 0..iterations {
        let mut grad_y_off = 0.0;
        let mut grad_p_off = 0.0;
        let mut grad_y_gain = 0.0;
        let mut grad_p_gain = 0.0;
        
        for p in dataset {
             let tx_rel = p.target_x - cx;
             let ty_rel = p.target_y - cy;
             
             let pred_x = (p.raw_yaw - y_off) * y_gain;
             let pred_y = (p.raw_pitch - p_off) * p_gain; 
             
             let err_x = pred_x - tx_rel;
             let err_y = pred_y - ty_rel;
             
             grad_y_gain += 2.0 * err_x * (p.raw_yaw - y_off);
             grad_y_off += 2.0 * err_x * (-y_gain);
             
             grad_p_gain += 2.0 * err_y * (p.raw_pitch - p_off);
             grad_p_off += 2.0 * err_y * (-p_gain);
        }
        
        let n = dataset.len() as f32;
        y_gain -= lr * (grad_y_gain / n);
        y_off -= lr * 0.01 * (grad_y_off / n); 
        p_gain -= lr * (grad_p_gain / n);
        p_off -= lr * 0.01 * (grad_p_off / n);
    }
    
    let magic_factor = 20.0;
    
    CalibrationParams {
        yaw_offset: y_off,
        pitch_offset: p_off,
        yaw_gain: y_gain / magic_factor,
        pitch_gain: -(p_gain / magic_factor),
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
    let magic_factor = 20.0;
    let mut entries = Vec::new();
    
    let mut total_error = 0.0;
    let mut error_sq_sum = 0.0;
    let mut max_error = 0.0f32;
    
    // Histogram buckets: 0-50, 50-100, 100-200, 200-500, 500+
    let mut hist_counts = vec![0; 5];
    
    for p in dataset {
        // Apply Params
        let cal_x = cx + ((p.raw_yaw - params.yaw_offset) * params.yaw_gain * magic_factor);
        let sol_pred_y = (p.raw_pitch - params.pitch_offset) * (-params.pitch_gain * magic_factor);
        let cal_y = cy + sol_pred_y;

        let dx = cal_x - p.target_x;
        let dy = cal_y - p.target_y;
        let delta = (dx*dx + dy*dy).sqrt();
        
        if delta > max_error { max_error = delta; }
        total_error += delta;
        error_sq_sum += delta * delta;
        
        // Histogram
        if delta < 50.0 { hist_counts[0] += 1; }
        else if delta < 100.0 { hist_counts[1] += 1; }
        else if delta < 200.0 { hist_counts[2] += 1; }
        else if delta < 500.0 { hist_counts[3] += 1; }
        else { hist_counts[4] += 1; }

        let diag = (cx*2.0 * cx*2.0 + cy*2.0 * cy*2.0).sqrt();
        let pct = (delta / diag) * 100.0;

        entries.push(ReportEntry {
            filename: p.filename.clone(),
            target_x: p.target_x,
            target_y: p.target_y,
            raw_yaw: p.raw_yaw,
            raw_pitch: p.raw_pitch,
            calibrated_x: cal_x,
            calibrated_y: cal_y,
            delta_pixels: delta,
            error_percent: pct,
        });
    }
    
    let n = dataset.len() as f32;
    let mean_error = total_error / n;
    let variance = (error_sq_sum / n) - (mean_error * mean_error);
    let std_dev = variance.sqrt();
    
    let metrics = CalibrationMetrics {
        mean_error_px: mean_error,
        std_dev_px: std_dev,
        max_error_px: max_error,
    };
    
    let histogram = vec![
        HistogramBin { range: "0-50px".to_string(), count: hist_counts[0] },
        HistogramBin { range: "50-100px".to_string(), count: hist_counts[1] },
        HistogramBin { range: "100-200px".to_string(), count: hist_counts[2] },
        HistogramBin { range: "200-500px".to_string(), count: hist_counts[3] },
        HistogramBin { range: "500px+".to_string(), count: hist_counts[4] },
    ];
    
    let report = DetailedReport {
        run_id: run_id.to_string(),
        timestamp: Local::now().to_rfc3339(),
        model: model_name.to_string(),
        summary: ReportSummary {
            mean_error,
            std_dev,
            max_error,
            histogram,
        },
        entries,
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
    let mut samples: Vec<(PathBuf, PathBuf)> = Vec::new(); 
    if data_dir.exists() {
        for entry in fs::read_dir(data_dir)? {
             let entry = entry?;
             let path = entry.path();
             if path.extension().map_or(false, |e| e == "json") {
                 if let Some(stem) = path.file_stem() {
                     let img_name = format!("{}.jpg", stem.to_string_lossy());
                     let img_path = data_dir.join(img_name);
                     if img_path.exists() {
                         samples.push((img_path, path));
                     }
                 }
             }
        }
    }
    
    println!("Found {} calibration pairs.", samples.len());
    if samples.is_empty() { return Ok(()); }

    let screen_w = 1440.0;
    let screen_h = 900.0;
    let cx = screen_w / 2.0;
    let cy = screen_h / 2.0;

    // 3. Collect Datasets (L2CS & Mobile)
    let mut dataset_l2cs = Vec::new();
    let mut dataset_mobile = Vec::new();
    
    // L2CS Inference
    {
        let mut l2cs = L2CSPipeline::new(
            &config.models.l2cs_path,
            &config.models.face_mesh_path,
            &config.models.face_detection_path,
        )?;
        l2cs.params = CalibrationParams::default(); 
        
        println!("Running L2CS Inference...");
        for (img_path, json_path) in &samples {
            let json_str = fs::read_to_string(json_path)?;
            let meta: JsonMeta = serde_json::from_str(&json_str)?;
            let img = ImageReader::open(img_path)?.decode()?.to_rgb8();
            
            if let Ok(Some(output)) = l2cs.process_raw_values(&img) {
                 if let rusty_eyes::types::PipelineOutput::Gaze { yaw, pitch, .. } = output {
                     dataset_l2cs.push(DataPoint {
                         filename: img_path.file_name().unwrap().to_string_lossy().to_string(),
                         raw_yaw: yaw, 
                         raw_pitch: pitch, 
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
        mobile.params = CalibrationParams::default(); 
        
        println!("Running Mobile Inference...");
        for (img_path, json_path) in &samples {
            let json_str = fs::read_to_string(json_path)?;
            let meta: JsonMeta = serde_json::from_str(&json_str)?;
            let img = ImageReader::open(img_path)?.decode()?.to_rgb8();
            
            if let Ok(Some(output)) = mobile.process_raw_values(&img) {
                 if let rusty_eyes::types::PipelineOutput::Gaze { yaw, pitch, .. } = output {
                     dataset_mobile.push(DataPoint {
                         filename: img_path.file_name().unwrap().to_string_lossy().to_string(),
                         raw_yaw: yaw, 
                         raw_pitch: pitch, 
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
        let params = optimize_params(dataset, cx, cy);
        let (metrics, report) = evaluate_and_report(&run_id, name, dataset, &params, cx, cy);
        
        println!("{} Results: MeanErr={:.2}px, StdDev={:.2}px", name, metrics.mean_error_px, metrics.std_dev_px);
        
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
        
        // Best Model Selection
        // Filter history for this model
        let model_history: Vec<&CalibrationRun> = history.iter()
            .filter(|r| r.model == name)
            .collect();
            
        // Find best by mean_error
        if let Some(best_run) = model_history.iter().min_by(|a, b| 
            a.metrics.mean_error_px.partial_cmp(&b.metrics.mean_error_px).unwrap_or(std::cmp::Ordering::Equal)
        ) {
            println!("Best {} found: Run {} (MeanErr={:.2}px)", name, best_run.run_id, best_run.metrics.mean_error_px);
            cfg_map.insert(name.to_string(), best_run.params.clone());
        }
        
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
