use rusty_eyes::config::AppConfig;
use rusty_eyes::gaze::L2CSPipeline;
use image::ImageReader;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let config = AppConfig::load()?;
    // Pick a sample
    let img_path = Path::new("calibration_data/img_1766014739324.jpg");
    println!("Loading {:?}", img_path);
    let img = ImageReader::open(img_path)?.decode()?.to_rgb8();
    
    // We need to access the internal crop logic.
    // Since `process_raw_values` encapsulates it, we might need to modify `L2CSPipeline` 
    // to expose the crop, OR just copy-paste the crop logic here for reproduction.
    // Copy-paste is safer than changing library code just for debug.
    
    // Replication of L2CS pipeline logic:
    // 1. Detect Face
    // 2. Crop
    // 3. Resize
    
    // We can use the public API if we modify the library temp, 
    // OR we can just use the `face_detection` model directly if we can load it.
    
    // Easier path: `L2CSPipeline` has `detect_face`.
    // It's private logic inside `process_raw_values`.
    // I will temporarily modify `src/gaze.rs` to save the crop to "debug_face_crop.jpg" inside `process_raw_values`.
    
    let mut l2cs = L2CSPipeline::new(
        &config.models.l2cs_path,
        &config.models.face_mesh_path,
        &config.models.face_detection_path,
    )?;
    
    println!("Running inference (and triggering debug save)...");
    let _ = l2cs.process_raw_values(&img)?;
    
    println!("Done. Checking 'debug_face_crop.png' dimensions...");
    if Path::new("debug_face_crop.png").exists() {
        let crop = ImageReader::open("debug_face_crop.png")?.decode()?;
        println!("Crop Dimensions: {}x{}", crop.width(), crop.height());
    } else {
        println!("Error: debug_face_crop.png not found.");
    }
    Ok(())
}

