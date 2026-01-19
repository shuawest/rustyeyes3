use image::ImageReader;
use rusty_eyes::config::AppConfig;
use rusty_eyes::gaze::L2CSPipeline;
use rusty_eyes::pipeline::Pipeline;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let config = AppConfig::load()?;
    let img_path = Path::new("calibration_data/img_1766014739324.jpg");
    let img = ImageReader::open(img_path)?.decode()?.to_rgb8();

    // We need to peek at the output tensor size.
    // L2CS Pipeline output hides the raw bins.
    // But `process_raw_values` debug prints `L2CS: idx=(..., ...)`.
    // The `idx` printed is the expectation index.
    // If I can't modify the code to print bins, I can assume it based on "L2CS-Net" standard.
    // However, I can temporarily hack `gaze.rs` again to print `bins` count.

    // Actually, let's just use the `trace_one` log I already have.
    // It didn't print bins.

    // I will modify `gaze.rs` directly to print "Model Output Bins: {}" in `process`.

    Ok(())
}
