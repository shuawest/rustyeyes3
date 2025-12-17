use clap::Parser;

mod args;
mod camera;
mod inference;
mod output;
mod types;
mod detector;
mod pipeline;
mod head_pose;
mod gaze;
mod overlay;

use args::Args;
use camera::CameraSource;
use output::WindowOutput;
use pipeline::Pipeline;
use types::PipelineOutput;
use overlay::OverlayWindow;

fn create_pipeline(name: &str) -> Box<dyn Pipeline> {
    match name {
        "detection" => Box::new(inference::FaceDetectionPipeline::new("face_detection.onnx").unwrap()),
        "pose" => Box::new(head_pose::HeadPosePipeline::new("head_pose.onnx", "face_detection.onnx").expect("Failed to load Pose")),
        "head_gaze" => Box::new(gaze::SimulatedGazePipeline::new("face_mesh.onnx", "head_pose.onnx", "face_detection.onnx").expect("Failed to load Head Gaze")),
        "pupil_gaze" => Box::new(gaze::PupilGazePipeline::new("face_mesh.onnx", "head_pose.onnx", "face_detection.onnx").expect("Failed to load Pupil Gaze")),
        "gaze" => Box::new(gaze::SimulatedGazePipeline::new("face_mesh.onnx", "head_pose.onnx", "face_detection.onnx").expect("Failed to load Gaze")), // Default alias
        _ => Box::new(inference::FaceMeshPipeline::new("face_mesh.onnx", "face_detection.onnx").expect("Failed to load mesh")),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.list {
        let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
        println!("Available Cameras:");
        println!("{:<5} | {:<30} | {:<10}", "Index", "Name", "Misc");
        println!("{}", "-".repeat(60));
        for cam in cameras {
            println!("{:<5} | {:<30} | {:?}", cam.index(), cam.human_name(), cam.misc());
        }
        return Ok(());
    }

    // 1. Setup Camera
    let index = args.cam_index as usize;
    let mut camera = CameraSource::new(index)?;
    println!("Opened camera: {}", camera.name());

    // 2. Setup Inference
    let mut current_pipeline = create_pipeline(&args.model.unwrap_or_else(|| "mesh".to_string()));
    println!("Active Pipeline: {}", current_pipeline.name());

    // 3. Setup Output
    // We get the actual format from the camera
    let width = camera.width();
    let height = camera.height();
    let mut window = WindowOutput::new("Rusty Eyes", width as usize, height as usize)?; // Added ?
    println!("Window created successfully.");

    println!("Starting Pipeline...");
    println!("Controls: [1-3] Basic [4] Head Gaze [5] Pupil Gaze [6] Toggle Overlay");

    // State for Overlay
    let mut show_overlay = false;
    let mut overlay_window: Option<OverlayWindow> = None;
    let screen_w = 1440; // Default Mac, ideally get from OS but minifb doesn't support it easily.
    let screen_h = 900;

    // 4. Loop
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        // Swap Pipeline?
        if window.is_key_down(minifb::Key::Key1) {
            current_pipeline = create_pipeline("mesh");
            println!("Switched to: {}", current_pipeline.name());
        }
        if window.is_key_down(minifb::Key::Key2) {
            current_pipeline = create_pipeline("detection");
            println!("Switched to: {}", current_pipeline.name());
        }
        if window.is_key_down(minifb::Key::Key3) {
            current_pipeline = create_pipeline("pose");
            println!("Switched to: {}", current_pipeline.name());
        }
        if window.is_key_down(minifb::Key::Key4) {
             current_pipeline = create_pipeline("head_gaze");
             println!("Switched to: {}", current_pipeline.name());
        }
        if window.is_key_down(minifb::Key::Key5) {
             current_pipeline = create_pipeline("pupil_gaze");
             println!("Switched to: {}", current_pipeline.name());
        }
        
        // Toggle Overlay
        if window.is_key_down(minifb::Key::Key6) {
             show_overlay = !show_overlay;
             println!("Overlay Mode: {}", show_overlay);
             std::thread::sleep(std::time::Duration::from_millis(200)); // Debounce
        }

        if let Ok(mut frame) = camera.capture() {
            // Mirror if requested
            if args.mirror {
                image::imageops::flip_horizontal_in_place(&mut frame);
            }

            let landmarks = current_pipeline.process(&frame)?;
            
            // Draw logic based on output type
            let mut display_buffer = frame.to_vec(); // clone for drawing
            // Simple drawing (inefficient but works)
            
            // --- OVERLAY SIDE CAR LOGIC ---
            if show_overlay {
                 if overlay_window.is_none() {
                     match OverlayWindow::new(screen_w, screen_h) {
                        Ok(win) => {
                             overlay_window = Some(win);
                             println!("Overlay Sidecar Launched");
                        },
                        Err(e) => println!("Failed to launch overlay: {}", e),
                     }
                 }
                 
                 if let Some(win) = overlay_window.as_mut() {
                     if let Some(PipelineOutput::Gaze { yaw, pitch, .. }) = landmarks.as_ref() {
                          let gain_x = 25.0; 
                          let gain_y = 25.0;
                          
                          let cx = screen_w as f32 / 2.0;
                          let cy = screen_h as f32 / 2.0;
                          
                          let sx = cx - (yaw * gain_x);
                          let sy = cy - (pitch * gain_y);
                          
                          // Send to Sidecar
                          let _ = win.update(sx, sy);
                     }
                 }
            } else {
                 if overlay_window.is_some() {
                     overlay_window = None;
                     println!("Overlay Sidecar Closed");
                 }
            }
            // ---------------------
            
            if let Some(output) = landmarks {
                match output {
                    PipelineOutput::Landmarks(l) => {
                         // Draw points
                         for point in l.points {
                             let x = point.x as usize;
                             let y = point.y as usize;
                             if x < width as usize && y < height as usize {
                                 // Draw red dot
                                 for dy in 0..2 { for dx in 0..2 {
                                     let idx = ((y + dy) * width as usize + (x + dx)) * 3;
                                     if idx < display_buffer.len() {
                                         display_buffer[idx] = 255; display_buffer[idx+1] = 0; display_buffer[idx+2] = 0;
                                     }
                                 }}
                             }
                         }
                    },
                    PipelineOutput::FaceRects(rects) => {
                         for r in rects {
                             let x = r.x as usize; let y = r.y as usize; let w = r.width as usize; let h = r.height as usize;
                             // Draw box (Green)
                             for i in x..(x+w).min(width as usize) {
                                  let idx1 = (y * width as usize + i) * 3;
                                  let idx2 = ((y+h).min(height as usize-1) * width as usize + i) * 3;
                                  if idx1 < display_buffer.len() { display_buffer[idx1] = 0; display_buffer[idx1+1] = 255; display_buffer[idx1+2] = 0; }
                                  if idx2 < display_buffer.len() { display_buffer[idx2] = 0; display_buffer[idx2+1] = 255; display_buffer[idx2+2] = 0; }
                             }
                             for j in y..(y+h).min(height as usize) {
                                  let idx1 = (j * width as usize + x) * 3;
                                  let idx2 = (j * width as usize + (x+w).min(width as usize-1)) * 3;
                                  if idx1 < display_buffer.len() { display_buffer[idx1] = 0; display_buffer[idx1+1] = 255; display_buffer[idx1+2] = 0; }
                                  if idx2 < display_buffer.len() { display_buffer[idx2] = 0; display_buffer[idx2+1] = 255; display_buffer[idx2+2] = 0; }
                             }
                         }
                    },
                    PipelineOutput::HeadPose(y, p, _r) => {
                         // Simple Center HUD
                         // Apply Gain to match Gaze mode feel
                         let yaw_gain = 1.5;
                         let pitch_gain = 2.5; 
                         
                         // Debug Log (every ~60 frames or just always? Always is too fast.
                         // We'll just print if it changes significantly? Or just rely on visual.)
                         // Console spam is annoying but useful for debugging sensitivity.
                         // Let's print nicely.
                         // print!("\rYaw: {:.2} Pitch: {:.2} Roll: {:.2}   ", y, p, _r);
                         // std::io::Write::flush(&mut std::io::stdout()).ok();

                         let cx = width as f32 / 2.0; let cy = height as f32 / 2.0;
                         let len = 100.0;
                         let tx = cx - (y * yaw_gain).to_radians().sin() * len;
                         let ty = cy - (p * pitch_gain).to_radians().sin() * len;
                         
                         // Draw center
                         for dy in 0..5 { for dx in 0..5 {
                             let idx = ((cy as usize + dy) * width as usize + (cx as usize + dx)) * 3;
                             if idx < display_buffer.len() { display_buffer[idx] = 255; display_buffer[idx+1] = 255; display_buffer[idx+2] = 0; }
                         }}
                         // Draw line
                         for t in 0..100 {
                             let f = t as f32 / 100.0;
                             let lx = cx + (tx - cx) * f;
                             let ly = cy + (ty - cy) * f;
                             let idx = (ly as usize * width as usize + lx as usize) * 3;
                             if idx < display_buffer.len() { display_buffer[idx] = 0; display_buffer[idx+1] = 255; display_buffer[idx+2] = 255; }
                         }
                    },
                    PipelineOutput::Gaze { left_eye, right_eye, yaw, pitch, .. } => {
                         // Draw Vectors from Eye Centers
                         let len = 80.0; // Draw length
                         
                         // We use Yaw/Pitch to determine end points
                         // Note: Ideally Gaze Model output vector is used, but we use head pose as proxy if vector is placeholder.
                         let y_rad = yaw.to_radians();
                         let p_rad = pitch.to_radians();
                         
                         let dx = -y_rad.sin() * len;
                         let dy = -p_rad.sin() * len;
                         
                         for &eye in &[left_eye, right_eye] {
                             let ex = eye.x;
                             let ey = eye.y;
                             let tx = ex + dx;
                             let ty = ey + dy;
                             
                             // Draw Eye Center (Blue)
                             for dy in 0..5 { for dx in 0..5 {
                                 let idx = ((ey as usize + dy) * width as usize + (ex as usize + dx)) * 3;
                                 if idx < display_buffer.len() { display_buffer[idx] = 0; display_buffer[idx+1] = 0; display_buffer[idx+2] = 255; }
                             }}
                             
                             // Draw Ray (Cyan)
                             for t in 0..50 {
                                 let f = t as f32 / 50.0;
                                 let lx = ex + (tx - ex) * f;
                                 let ly = ey + (ty - ey) * f;
                                 let idx = (ly as usize * width as usize + lx as usize) * 3;
                                 if idx < display_buffer.len() { display_buffer[idx] = 0; display_buffer[idx+1] = 255; display_buffer[idx+2] = 255; }
                             }
                         }
                    }
                }
            }

            window.update(&display_buffer)?;
        }
    }

    Ok(())
}
