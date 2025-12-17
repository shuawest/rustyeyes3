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
mod moondream;
mod calibration;

use args::Args;
use camera::CameraSource;
use output::WindowOutput;
use pipeline::Pipeline;
use types::PipelineOutput;
use overlay::OverlayWindow;
use calibration::CalibrationManager;

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
    let mut current_pipeline = create_pipeline(&args.model.unwrap_or_else(|| "pupil_gaze".to_string()));
    println!("Active Pipeline: {}", current_pipeline.name());

    // 3. Setup Output
    // We get the actual format from the camera
    let width = camera.width();
    let height = camera.height();
    let mut window = WindowOutput::new("Rusty Eyes", width as usize, height as usize)?; // Added ?
    println!("Window created successfully.");

    // Setup Calibration Manager
    let mut calibration_manager = CalibrationManager::new("calibration_data")?;
    let mut calibration_mode = false;

    println!("Starting Pipeline...");
    println!("Controls: [0] Combined [1-3] Basic [4] Head Gaze [5] Pupil Gaze [6] Toggle Overlay [9] Calibration Mode");

    // State for Overlay
    let mut show_overlay = true;
    let mut overlay_window: Option<OverlayWindow> = None;
    let screen_w = 1440; // Default Mac, ideally get from OS but minifb doesn't support it easily.
    let screen_h = 900;

    // State for Milestone 1: Moondream
    let mut moondream_oracle: Option<moondream::MoondreamOracle> = None;
    let mut paused_frame: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> = None;
    let mut moondream_result: Option<types::Point3D> = None;
    let mut moondream_active = false;
    
    // Smoothing State
    let mut smooth_x = screen_w as f32 / 2.0;
    let mut smooth_y = screen_h as f32 / 2.0;
    
    // Mouse Interaction State
    let mut mouse_down_prev = false;

    // --- MOONDREAM WORKER SETUP ---
    // Unbounded channel - worker will drain to get latest frame
    let (tx_frame, rx_frame) = std::sync::mpsc::channel::<(image::DynamicImage, Option<(f32, f32)>)>();
    let (tx_result, rx_result) = std::sync::mpsc::channel::<(types::Point3D, Option<(f32, f32)>)>();
    
    std::thread::spawn(move || {
        use moondream::MoondreamOracle;
        use types::Point3D;
        let mut oracle = MoondreamOracle::new().expect("Failed to init Moondream Worker");
        println!("Moondream Worker Started.");
        
        while let Ok(first_frame) = rx_frame.recv() {
            // Got first frame, now drain any backlog to get the LATEST
            let mut latest_frame = first_frame;
            while let Ok(newer_frame) = rx_frame.try_recv() {
                latest_frame = newer_frame; // Keep replacing with newer
            }
            
            // Process only the latest frame
            let (img, onnx_data) = latest_frame;
            if let Ok(gaze) = oracle.gaze_at(&img) {
                let _ = tx_result.send((gaze, onnx_data));
            }
        }
    });

    // --- MAIN LOOP ---
    println!("Starting Pipeline...");
    println!("Controls: [0] Combined [1-3] Basic [4] Head Gaze [5] Pupil Gaze [6] Toggle Overlay");

    // 4. Loop
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        // Swap Pipeline?
        if window.is_key_down(minifb::Key::Key0) {
            current_pipeline = create_pipeline("pupil_gaze");
            show_overlay = true;
            println!("Switched to: Combined (Mesh + Gaze + Overlay)");
        }
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

         // Milestone 1: Moondream Snapshot (Key 7)
         // Milestone 1 (Refined): Continuous Toggle (Key 7)
         if window.is_key_down(minifb::Key::Key7) {
             moondream_active = !moondream_active;
             println!("Moondream Continuous Mode: {}", if moondream_active { "ACTIVE" } else { "INACTIVE" });
             std::thread::sleep(std::time::Duration::from_millis(500)); // Debounce
         }

         // Calibration Mode Toggle (Key 9)
         if window.is_key_down(minifb::Key::Key9) {
             calibration_mode = !calibration_mode;
             println!("Calibration Mode: {}", if calibration_mode { "ON (Press SPACE to capture, 8 to compute)" } else { "OFF" });
             std::thread::sleep(std::time::Duration::from_millis(500));
         }

         
         
         // Capture or Reuse Frame
         let frame = if let Some(frozen) = &paused_frame {
             frozen.clone()
         } else {
             if let Ok(mut cam_frame) = camera.capture() {
                 if args.mirror {
                     image::imageops::flip_horizontal_in_place(&mut cam_frame);
                 }
                 cam_frame
             } else {
                 continue;
             }
         };

         // CALIBRATION CAPTURE
         if calibration_mode {
             // Check for SPACE
             if window.is_key_down(minifb::Key::Space) {
                 if !mouse_down_prev {
                     // Trigger Capture
                     // Get Mouse Pos (Simulator or Real?)
                     // Minifb doesn't give global mouse pos easily for window.
                     // IMPORTANT: We need logic to get cursor pos.
                     // For now, let's just assume center (user looks at center) OR
                     // implement a "Click where looking" via window mouse events if window is fullscreen?
                     // BUT, users look at *screen*, not just window.
                     // For MVP: We will just save a point and maybe ask user to input?
                     // BETTER MVP: We assume the user looks at the MOUSE CURSOR which they move.
                     // getting global mouse from minifb is tricky.
                     // Let's hardcode a sequence?
                     // No, let's just use `window.get_mouse_pos(minifb::MouseMode::Pass)` which gives *relative* to window.
                     
                     if let Some((mx, my)) = window.get_mouse_pos(minifb::MouseMode::Pass) {
                         let _ = calibration_manager.save_data_point(&frame, mx, my);
                     } else {
                         println!("Mouse not in window!");
                     }
                     mouse_down_prev = true;
                 }
             } else {
                 mouse_down_prev = false;
             }
             
             // Compute (Key 8)
             if window.is_key_down(minifb::Key::Key8) {
                 println!("Computing Calibration (Not Implemented in MVP yet, logic is placeholder)");
                 // calibration_manager.compute_regression(...);
                 std::thread::sleep(std::time::Duration::from_millis(500));
             }
         }


         // Process (Only run pipeline if not paused, or run on static frame?)
         // For now, let's run pipeline on the frame (realtime or frozen) so we see the red mesh match.
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
                           
                           
                           let raw_sx = cx - (yaw * gain_x);
                           let raw_sy = cy - (pitch * gain_y);
                           
                           // APPLY CALIBRATION
                           let (cal_sx, cal_sy) = calibration_manager.apply(raw_sx, raw_sy);

                           // Simple Smoothing (LPF)
                           // Initialize static or use window state? 
                           // For now, use a hacky "Option" or just overwrite if jump is too big?
                           // Ideally we need state persistence.
                           // Let's rely on the outer loop state.
                           
                           // We need to store smooth_x, smooth_y in the loop context.
                           // Assuming variables defined above loop:
                           // let mut smooth_x = screen_w as f32 / 2.0;
                           // let mut smooth_y = screen_h as f32 / 2.0;
                           
                           smooth_x = smooth_x * 0.7 + cal_sx * 0.3;
                           smooth_y = smooth_y * 0.7 + cal_sy * 0.3;
                           
                           let sx = smooth_x;
                           let sy = smooth_y;
                           
                           // Send to Sidecar (Gaze)
                           let _ = win.update_gaze(sx, sy);
                           
                           // --- MOONDREAM TRIGGER ---
                           // Throttle sending to avoid queue buildup? 
                           // For sim, we trust worker speed or channel buffer.
                           if moondream_active {
                               let img_buffer = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, frame.to_vec()).unwrap();
                               let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);
                                                              // Calculate current ONNX gaze point (sx, sy) to send for logging
                                let onnx_pt = (sx, sy);
                                // Send frame - worker will drain to get latest
                                let _ = tx_frame.send((dynamic_img, Some(onnx_pt)));
                            }
                                                      // Check for Results (Non-blocking) - drain to get latest only
                            if let Ok(first_result) = rx_result.try_recv() {
                                // Got first result, drain any backlog
                                let mut latest_result = first_result;
                                let mut drain_count = 0;
                                while let Ok(newer_result) = rx_result.try_recv() {
                                    latest_result = newer_result;
                                    drain_count += 1;
                                }
                                if drain_count > 0 {
                                    println!("[WARN] Drained {} stale results", drain_count);
                                }
                                
                                let (md_gaze, onnx_gaze) = latest_result;
                                let md_sx = md_gaze.x * screen_w as f32;
                                let md_sy = md_gaze.y * screen_h as f32;
                                
                                println!("[DATA] Moondream: ({:.2}, {:.2}), ONNX@Capture: {:?}", md_sx, md_sy, onnx_gaze);
                                
                                moondream_result = Some(md_gaze);
                                
                                // Send all updates to Sidecar
                                let _ = win.update_moondream(md_sx, md_sy);
                                
                                // If we have ONNX capture data, send it too
                                if let Some((ox, oy)) = onnx_gaze {
                                    let _ = win.update_captured_onnx(ox, oy);
                                }
                            }
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
                    PipelineOutput::Gaze { left_eye, right_eye, yaw, pitch, landmarks, .. } => {
                         // Draw Mesh if present
                         if let Some(l) = landmarks {
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
                         }

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
            
            // Draw Moondream Gold Gaze
            if let Some(_pt) = moondream_result {
                // Determine screen coordinates (assuming normalized?)
                // Wait, Moondream output needs to be mapped.
                // For now, assume MoondreamOracle returns result.
                // We'll just draw a Gold Star at the result.
                
                // Note: Implement proper mapping later.
            }

            window.update(&display_buffer)?;
        }

    Ok(())
}

