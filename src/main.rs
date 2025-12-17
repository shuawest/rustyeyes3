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
mod font;

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
    println!("Controls: [0] Combined [1-3] Basic [4] Head Gaze [5] Pupil Gaze [6] Toggle Overlay [7] Moondream [9] Calibration");

    // State for Overlay
    let mut show_overlay = true;
    let mut overlay_window: Option<OverlayWindow> = None;
    let screen_w = 1440; // Default Mac, ideally get from OS but minifb doesn't support it easily.
    let screen_h = 900;

    // State for Milestone 1: Moondream
    // let mut moondream_oracle: Option<moondream::MoondreamOracle> = None; // Now in worker thread
    let paused_frame: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> = None; // Still used?
    let mut moondream_result: Option<types::Point3D> = None;
    let mut moondream_active = false;
    
    // Smoothing State
    let mut smooth_x = screen_w as f32 / 2.0;
    let mut smooth_y = screen_h as f32 / 2.0;
    
    // Mouse Interaction State
    let mut mouse_down_prev = false;
    
    // HUD State
    let mut last_calibration_point: Option<(f32, f32, u64)> = None;

    // --- MOONDREAM WORKER SETUP ---
    // Unbounded channel - worker will drain to get latest frame
    // Payload: (Image, OnnxGazeForLogging, CalibrationTimestampID)
    let (tx_frame, rx_frame) = std::sync::mpsc::channel::<(image::DynamicImage, Option<(f32, f32)>, Option<u64>)>();
    // Result: (MoondreamGaze, OnnxGazeForLogging, CalibrationTimestampID)
    let (tx_result, rx_result) = std::sync::mpsc::channel::<(types::Point3D, Option<(f32, f32)>, Option<u64>)>();
    
    std::thread::spawn(move || {
        use moondream::MoondreamOracle;
        use types::Point3D;
        let mut oracle = MoondreamOracle::new().expect("Failed to init Moondream Worker");
        println!("Moondream Worker Started.");
        
        while let Ok(first_frame) = rx_frame.recv() {
            // Got first frame...
            // Logic Change for Calibration: 
            // If we have a backlog, we usually skip to latest. 
            // BUT if there are calibration requests in the queue, we MUST process them?
            // For simplicity, let's keep "Latest Frame" logic for live view, 
            // but if a frame has a timestamp ID, we should prioritize/buffer it?
            // Actually, for "Latest Frame" draining, we might lose a calibration frame if the user spams.
            
            // Revised Logic:
            // Check if queue has any item with Some(id). If so, process that one.
            // If multiple have ID, process all? Or just first?
            // Given Moondream is slow (10s), we can't process a burst.
            // But usually calibration is spaced out.
            
            // Simple approach: Drain, but if we encounter a frame with an ID, we STOP replacing `latest_frame` and use that one?
            // Or just process every frame that has an ID?
            
            // Let's iterate the pending messages.
            let mut priority_frame = if first_frame.2.is_some() { Some(first_frame.clone()) } else { None };
            let mut latest_realtime_frame = first_frame;
            
            while let Ok(newer_frame) = rx_frame.try_recv() {
                if newer_frame.2.is_some() {
                     // We found a calibration frame!
                     // If we already had one, we might be skipping the previous one. 
                     // Ideally we process all calibration frames.
                     // But for now, let's just take the LAST calibration frame if multiple, or simple queue?
                     // Let's just hold onto it.
                     priority_frame = Some(newer_frame.clone()); 
                }
                latest_realtime_frame = newer_frame;
            }
            
            // Decide which to process
            // If we have a priority frame, process it. Otherwise process latest realtime.
            let (img, onnx_data, cal_id) = if let Some(p) = priority_frame {
                p
            } else {
                latest_realtime_frame
            };
            
            // Process
            if let Ok(gaze) = oracle.gaze_at(&img) {
                let _ = tx_result.send((gaze, onnx_data, cal_id));
            }
        }
    });

    // --- MAIN LOOP ---
    println!("Starting Pipeline...");
    println!("Controls: [0] Combined [1-3] Basic [4] Head Gaze [5] Pupil Gaze [6] Toggle Overlay [7] Moondream [9] Calibration");

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
         
         // Process Pipeline FIRST to get inference data for calibration capture
         let landmarks = current_pipeline.process(&frame)?;

                   // CALIBRATION CAPTURE
                   if calibration_mode {
                       // Check for SPACE
                       if window.is_key_down(minifb::Key::Space) {
                           if !mouse_down_prev {
                               if let Some((mx, my)) = window.get_mouse_pos(minifb::MouseMode::Pass) {
                                   // Pass the current landmarks/inference result to save with the image
                                   // IMPORTANT: Get timestamp back to send to Moondream
                                    if let Ok(ts) = calibration_manager.save_data_point(&frame, mx, my, landmarks.clone()) {
                                         // Update HUD State
                                         last_calibration_point = Some((mx, my, ts));
                                         
                                         // Send to Moondream Worker for background processing/correction
                                         let img_buffer = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, frame.to_vec()).unwrap();
                                         let dynamic_img = image::DynamicImage::ImageRgb8(img_buffer);
                                         // We pass None for logging-data, and Some(ts) for ID
                                         let _ = tx_frame.send((dynamic_img, None, Some(ts)));
                                    }
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
                                          // Note: Pass None for calibration ID in live mode
                                          let _ = tx_frame.send((dynamic_img, Some(onnx_pt), None));
                                      }
                                                                // Check for Results (Non-blocking) - drain to get latest only
                                      if let Ok(first_result) = rx_result.try_recv() {
                                          // Got first result, drain any backlog
                                          let mut latest_result = first_result;
                                          let mut drain_count = 0;
                                          
                                          // IMPORTANT: If we encounter a result WITH an ID, we must process it!
                                          // If `latest_result` has an ID, we keep it. 
                                          // If `newer_result` has an ID, we adopt it.
                                          // What if we have multiple IDs? We might lose one.
                                          // For now, let's just drain to latest.
                                          // BUT, if we have a calibration ID, we should process it.
                                          // Let's iterate.
                                          
                                          // Handle First
                                          if let Some(ts) = latest_result.2 {
                                                let _ = calibration_manager.update_point_with_moondream(ts, latest_result.0);
                                          }

                                          while let Ok(newer_result) = rx_result.try_recv() {
                                              if let Some(ts) = newer_result.2 {
                                                   let _ = calibration_manager.update_point_with_moondream(ts, newer_result.0);
                                              }
                                              latest_result = newer_result;
                                              drain_count += 1;
                                          }
                                          if drain_count > 0 {
                                              // println!("[WARN] Drained {} stale results", drain_count);
                                          }
                                          
                                          let (md_gaze, onnx_gaze, _cal_id) = latest_result;
                                          let md_sx = md_gaze.x * screen_w as f32;
                                          let md_sy = md_gaze.y * screen_h as f32;
                                          
                                          if onnx_gaze.is_some() {
                                              println!("[DATA] Moondream: ({:.2}, {:.2}), ONNX@Capture: {:?}", md_sx, md_sy, onnx_gaze);
                                          } else {
                                              // Calibration result likely
                                              println!("[CAL] Processed Moondream for timestamp {:?}", _cal_id);
                                          }
                                          
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
            
            // Draw Calibration HUD (Mouse Position + Timestamp)
            if calibration_mode {
                if let Some((lx, ly, ts)) = last_calibration_point {
                    use chrono::TimeZone;
                    let dt = chrono::Local.timestamp_millis_opt(ts as i64).unwrap();
                    let time_str = dt.format("%H:%M:%S").to_string();
                    let hud_text = format!("LAST: ({:.0},{:.0}) AT {}", lx, ly, time_str);
                    
                    // Draw green crosshair at last point
                    let cx = lx as usize;
                    let cy = ly as usize;
                    let size = 10;
                    if cx < width as usize && cy < height as usize {
                        // Horizontal
                        for i in (cx.saturating_sub(size))..((cx+size).min(width as usize)) {
                             let idx = (cy * width as usize + i) * 3;
                             if idx < display_buffer.len() { display_buffer[idx] = 0; display_buffer[idx+1] = 255; display_buffer[idx+2] = 0; }
                        }
                        // Vertical
                        for j in (cy.saturating_sub(size))..((cy+size).min(height as usize)) {
                             let idx = (j * width as usize + cx) * 3;
                             if idx < display_buffer.len() { display_buffer[idx] = 0; display_buffer[idx+1] = 255; display_buffer[idx+2] = 0; }
                        }
                    }
                    
                    // Draw Text
                    font::draw_text_line(&mut display_buffer, width as usize, height as usize, 10, height as usize - 20, &hud_text, (0, 255, 0));
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

