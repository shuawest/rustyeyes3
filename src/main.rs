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
mod config;
mod ttf;

use args::Args;
use camera::CameraSource;
use output::WindowOutput;
use pipeline::Pipeline;
use types::PipelineOutput;
use overlay::OverlayWindow;
use calibration::CalibrationManager;
use config::AppConfig;
use ttf::FontRenderer;

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

    // 0. Load Config
    let config = AppConfig::load()?;

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
    // let mut show_overlay = true; // Moved to config above
    let mut overlay_window: Option<OverlayWindow> = None;
    let screen_w = 1440; // Default Mac, ideally get from OS but minifb doesn't support it easily.
    let screen_h = 900;

    // State for Milestone 1: Moondream
    // let mut moondream_oracle: Option<moondream::MoondreamOracle> = None; // Now in worker thread
    let paused_frame: Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> = None; // Still used?
    let mut moondream_result: Option<types::Point3D> = None;
    let mut captured_gaze_result: Option<(f32, f32)> = None; // For "Green Dot" comparison
    let mut moondream_active = false;
    
    // Smoothing State
    let mut smooth_x = screen_w as f32 / 2.0;
    let mut smooth_y = screen_h as f32 / 2.0;
    
    // Mouse Interaction State
    let mut mouse_down_prev = false;
    
    // HUD State
    let mut last_calibration_point: Option<(f32, f32, u64)> = None;

    // --- MOONDREAM WORKER SETUP ---
    // Bounded Channel (1) to prevent memory explosion if worker is slow
    // Payload: (Image, OnnxGazeForLogging, CalibrationTimestampID)
    let (tx_frame, rx_frame) = std::sync::mpsc::sync_channel::<(image::DynamicImage, Option<(f32, f32)>, Option<u64>)>(1);
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

    // 4. Loop
    let mut last_pipeline_output: Option<PipelineOutput> = None;
    
    // Feature Toggles (Loaded from Config)
    let mut show_mesh = config.defaults.show_mesh;
    let mut show_pose = config.defaults.show_pose;
    let mut show_gaze = config.defaults.show_gaze;
    let mut mirror_mode = config.defaults.mirror_mode;
    let mut show_overlay = config.defaults.show_overlay;
    
    // We keep one robust pipeline active
    // We keep one robust pipeline active
    let mut pipeline = create_pipeline("pupil_gaze"); 
    
    // Initialize Font Renderer (Try to load custom, else None -> Fallback to bitmap)
    let font_renderer = FontRenderer::try_load(&config.ui.font_family);
    
    while window.is_open() && !window.is_key_down(minifb::Key::Escape) {
        
        // --- INPUT HANDLING ---
        // (Moved to dedicated block below for clean toggling)

        // Capture Frame (Unified)
        let mut latest_realtime_frame = if let Ok(mut cam_frame) = camera.capture() {
             if mirror_mode {
                 image::imageops::flip_horizontal_in_place(&mut cam_frame);
             }
             cam_frame
        } else {
             continue;
        };
        
        let mut display_buffer = latest_realtime_frame.to_vec();
        let (width, height) = latest_realtime_frame.dimensions();
        
        // --- PROCESSING (Moved Up for Sync) ---
        // Always run the full pipeline FIRST so we have the Gaze for this exact frame
        let output = pipeline.process(&latest_realtime_frame)?;
        last_pipeline_output = output.clone();

        // --- MOONDREAM DISPATCH ---
        if moondream_active {
             // Clone frame? This might be heavy. dynamic_image from cam_frame?
             // cam_frame is RgbBuffer.
             let img = image::DynamicImage::ImageRgb8(latest_realtime_frame.clone());
             
             // Extract current Gaze for comparison (Now fully synchronized)
             let current_gaze_coords = if let Some(out) = &output {
                  if let PipelineOutput::Gaze { yaw, pitch, .. } = out {
                       let eff_yaw = if mirror_mode { -yaw } else { *yaw };
                       // Precise calculation matching drawing logic
                       let mut sx = width as f32 / 2.0; 
                       let mut sy = height as f32 / 2.0;
                       sx += eff_yaw * 20.0;
                       sy -= pitch * 20.0;
                       Some((sx, sy))
                  } else { None }
             } else { None };

             // We use try_send on a bounded channel (1). 
             // If worker is busy, this fails immediately and we skip sending the frame.
             let _ = tx_frame.try_send((img, current_gaze_coords, None));
        }

        // --- MOONDREAM ASYNC UPDATE ---
        // Iterate through all available results from the worker
        while let Ok((pt, onnx_data, _cal_id)) = rx_result.try_recv() {
             moondream_result = Some(pt);
             if let Some(capt) = onnx_data {
                 captured_gaze_result = Some(capt);
             }
        }
        // --- INPUT HANDLING ---
        for key in window.get_keys_pressed(minifb::KeyRepeat::No) {
            match key {
                minifb::Key::Escape => {
                     // We can't break outer loop easily, but we can rely on loop check
                     // or window.is_key_down(Escape) works but we want single press for toggles?
                     // Actually, if Escape is pressed, we just return/break.
                     return Ok(());
                },
                minifb::Key::Key1 => show_mesh = !show_mesh,
                minifb::Key::Key2 => show_pose = !show_pose,
                minifb::Key::Key3 => show_gaze = !show_gaze,
                
                minifb::Key::Key5 => mirror_mode = !mirror_mode,

                minifb::Key::Key6 => show_overlay = !show_overlay,
                minifb::Key::Key7 => {
                    if moondream_active {
                        moondream_active = false;
                        moondream_result = None;
                    } else {
                        moondream_active = true;
                    }
                },
                minifb::Key::Key9 => {
                    calibration_mode = !calibration_mode;
                    if calibration_mode {
                        println!("Calibration Mode: ON");
                    } else {
                        println!("Calibration Mode: OFF");
                    }
                },
                _ => {}
            }
        }

        // --- PROCESSING ---
        // Always run the full pipeline
        let output = pipeline.process(&latest_realtime_frame)?;
        last_pipeline_output = output.clone();

        // --- DRAWING ---
        // display_buffer is already init with frame pixels above
        
        if let Some(out) = output {
            match out {
                    PipelineOutput::Gaze { left_eye, right_eye, yaw, pitch, roll: _ , vector: _, landmarks } => {
                       
                        // Correct for Mirror Mode
                        let yaw = if mirror_mode { -yaw } else { yaw };

                        // 1. Face Mesh
                        if show_mesh {
                            if let Some(l) = &landmarks {
                                let (mr, mg, mb) = parse_hex(&config.ui.mesh_color_hex);
                                let dot_size = config.ui.mesh_dot_size;
                                
                                for p in &l.points {
                                    let x = p.x as usize;
                                    let y = p.y as usize;
                                    if x < width as usize && y < height as usize {
                                         // Draw Configurable Dot
                                         for dy in 0..dot_size {
                                             for dx in 0..dot_size {
                                                 let idx = ((y + dy) * width as usize + (x + dx)) * 3;
                                                 if idx + 2 < display_buffer.len() {
                                                     display_buffer[idx] = mr;
                                                     display_buffer[idx+1] = mg;
                                                     display_buffer[idx+2] = mb;
                                                 }
                                             }
                                         }
                                    }
                                }
                            }
                        }

                        // 2. Head Pose (Green Lines)
                        if show_pose {
                            let cx = width as f32 / 2.0;
                            let cy = height as f32 / 2.0;
                            
                            let len = config.defaults.head_pose_length;
                            let end_x = cx + (yaw.to_radians().sin() * len);
                            let end_y = cy - (pitch.to_radians().sin() * len);
                            
                             let mut t = 0.0;
                             while t < 1.0 {
                                 let px = cx + (end_x - cx) * t;
                                 let py = cy + (end_y - cy) * t;
                                 let idx = (py as usize * width as usize + px as usize) * 3;
                                 if idx + 2 < display_buffer.len() {
                                      display_buffer[idx] = 0;
                                      display_buffer[idx+1] = 255;
                                      display_buffer[idx+2] = 0;
                                 }
                                 t += 0.005; // Finer grain for longer lines
                             }
                        }
                        
                         // 3. Gaze (Blue Ray) - Extend from Pupils
                        if show_gaze {
                            let len = config.defaults.head_pose_length * 1.5;
                            
                            // Define a closure to draw a ray
                            let mut draw_ray = |start_x: f32, start_y: f32| {
                                let end_x = start_x + (yaw.to_radians().sin() * len);
                                let end_y = start_y - (pitch.to_radians().sin() * len);
                                
                                let mut t = 0.0;
                                while t < 1.0 {
                                    let px = start_x + (end_x - start_x) * t;
                                    let py = start_y + (end_y - start_y) * t;
                                    let idx = (py as usize * width as usize + px as usize) * 3;
                                    if idx + 2 < display_buffer.len() {
                                         display_buffer[idx] = 0;
                                         display_buffer[idx+1] = 255;
                                         display_buffer[idx+2] = 255; // Cyan
                                    }
                                    t += 0.005;
                                }
                            };

                            // Draw from both eyes
                            draw_ray(left_eye.x, left_eye.y);
                            draw_ray(right_eye.x, right_eye.y);
                        }
                        
                        // Send data to overlay if enabled
                        if show_overlay {
                            // Ensure overlay is open
                            if overlay_window.is_none() {
                                if let Ok(mut win) = OverlayWindow::new(screen_w, screen_h) {
                                    // Init font
                                    let _ = win.update_font(&config.ui.font_family, config.ui.font_size_pt);
                                    overlay_window = Some(win);
                                }
                            }
                            
                            // Map coordinates to screen
                            let mut screen_x = width as f32 / 2.0; 
                            let mut screen_y = height as f32 / 2.0;
                            
                            // Naive mapping: Just center + angle * factor
                            let x_factor = 20.0;
                            let y_factor = 20.0;
                            
                            screen_x += yaw * x_factor;
                            screen_y -= pitch * y_factor;
                            
                            if let Some(win) = overlay_window.as_mut() {
                                let _ = win.update_gaze(screen_x, screen_y);
                                
                                // Send Captured Gaze if available
                                if let Some((cx, cy)) = captured_gaze_result {
                                    let _ = win.update_captured_onnx(cx, cy);
                                }
                                
                                // Send Moondream Result if available (Fix for blank overlay)
                                if let Some(pt) = moondream_result {
                                     // Moondream is normalized 0..1. Map to screen.
                                     let mx = pt.x * width as f32;
                                     let my = pt.y * height as f32;
                                     let _ = win.update_moondream(mx, my);
                                }
                                
                                // Send Config Upates occasionally? 
                                // Actually, we should send font config on init or change.
                                // For now, let's just send it every frame? No, that's spammy.
                                // We'll assume the user hasn't changed it since startup for this simple implementation.
                                // Or we can send it once if a flag is set?
                                // Let's simplify: Send it every 60 frames?
                                // Or just send it here blindly, pipes are fast.
                                // Optimization: Only if changed.
                                // Implementation: sending every time is safe for pipe but wasteful.
                                // Let's send it ONCE on init. (Doing that in setup block below).
                            }
                        } else {
                            if overlay_window.is_some() {
                                overlay_window = None; // Close it
                            }
                        }
                    },
                    _ => {} // Other variants not expected from PupilGazePipeline
                }
            }


            // --- MOONDREAM TARGET ---
            if let Some(pt) = moondream_result {
                // If active or we have a sticky result?
                // Let's draw if we have a result.
                let mx = (pt.x * width as f32) as i32;
                let my = (pt.y * height as f32) as i32;
                
                // Draw Gold Crosshair (Target)
                let size = 20;
                let thickness = 2;
                let color = (255, 215, 0); // Gold
                
                for i in -size..=size {
                    for t in -thickness..=thickness {
                        // Horizontal
                        let px = mx + i;
                        let py = my + t;
                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                             let idx = (py as usize * width as usize + px as usize) * 3;
                             if idx + 2 < display_buffer.len() {
                                 display_buffer[idx] = color.0;
                                 display_buffer[idx+1] = color.1;
                                 display_buffer[idx+2] = color.2;
                             }
                        }
                        
                        // Vertical
                        let px = mx + t;
                        let py = my + i;
                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                             let idx = (py as usize * width as usize + px as usize) * 3;
                             if idx + 2 < display_buffer.len() {
                                 display_buffer[idx] = color.0;
                                 display_buffer[idx+1] = color.1;
                                 display_buffer[idx+2] = color.2;
                             }
                        }
                    }
                }
            }
            
            // --- CAPTURED GAZE (Green Dot) ---
            if moondream_active {
               if let Some((cx, cy)) = captured_gaze_result {
                   let mx = cx as i32;
                   let my = cy as i32;
                   let radius = 10;
                   let color = (0, 255, 0); // Green
                   
                   for dy in -radius..=radius {
                       for dx in -radius..=radius {
                           if dx*dx + dy*dy <= radius*radius {
                               let px = mx + dx;
                               let py = my + dy;
                               if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                                   let idx = (py as usize * width as usize + px as usize) * 3;
                                   if idx + 2 < display_buffer.len() {
                                       display_buffer[idx] = color.0;
                                       display_buffer[idx+1] = color.1;
                                       display_buffer[idx+2] = color.2;
                                   }
                               }
                           }
                       }
                   }
               }
            }

            // --- VISUAL MENU ---
            // Updated Toggle-based Menu
            let menu_items = [
                ("1", "Face Mesh", show_mesh),
                ("2", "Head Pose", show_pose),
                ("3", "Eye Gaze", show_gaze),
                ("5", "Mirror", mirror_mode),
                ("6", "Overlay", show_overlay),
                ("7", "Moondream", moondream_active),
                ("9", "Calibration", calibration_mode),
            ];
            
            let mut y_start = height as usize / 2 - 150;
            let menu_scale = config.ui.menu_scale;
            let line_height = 12 * menu_scale;
            
            // Draw Toggles
            // Helper to draw text with fallback
            let draw_text = |buf: &mut [u8], w: usize, h: usize, x: usize, y: usize, txt: &str, col: (u8, u8, u8)| {
                 if let Some(fr) = &font_renderer {
                     fr.draw_text(buf, w, h, x, y, txt, col, config.ui.font_size_pt as f32);
                 } else {
                     font::draw_text_line(buf, w, h, x, y, txt, col, menu_scale);
                 }
            };
            
            // Calc line height based on renderer
            let line_height = if let Some(fr) = &font_renderer {
                fr.measure_height(config.ui.font_size_pt as f32) + 5
            } else {
                12 * menu_scale
            };

            for (key, label, active) in menu_items.iter() {
                let color = if *active { (0, 255, 0) } else { (255, 255, 255) };
                let status = if *active { "ON" } else { "OFF" };
                let text = format!("[{}] {} [{}]", key, label, status);
                draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, color);
                y_start += line_height;
            }
            
            y_start += line_height; // Spacer
            
            // Draw System Toggles
            let system_toggles = [
                ("6", "Overlay", show_overlay),
                ("7", "Moondream", moondream_active),
                ("9", "Calibration", calibration_mode),
            ];
            
            for (key, label, active) in system_toggles.iter() {
                let color = if *active { (0, 255, 0) } else { (255, 255, 255) };
                let status = if *active { "ON" } else { "OFF" };
                let text = format!("[{}] {} [{}]", key, label, status);
                 draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, color);
                y_start += line_height;
            }

            y_start += line_height; // Spacer
            
            // --- HUD STATS ---
            // Draw Gaze / Moondream Coordinates
            if let Some(out) = &last_pipeline_output {
               if let PipelineOutput::Gaze { yaw, pitch, .. } = out {
                   // Raw Angles
                   let text = format!("Gaze: {:.1}, {:.1}", yaw, pitch);
                   draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, (200, 200, 200));
                   y_start += line_height;

                   // Screen Coordinates (Recalculate to match Overlay)
                   let eff_yaw = if mirror_mode { -yaw } else { *yaw };
                   let mut sx = width as f32 / 2.0; 
                   let mut sy = height as f32 / 2.0;
                   let x_factor = 20.0;
                   let y_factor = 20.0;
                   sx += eff_yaw * x_factor;
                   sy -= pitch * y_factor;

                   let text_screen = format!("Screen: {:.0}, {:.0}", sx, sy);
                   draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text_screen, (0, 255, 255)); // Cyan
                   y_start += line_height;
               }
            }
            
            if moondream_active {
                if let Some((cx, cy)) = captured_gaze_result {
                     let text = format!("Captured: {:.0}, {:.0}", cx, cy);
                     draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, (0, 255, 0));
                     y_start += line_height;
                } else {
                     let text = "Captured: ----, ----";
                     draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, (100, 100, 100));
                     y_start += line_height;
                }

                if let Some(pt) = moondream_result {
                    let text = format!("Moon: ({:.2}, {:.2})", pt.x, pt.y);
                    draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, (255, 215, 0));
                    y_start += line_height;
                } else {
                    let text = "Moon: Waiting...";
                     draw_text(&mut display_buffer, width as usize, height as usize, 10, y_start, &text, (255, 255, 255));
                     y_start += line_height;
                }
            }

            // --- WINDOW UPDATE ---
            // CRITICAL: Must be called to show the frame!
            window.update(&display_buffer)?;
        }

    Ok(())
}

fn parse_hex(hex: &str) -> (u8, u8, u8) {
    if hex.len() == 7 && hex.starts_with('#') {
        let r = u8::from_str_radix(&hex[1..3], 16).unwrap_or(255);
        let g = u8::from_str_radix(&hex[3..5], 16).unwrap_or(0);
        let b = u8::from_str_radix(&hex[5..7], 16).unwrap_or(0);
        (r, g, b)
    } else {
        (255, 0, 0) // Default Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_hex() {
        assert_eq!(parse_hex("#FF0000"), (255, 0, 0));
        assert_eq!(parse_hex("#00FF00"), (0, 255, 0));
        assert_eq!(parse_hex("#0000FF"), (0, 0, 255));
        assert_eq!(parse_hex("#FFFFFF"), (255, 255, 255));
        assert_eq!(parse_hex("invalid"), (255, 0, 0)); // Fallback
    }
}
