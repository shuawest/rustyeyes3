use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::window::{Window, WindowId, WindowAttributes, WindowLevel};
use winit::keyboard::{Key, NamedKey};
use winit::dpi::LogicalSize;
use softbuffer::{Context, Surface};
use std::sync::Arc;
use std::thread;
use std::io::{self, BufRead};
use std::num::NonZeroU32;
use rusttype::{Font, Scale, Point, PositionedGlyph};

// Custom Events from blocking stdin thread
#[derive(Debug)]
enum UserEvent {
    UpdateGaze(f32, f32),
    UpdateMoondream(f32, f32),
    UpdateVerified(f32, f32),
    UpdatePending(f32, f32),
    UpdateMenu(String),
    UpdateFont(String, u32),
    Quit,
}

struct AppState {
    window: Option<Arc<Window>>,
    surface: Option<Surface<Arc<Window>, Arc<Window>>>,
    context: Option<Context<Arc<Window>>>,
    
    // Data State
    gaze_pos: Option<(f32, f32)>,
    moondream_pos: Option<(f32, f32)>,
    verified_pos: Option<(f32, f32)>,
    pending_pos: Option<(f32, f32)>,
    menu_text: String,
    
    // Font State
    font: Option<Font<'static>>,
    font_size: f32,
    
    // Proxy to wake up event loop
    proxy: EventLoopProxy<UserEvent>,
}

impl AppState {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        // Load default font (embedded or system fallback)
        // let font_data = include_bytes!("../../models/font.ttf"); 
        // We don't have this packaged, fallback to simple drawing if needed? 
        
        Self {
            window: None,
            surface: None,
            context: None,
            gaze_pos: None,
            moondream_pos: None,
            verified_pos: None,
            pending_pos: None,
            menu_text: String::new(),
            font: None,
            font_size: 24.0,
            proxy,
        }
    }
    
    fn redraw(&mut self) {
        if let (Some(window), Some(surface)) = (&self.window, &mut self.surface) {
            let (width, height) = {
                let size = window.inner_size();
                (size.width, size.height)
            };
            
            if width == 0 || height == 0 { return; }

            surface.resize(
                NonZeroU32::new(width).unwrap(),
                NonZeroU32::new(height).unwrap(),
            ).unwrap();

            let mut buffer = surface.buffer_mut().unwrap();

            // 1. Clear to Transparent (0x00000000)
            buffer.fill(0);

            // Helper to draw circle
            let mut draw_circle = |cx: f32, cy: f32, r: f32, color: u32| {
                let r_sq = r * r;
                let min_x = (cx - r) as i32;
                let max_x = (cx + r) as i32;
                let min_y = (cy - r) as i32;
                let max_y = (cy + r) as i32;
                
                for py in min_y..=max_y {
                     for px in min_x..=max_x {
                         if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                             let dx = px as f32 - cx;
                             let dy = py as f32 - cy;
                             if dx*dx + dy*dy <= r_sq {
                                  let idx = py as usize * width as usize + px as usize;
                                  buffer[idx] = color;
                             }
                         }
                     }
                }
            };

            // 2. Draw Gaze (Blue Dot)
            if let Some((gx, gy)) = self.gaze_pos {
                // Blue: 0x0000FFFF (Alpha, Red, Green, Blue)? No softbuffer is usually XRGB or ARGB.
                // Assuming ARGB. Solid Blue.
                draw_circle(gx, gy, 15.0, 0xFF0000FF);
                
                // Red center
                draw_circle(gx, gy, 5.0, 0xFFFF0000);
            }
            
            // 3. Draw Moondream (Cyan)
            if let Some((mx, my)) = self.moondream_pos {
                 draw_circle(mx, my, 12.0, 0xFF00FFFF);
                 draw_circle(mx, my, 4.0, 0xFFFFFF00); // Yellow center
            }
            
             // 4. Draw Verified (Green/Yellow)
            if let Some((vx, vy)) = self.verified_pos {
                 draw_circle(vx, vy, 10.0, 0xFF00FF00); // Green
                 draw_circle(vx, vy, 3.0, 0xFFFFFF00); // Yellow
            }
            
             // 5. Draw Pending (Green/Red)
            if let Some((px, py)) = self.pending_pos {
                 draw_circle(px, py, 10.0, 0xFF00FF00); // Green
                 draw_circle(px, py, 3.0, 0xFFFF0000); // Red
            }

            // 6. Draw Menu Text
            // Simple bitmap font fallback if no ttf, or just rely on console? 
            // Implementing a full font renderer is heavy. 
            // We can draw a simple "box" for text present.
            // OR use a minimal embedded font.
            // For now, let's just log it? No, needs to be visible.
            // Let's implement basic "Text" using block characters? 
            // A challenge.
            
            // Let's try to load a system font or just skip text for MVP?
            // "The user should be wowed". Text is needed.
            // We can use `rusttype`. We need a font file.
            // Let's assume user has /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf or similar?
            if let Some(font) = &self.font {
                 // Draw text
                 let scale = Scale::uniform(self.font_size);
                 let _v_metrics = font.v_metrics(scale);
                 
                 let lines: Vec<&str> = self.menu_text.split('|').collect();
                 let start_y = height as f32 - (lines.len() as f32 * self.font_size * 1.5) - 20.0;
                 let start_x = 20.0;
                 
                 for (i, line) in lines.iter().enumerate() {
                      let y = start_y + (i as f32 * self.font_size * 1.5);
                      
                      let glyphs: Vec<PositionedGlyph> = font.layout(line, scale, Point { x: start_x, y }).collect();
                      
                      for glyph in glyphs {
                          if let Some(bb) = glyph.pixel_bounding_box() {
                              glyph.draw(|x, y, v| {
                                   // v is alpha 0..1
                                   if v > 0.5 {
                                        let px = x as i32 + bb.min.x;
                                        let py = y as i32 + bb.min.y;
                                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                                             let idx = py as usize * width as usize + px as usize;
                                             buffer[idx] = 0xFFFFFFFF; // White
                                        }
                                   }
                              });
                          }
                      }
                 }
            }

            buffer.present().unwrap();
        }
    }
}

struct App {
    state: AppState,
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.window.is_some() {
            return;
        }

        let win_attr = WindowAttributes::default()
            .with_title("Rusty Eyes Overlay")
            .with_transparent(true)
            .with_decorations(false)
            .with_window_level(WindowLevel::AlwaysOnTop)
            .with_fullscreen(None) // We want maximized or custom size?
            // On X11, transparent often requires a compositor.
            // We'll set maximized to cover screen? 
            // Or just very large?
            // Let's use 1920x1080 default or what was passed?
            // The protocol doesn't send size on init easily, main passes args?
            // main uses OverlayWindow::new(screen_w, screen_h).
            // But we are a binary. We can take args?
            .with_inner_size(LogicalSize::new(1440.0, 900.0)); // Default fallback

        // Note: passthrough (click-through) is platform specific.
        // winit 0.29+ has `.with_cursor_hittest(false)`?
        // Checked docs: yes, `with_cursor_passthrough` or similar depending on platform traits.
        // But winit::window::WindowAttributes doesn't have it directly universal?
        // It's in `WindowBuilderExtX11`?
        
        let window = event_loop.create_window(win_attr).unwrap();
        
        // Window is created. Try to set passthrough.
        // window.set_cursor_hittest(false); // New api
        let _ = window.set_cursor_hittest(false);
        // let _ = window.set_ignore_cursor_events(true); // Deprecated/Removed in 0.30
        
        let window = Arc::new(window);
    
        // Setup Softbuffer
        // Context::new needs a reference to window which implements HasDisplayHandle
        // Arc<Window> implements it? Yes.
        // But softbuffer 0.4 changed signature.
        let context = Context::new(window.clone()).unwrap();
        let surface = Surface::new(&context, window.clone()).unwrap();

        self.state.window = Some(window.clone());
        // self.state.context = Some(context); // Softbuffer context is not easily stored? 
        // Actually it is.
        self.state.context = Some(context);
        self.state.surface = Some(surface);
        
        // Try to load a default font from typical Linux paths
        let paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
        ];
        
        for p in paths {
             if let Ok(data) = std::fs::read(p) {
                 if let Some(f) = Font::try_from_vec(data) {
                      self.state.font = Some(f);
                      break;
                 }
             }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.state.redraw();
            }
            _ => (),
        }
    }
    
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
             UserEvent::UpdateGaze(x, y) => self.state.gaze_pos = Some((x, y)),
             UserEvent::UpdateMoondream(x, y) => self.state.moondream_pos = Some((x, y)),
             UserEvent::UpdateVerified(x, y) => self.state.verified_pos = Some((x, y)),
             UserEvent::UpdatePending(x, y) => self.state.pending_pos = Some((x, y)),
             UserEvent::UpdateMenu(s) => self.state.menu_text = s,
             UserEvent::UpdateFont(_f, s) => self.state.font_size = s as f32, // Ignore family for now
             UserEvent::Quit => std::process::exit(0),
        }
        self.state.window.as_ref().unwrap().request_redraw();
    }
}

fn main() -> io::Result<()> {
    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();
    
    // Stdin Thread
    let proxy_clone = proxy.clone();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(l) = line {
                let parts: Vec<&str> = l.split_whitespace().collect();
                if parts.is_empty() { continue; }
                
                match parts[0] {
                    "G" => {
                        if parts.len() >= 3 {
                             let x = parts[1].parse().unwrap_or(0.0);
                             let y = parts[2].parse().unwrap_or(0.0);
                             let _ = proxy_clone.send_event(UserEvent::UpdateGaze(x, y));
                        }
                    },
                    "M" => {
                        if parts.len() >= 3 {
                             let x = parts[1].parse().unwrap_or(0.0);
                             let y = parts[2].parse().unwrap_or(0.0);
                             let _ = proxy_clone.send_event(UserEvent::UpdateMoondream(x, y));
                        }
                    },
                    "C" => { // Shared for Completed/Verified
                        if parts.len() >= 3 {
                             let x = parts[1].parse().unwrap_or(0.0);
                             let y = parts[2].parse().unwrap_or(0.0);
                             let _ = proxy_clone.send_event(UserEvent::UpdateVerified(x, y));
                        }
                    },
                     "P" => {
                        if parts.len() >= 3 {
                             let x = parts[1].parse().unwrap_or(0.0);
                             let y = parts[2].parse().unwrap_or(0.0);
                             let _ = proxy_clone.send_event(UserEvent::UpdatePending(x, y));
                        }
                    },
                    "S" => {
                        // Rest of line
                        if l.len() > 2 {
                             let text = l[2..].to_string();
                             let _ = proxy_clone.send_event(UserEvent::UpdateMenu(text));
                        }
                    },
                    "Q" => {
                         let _ = proxy_clone.send_event(UserEvent::Quit);
                         break;
                    },
                    _ => {}
                }
            } else {
                break; // EOF
            }
        }
        let _ = proxy_clone.send_event(UserEvent::Quit);
    });

    let mut app = App {
        state: AppState::new(proxy),
    };
    
    event_loop.run_app(&mut app).map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}
