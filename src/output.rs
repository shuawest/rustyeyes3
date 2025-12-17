use image::{ImageBuffer, Rgb};
use anyhow::Result;

/// Trait for handling the output of the pipeline
use crate::types::Landmarks;


pub struct WindowOutput {
    window: minifb::Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
}

impl WindowOutput {
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self> {
        let mut window = minifb::Window::new(
            title,
            width,
            height,
            minifb::WindowOptions {
                resize: true,
                ..minifb::WindowOptions::default()
            },
        ).map_err(|e| anyhow::anyhow!("Failed to create window: {}", e))?;

        window.limit_update_rate(Some(std::time::Duration::from_micros(16600))); // ~60 FPS

        Ok(Self {
            window,
            buffer: vec![0; width * height],
            width,
            height,
        })
    }
    
    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn is_key_down(&self, key: minifb::Key) -> bool {
        self.window.is_key_down(key)
    }

    pub fn get_mouse_pos(&self, mode: minifb::MouseMode) -> Option<(f32, f32)> {
        self.window.get_mouse_pos(mode)
    }

    pub fn update(&mut self, buffer: &[u8]) -> Result<()> {
        // buffer is RGB8, need to convert to u32 ARGB
        if self.buffer.len() != self.width * self.height {
             self.buffer.resize(self.width * self.height, 0);
        }
        
        for (i, chunk) in buffer.chunks(3).enumerate() {
            if i >= self.buffer.len() { break; }
            let r = chunk[0] as u32;
            let g = chunk[1] as u32;
            let b = chunk[2] as u32;
            self.buffer[i] = (r << 16) | (g << 8) | b;
        }
        
        self.window.update_with_buffer(&self.buffer, self.width, self.height)
            .map_err(|e| anyhow::anyhow!(e))
    }

    #[allow(dead_code)]
    fn draw_point(&mut self, x: usize, y: usize, color: u32) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            if idx < self.buffer.len() {
                self.buffer[idx] = color;
            }
        }
    }

    #[allow(dead_code)]
    pub fn handle_frame(&mut self, frame: &ImageBuffer<Rgb<u8>, Vec<u8>>, landmarks: Option<&Landmarks>) -> Result<()> {
        // ... (resize checks)
        // Resize buffer if needed
        if self.buffer.len() != self.width * self.height {
             self.buffer.resize(self.width * self.height, 0);
        }
        
        let target_w = frame.width() as usize;
        let target_h = frame.height() as usize;

        if target_w != self.width || target_h != self.height {
             self.width = target_w;
             self.height = target_h;
             self.buffer.resize(self.width * self.height, 0);
        }

        // Copy frame to buffer
        for (i, pixel) in frame.pixels().enumerate() {
            if i >= self.buffer.len() { break; }
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            self.buffer[i] = (r << 16) | (g << 8) | b;
        }

        // Draw Landmarks
        if let Some(lm) = landmarks {
            let color = 0x00FF0000; // Red
            for point in &lm.points {
                 let px = point.x as usize;
                 let py = point.y as usize;
                 // Draw 3x3 point
                 for dx in 0..3 {
                     for dy in 0..3 {
                         self.draw_point(px + dx, py + dy, color);
                     }
                 }
            }
        }

        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)
            .map_err(|e| anyhow::anyhow!("Window update failed: {}", e))?;

        Ok(())
    }
}
