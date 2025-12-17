use rusttype::{Font, Scale, point, PositionedGlyph};
use std::fs;
use std::path::Path;

pub struct FontRenderer {
    font: Font<'static>,
}

impl FontRenderer {
    pub fn try_load(family: &str) -> Option<Self> {
        // Naive search for macOS/Linux common paths
        // Family is like "Calibri", "Arial"
        // Filenames might be "Calibri.ttf", "Arial.ttf"
        
        let paths = [
            format!("/Library/Fonts/{}.ttf", family),
            format!("/System/Library/Fonts/{}.ttf", family),
            format!("/System/Library/Fonts/Supplemental/{}.ttf", family),
            format!("/usr/share/fonts/truetype/{}.ttf", family),
             // Check local dir
            format!("{}.ttf", family),
        ];

        for p in paths.iter() {
            if Path::new(p).exists() {
                 if let Ok(data) = fs::read(p) {
                     if let Some(font) = Font::try_from_vec(data) {
                         println!("Loaded font from {}", p);
                         return Some(Self { font });
                     }
                 }
            }
        }
        
        println!("Could not find font family '{}'. Falling back to bitmap.", family);
        None
    }

    pub fn draw_text(&self, buffer: &mut [u8], width: usize, height: usize, x: usize, y: usize, text: &str, color: (u8, u8, u8), size_pt: f32) {
        let scale = Scale::uniform(size_pt);
        let v_metrics = self.font.v_metrics(scale);
        
        let start_point = point(x as f32, y as f32 + v_metrics.ascent);

        for glyph in self.font.layout(text, scale, start_point) {
            if let Some(bb) = glyph.pixel_bounding_box() {
                glyph.draw(|gx, gy, v| {
                    let alpha = (v * 255.0) as u8;
                    if alpha > 50 { // Threshold for visibility
                        let px = (bb.min.x + gx as i32) as usize;
                        let py = (bb.min.y + gy as i32) as usize;

                        if px < width && py < height {
                            let idx = (py * width + px) * 3;
                            if idx + 2 < buffer.len() {
                                // Simple blending or overwrite?
                                // Overwrite for now, or match color
                                buffer[idx] = color.0;
                                buffer[idx+1] = color.1;
                                buffer[idx+2] = color.2;
                            }
                        }
                    }
                });
            }
        }
    }
    
    pub fn measure_height(&self, size_pt: f32) -> usize {
         let scale = Scale::uniform(size_pt);
         let v_metrics = self.font.v_metrics(scale);
         (v_metrics.ascent - v_metrics.descent + v_metrics.line_gap) as usize
    }
}
