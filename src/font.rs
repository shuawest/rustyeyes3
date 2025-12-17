/// A very simple 5x7 bitmap font for ASCII characters.
/// Only supports uppercase, numbers, and basic punctuation.


// Minimal 5-row encoded font (3 bits width per row)
// Wait, standard font is easier to read if I just use a simple logic.
// Let's assume a fixed buffer drawing.

pub fn draw_text_line(buffer: &mut [u8], width: usize, height: usize, x: usize, y: usize, text: &str, color: (u8, u8, u8), scale: usize) {
    let mut cx = x;
    for c in text.chars() {
        draw_char(buffer, width, height, cx, y, c, color, scale);
        cx += (3 * scale) + scale; // 3 width + 1 spacing, scaled
    }
}

pub fn measure_text_width(text: &str, scale: usize) -> usize {
    text.len() * ((3 * scale) + scale)
}

fn draw_char(buffer: &mut [u8], width: usize, height: usize, x: usize, y: usize, c: char, color: (u8, u8, u8), scale: usize) {
    // 3x5 font definition (compact)
    // Encoded as 5 integers, 3 bits each
    let map = match c.to_ascii_uppercase() {
        '0' => [0x7, 0x5, 0x5, 0x5, 0x7],
        '1' => [0x2, 0x6, 0x2, 0x2, 0x7],
        '2' => [0x7, 0x1, 0x7, 0x4, 0x7],
        '3' => [0x7, 0x1, 0x7, 0x1, 0x7],
        '4' => [0x5, 0x5, 0x7, 0x1, 0x1],
        '5' => [0x7, 0x4, 0x7, 0x1, 0x7],
        '6' => [0x7, 0x4, 0x7, 0x5, 0x7],
        '7' => [0x7, 0x1, 0x2, 0x4, 0x4],
        '8' => [0x7, 0x5, 0x7, 0x5, 0x7],
        '9' => [0x7, 0x5, 0x7, 0x1, 0x7],
        ' ' => [0x0, 0x0, 0x0, 0x0, 0x0],
        ':' => [0x0, 0x2, 0x0, 0x2, 0x0],
        'L' => [0x4, 0x4, 0x4, 0x4, 0x7],
        'A' => [0x2, 0x5, 0x7, 0x5, 0x5],
        'S' => [0x3, 0x4, 0x2, 0x1, 0x6], // S is hard in 3x5, approx
        'T' => [0x7, 0x2, 0x2, 0x2, 0x2],
        'C' => [0x7, 0x4, 0x4, 0x4, 0x7],
        'E' => [0x7, 0x4, 0x6, 0x4, 0x7],
        'N' => [0x6, 0x5, 0x5, 0x5, 0x5], // n
        'P' => [0x7, 0x5, 0x7, 0x4, 0x4],
        'O' => [0x7, 0x5, 0x5, 0x5, 0x7],
        'I' => [0x7, 0x2, 0x2, 0x2, 0x7],
        'D' => [0x6, 0x5, 0x5, 0x5, 0x6],
        'R' => [0x6, 0x5, 0x6, 0x5, 0x5],
        'M' => [0x5, 0x7, 0x5, 0x5, 0x5],
        ',' => [0x0, 0x0, 0x0, 0x2, 0x4],
        '(' => [0x2, 0x4, 0x4, 0x4, 0x2],
        ')' => [0x2, 0x1, 0x1, 0x1, 0x2],
        '[' => [0x7, 0x4, 0x4, 0x4, 0x7], // Square bracket left
        ']' => [0x7, 0x1, 0x1, 0x1, 0x7], // Square bracket right
        _ =>   [0x7, 0x7, 0x7, 0x7, 0x7], // block
    };

    for (row, bits) in map.iter().enumerate() {
        for col in 0..3 {
            // Check bit (column 0 is highest bit 2, col 2 is bit 0)
            if (bits >> (2 - col)) & 1 == 1 {
                // Scaled drawing
                for dy in 0..scale {
                    for dx in 0..scale {
                         let px = x + (col * scale) + dx;
                         let py = y + (row * scale) + dy;
                         if px < width && py < height {
                             let idx = (py * width + px) * 3;
                             if idx + 2 < buffer.len() {
                                 buffer[idx] = color.0;
                                 buffer[idx+1] = color.1;
                                 buffer[idx+2] = color.2;
                             }
                         }
                    }
                }
            }
        }
    }
}
