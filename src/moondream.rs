use anyhow::Result;
use image::DynamicImage;
use crate::types::Point3D;
use std::thread;
use std::time::Duration;

pub struct MoondreamOracle;

impl MoondreamOracle {
    pub fn new() -> Result<Self> {
        println!("Initializing Moondream Oracle (Simulated Strategy)...");
        // No heavy model loading for Milestone 1
        Ok(Self)
    }

    pub fn gaze_at(&mut self, _image: &DynamicImage) -> Result<Point3D> {
        println!("Moondream: Analyzing image for gaze...");
        
        // Simulate inference delay (reduced to 0.5s for more frequent updates)
        thread::sleep(Duration::from_millis(500));

        // Mock Strategy: "Scanning Logic" to verify Overlay Coordinates
        // Cycle through 5 points: Center -> TL -> TR -> BR -> BL
        // Each call (every 2s) moves to next point.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static INDEX: AtomicUsize = AtomicUsize::new(0);
        let idx = INDEX.fetch_add(1, Ordering::Relaxed) % 5;
        
        let (x, y) = match idx {
            0 => (0.5, 0.5), // Center
            1 => (0.1, 0.1), // Top-Left
            2 => (0.9, 0.1), // Top-Right
            3 => (0.9, 0.9), // Bot-Right
            4 => (0.1, 0.9), // Bot-Left
            _ => (0.5, 0.5),
        };

        println!("Moondream: Gaze Detected at ({:.2}, {:.2}) [Scanning Mock]", x, y);
        Ok(Point3D { x, y, z: 0.0 })
    }
}
