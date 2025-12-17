use anyhow::{Result, Context};
use image::DynamicImage;
use crate::types::Point3D;
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::io::{Write, BufRead, BufReader};
use serde::{Serialize, Deserialize};

#[derive(Serialize)]
struct MoondreamRequest {
    image_path: String,
    timestamp: u64,
}

#[derive(Deserialize, Debug)]
struct MoondreamResponse {
    status: String,
    response: Option<String>,
    error: Option<String>,
    #[allow(dead_code)]
    timestamp: u64,
}

pub struct MoondreamOracle {
    process: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl MoondreamOracle {
    pub fn new() -> Result<Self> {
        println!("Launching Moondream2 Python Server...");
        
        let mut child = Command::new("venv/bin/python3")
            .arg("scripts/moondream_server.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // Show Python debug logs
            .spawn()
            .context("Failed to spawn Python server. Please run `./run.sh` to set up the environment.")?;
        
        let stdin = child.stdin.take()
            .context("Failed to open stdin to Python server")?;
        let stdout = child.stdout.take()
            .context("Failed to open stdout from Python server")?;
        
        let stdout = BufReader::new(stdout);
        
        println!("Moondream2 server started (Python subprocess)");
        
        Ok(Self {
            process: child,
            stdin,
            stdout,
        })
    }

    pub fn gaze_at(&mut self, image: &DynamicImage) -> Result<Point3D> {
        // Save image to temporary file
        let temp_path = "/tmp/rustyeyes_moondream_frame.jpg";
        image.save(temp_path)
            .context("Failed to save temporary image")?;
        
        // Send request to Python server
        let request = MoondreamRequest {
            image_path: temp_path.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        
        let request_json = serde_json::to_string(&request)?;
        writeln!(self.stdin, "{}", request_json)
            .context("Failed to write to Python server stdin")?;
        self.stdin.flush()?;
        
        // Read response (blocking until Python responds)
        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line)
            .context("Failed to read from Python server stdout")?;
        
        let response: MoondreamResponse = serde_json::from_str(&response_line)
            .context("Failed to parse Python server response")?;
        
        if response.status != "success" {
            anyhow::bail!("Moondream error: {:?}", response.error);
        }
        
        // Parse gaze from natural language response
        let gaze_text = response.response.unwrap_or_default();
        println!("[MOONDREAM] Gaze response: \"{}\"", gaze_text);
        
        // Try to extract coordinates from response
        // Moondream might say things like "center", "top-left", "looking at the screen", etc.
        let point = parse_gaze_response(&gaze_text);
        
        Ok(point)
    }
}

impl Drop for MoondreamOracle {
    fn drop(&mut self) {
        let _ = self.process.kill();
        println!("Moondream2 server stopped");
    }
}

/// Parse natural language gaze response into screen coordinates
fn parse_gaze_response(text: &str) -> Point3D {
    // PRIORITY 1: Check for COORDS: prefix from point() API
    if text.starts_with("COORDS:") {
        let coords_str = text.strip_prefix("COORDS:").unwrap();
        if let Some((x, y)) = extract_coordinates(coords_str) {
            println!("[MOONDREAM] Using direct coordinates from point() API");
            return Point3D { x, y, z: 0.0 };
        }
    }
    
    let text_lower = text.to_lowercase();
    
    // PRIORITY 2: Try to extract explicit coordinates from natural language
    if let Some((x, y)) = extract_coordinates(&text_lower) {
        return Point3D { x, y, z: 0.0 };
    }
    
    // Fallback: Parse directional words
    let (x, y) = if text_lower.contains("center") || text_lower.contains("middle") {
        (0.5, 0.5)
    } else if text_lower.contains("top") && text_lower.contains("left") {
        (0.2, 0.2)
    } else if text_lower.contains("top") && text_lower.contains("right") {
        (0.8, 0.2)
    } else if text_lower.contains("bottom") && text_lower.contains("left") {
        (0.2, 0.8)
    } else if text_lower.contains("bottom") && text_lower.contains("right") {
        (0.8, 0.8)
    } else if text_lower.contains("top") {
        (0.5, 0.2)
    } else if text_lower.contains("bottom") {
        (0.5, 0.8)
    } else if text_lower.contains("left") {
        (0.2, 0.5)
    } else if text_lower.contains("right") {
        (0.8, 0.5)
    } else {
        // Unknown - default to center
        (0.5, 0.5)
    };
    
    Point3D { x, y, z: 0.0 }
}

/// Try to extract numeric coordinates from text
fn extract_coordinates(text: &str) -> Option<(f32, f32)> {
    // Try patterns like "0.5, 0.3" or "(0.5, 0.3)" or "x: 0.5 y: 0.3"
    use regex::Regex;
    
    // Pattern: two floats separated by comma or space
    let re = Regex::new(r"(\d+\.?\d*)[,\s]+(\d+\.?\d*)").ok()?;
    
    if let Some(caps) = re.captures(text) {
        let x: f32 = caps.get(1)?.as_str().parse().ok()?;
        let y: f32 = caps.get(2)?.as_str().parse().ok()?;
        
        // Validate range
        if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) {
            return Some((x, y));
        }
    }
    
    None
}
