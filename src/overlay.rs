use std::process::{Command, Child, Stdio};
use std::io::Write;
use anyhow::Result;

pub struct OverlayWindow {
    process: Child,
}

impl OverlayWindow {
    pub fn new(_width: usize, _height: usize) -> Result<Self> {
        let process = Command::new("./overlay_app")
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
            
        Ok(Self { process })
    }

    pub fn update_gaze(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            writeln!(stdin, "G {} {}", x, y)?;
        }
        Ok(())
    }

    pub fn update_moondream(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            writeln!(stdin, "M {} {}", x, y)?;
        }
        Ok(())
    }

    pub fn update_captured_onnx(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            writeln!(stdin, "C {} {}", x, y)?;
        }
        Ok(())
    }

    pub fn update_captured_onnx_verified(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            // C = Completed/Verified
            writeln!(stdin, "C {} {}", x, y)?;
        }
        Ok(())
    }

    pub fn update_captured_onnx_pending(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            // P = Pending
            writeln!(stdin, "P {} {}", x, y)?;
        }
        Ok(())
    }

    pub fn update_font(&mut self, family: &str, size: u32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            writeln!(stdin, "F {} {}", family, size)?;
        }
        Ok(())
    }

    pub fn update_menu(&mut self, menu_text: &str) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            let safe_text = menu_text.replace('\n', "|");
            writeln!(stdin, "S {}", safe_text)?;
        }
        Ok(())
    }
}

impl Drop for OverlayWindow {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
