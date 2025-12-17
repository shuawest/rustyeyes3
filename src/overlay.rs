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

    pub fn update(&mut self, x: f32, y: f32) -> Result<()> {
        if let Some(stdin) = self.process.stdin.as_mut() {
            writeln!(stdin, "{} {}", x, y)?;
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
