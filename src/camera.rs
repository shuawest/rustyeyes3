use image::{ImageBuffer, Rgb};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use anyhow::{Result, Context, anyhow};
use colored::*;

pub struct CameraSource {
    camera: Camera,
}

impl CameraSource {
    pub fn new(index: usize) -> Result<Self> {
        let cam_index = CameraIndex::Index(index as u32);
        let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut camera = Camera::new(cam_index, requested).context("Failed to create camera instance")?;
        
        camera.open_stream().map_err(|e| anyhow!(e)).context("Failed to open camera stream")?;
        
        println!("{}", format!("Opened camera: {}", camera.info().human_name()).green());
        println!("Format: {}", camera.camera_format());

        Ok(Self { camera })
    }

    pub fn capture(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let frame = self.camera.frame().map_err(|e| anyhow!(e)).context("Failed to get frame")?;
        let decoded = frame.decode_image::<RgbFormat>().map_err(|e| anyhow!(e)).context("Failed to decode frame")?;
        Ok(decoded)
    }

    pub fn width(&self) -> u32 {
        self.camera.resolution().width()
    }

    pub fn height(&self) -> u32 {
        self.camera.resolution().height()
    }

    pub fn name(&self) -> String {
        self.camera.info().human_name()
    }
}
