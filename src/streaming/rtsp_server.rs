use anyhow::{Result, anyhow};
#[cfg(target_os = "linux")]
use gstreamer as gst;
#[cfg(target_os = "linux")]
use gstreamer_app as gst_app;
#[cfg(target_os = "linux")]
use gstreamer_rtsp_server as gst_rtsp;
#[cfg(target_os = "linux")]
use gstreamer_rtsp_server::prelude::*;

#[cfg(target_os = "linux")]
pub struct CameraStreamServer {
    server: gst_rtsp::RTSPServer,
    mainloop: Option<glib::MainLoop>,
}

#[cfg(not(target_os = "linux"))]
pub struct CameraStreamServer;

#[cfg(target_os = "linux")]
impl CameraStreamServer {
    pub fn new(port: u16) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;
        
        let server = gst_rtsp::RTSPServer::new();
        server.set_service(&port.to_string());
        
        Ok(Self {
            server,
            mainloop: None,
        })
    }
    
    /// Add a camera stream from a V4L2 device
    pub fn add_camera_stream(&self, path: &str, device_index: usize) -> Result<()> {
        let mounts = self.server.mount_points().ok_or_else(|| anyhow!("No mount points"))?;
        let factory = gst_rtsp::RTSPMediaFactory::new();
        
        // Create GStreamer pipeline for V4L2 camera
        // Format: v4l2src device=/dev/video0 ! videoconvert ! x264enc ! rtph264pay name=pay0
        let pipeline = format!(
            "v4l2src device=/dev/video{} ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! rtph264pay name=pay0 pt=96",
            device_index
        );
        
        factory.set_launch(&pipeline);
        factory.set_shared(true);
        
        mounts.add_factory(path, &factory);
        
        println!("[RTSP] Added camera stream: rtsp://<host>:{}{}", 
                 self.server.service().unwrap(), path);
        
        Ok(())
    }
    
    /// Start the RTSP server
    pub fn start(&mut self) -> Result<()> {
        let server_id = self.server.attach(None)?;
        
        println!("[RTSP] Server started on port {}", self.server.service().unwrap());
        println!("[RTSP] Server ID: {:?}", server_id);
        
        // Store the mainloop for later shutdown
        let mainloop = glib::MainLoop::new(None, false);
        self.mainloop = Some(mainloop.clone());
        
        // Run mainloop in separate thread
        std::thread::spawn(move || {
            println!("[RTSP] Starting GLib mainloop...");
            mainloop.run();
        });
        
        Ok(())
    }
    
    pub fn stop(&mut self) {
        if let Some(mainloop) = &self.mainloop {
            mainloop.quit();
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl CameraStreamServer {
    pub fn new(_port: u16) -> Result<Self> {
        println!("[WARN] RTSP server only supported on Linux");
        Ok(Self)
    }
    
    pub fn add_camera_stream(&self, _path: &str, _device_index: usize) -> Result<()> {
        Ok(())
    }
    
    pub fn start(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for CameraStreamServer {
    fn drop(&mut self) {
        self.stop();
    }
}
