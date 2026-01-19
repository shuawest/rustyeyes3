use anyhow::{Result, Context};
use tonic::transport::Channel;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use crate::types::{Point3D, PipelineOutput, Landmarks, Rect};

// Import generated proto code
pub mod proto {
    tonic::include_proto!("gazestream");
}

pub struct GazeStreamClient {
    // We store the endpoint URL so we can create new clients for health checks
    url: String,
    // We keep the main client ready
    client: proto::gaze_stream_service_client::GazeStreamServiceClient<Channel>,
}

pub struct RemoteResult {
    pub face_mesh: Option<Landmarks>,
    pub face_box: Option<Rect>,  // Bounding box for face region
    pub gaze: Option<(f32, f32)>, // yaw, pitch
    pub timestamp: u64,
    pub stream_id: String,
    pub trace_timestamps: std::collections::HashMap<String, i64>,
}

impl GazeStreamClient {
    pub async fn connect(url: String) -> Result<Self> {
        // Configure endpoint with TCP_NODELAY to fix 250ms latency on small payloads
        let channel = tonic::transport::Endpoint::from_shared(url.clone())?
            .tcp_nodelay(true)
            .connect()
            .await
            .context("Failed to connect to gRPC server")?;
            
        Ok(Self { 
            url,
            client: proto::gaze_stream_service_client::GazeStreamServiceClient::new(channel),
        })
    }
    
    /// Stream video frames to server and receive results
    /// Returns a channel to send frames to, and a stream of results
    pub async fn stream_video(
        &mut self,
    ) -> Result<(
        mpsc::Sender<proto::VideoFrame>,
        tonic::Streaming<proto::FaceMeshResult>,
    )> {
        // Use size 1 to prevent bufferbloat. If network is slow, drop frames immediately.
        let (tx, rx) = mpsc::channel(1);
        let request_stream = ReceiverStream::new(rx);
        
        let response = self.client.stream_gaze(request_stream).await?;
        let inbound_stream = response.into_inner();
        
        Ok((tx, inbound_stream))
    }
    
    /// Check health of the remote service using standard gRPC Health Checking Protocol
    pub async fn check_health(&self) -> Result<bool> {
        // Create Endpoint from URL
        let endpoint = tonic::transport::Endpoint::from_shared(self.url.clone())
            .context("Invalid URL for health check")?;
            
        let channel = endpoint.connect().await
            .context("Failed to connect for health check")?;

        let mut health_client = tonic_health::pb::health_client::HealthClient::new(channel);
            
        let request = tonic::Request::new(tonic_health::pb::HealthCheckRequest {
            service: "gazestream.GazeStreamService".to_string(),
        });
        
        match health_client.check(request).await {
            Ok(response) => {
                let status = response.into_inner().status;
                Ok(status == tonic_health::pb::health_check_response::ServingStatus::Serving as i32)
            },
            Err(e) => {
                // Try checking overall server health (empty service name)
                 let request_all = tonic::Request::new(tonic_health::pb::HealthCheckRequest {
                    service: "".to_string(),
                });
                if let Ok(resp) = health_client.check(request_all).await {
                     let status = resp.into_inner().status;
                     return Ok(status == tonic_health::pb::health_check_response::ServingStatus::Serving as i32);
                }
                
                // Fallback: If we can't check standard health, but we connected (channel is up), we assume somewhat healthy?
                // But endpoint.connect() just establishes TCP.
                // Log warning and return error if official check fails.
                Err(anyhow::anyhow!("Health check failed: {}", e))
            }
        }
    }
}

// Helper to convert internal ImageBuffer to Proto VideoFrame
pub fn frame_to_proto(
    frame: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>, 
    timestamp_us: i64, 
    stream_id: &str
) -> proto::VideoFrame {
    // For MVP: Send raw RGB bytes or encode as JPEG
    // Encoding as JPEG reduces bandwidth significantly
    let mut buf = Vec::new();
    let dyn_imgs = image::DynamicImage::ImageRgb8(frame.clone());
    
    // Attempt JPEG encoding
    // Quality 50 is sufficient for face mesh and reduces size significantly vs default 75
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 50);
    let _ = encoder.encode_image(&dyn_imgs);

    // Convert to microrsecond timestamp
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_micros() as i64;
    let mut stamps = std::collections::HashMap::new();
    stamps.insert("client_send".to_string(), now);

    proto::VideoFrame {
        frame_data: buf,
        timestamp_us,
        width: frame.width() as i32,
        height: frame.height() as i32,
        stream_id: stream_id.to_string(),
        trace_timestamps: stamps,
    }
}

// Helper to convert Proto Result to internal RemoteResult
pub fn proto_to_result(proto_res: proto::FaceMeshResult) -> RemoteResult {
    let mut landmarks = None;
    
    if !proto_res.landmarks.is_empty() {
        let points: Vec<Point3D> = proto_res.landmarks.iter().map(|lm| Point3D {
            x: lm.x,
            y: lm.y,
            z: lm.z,
        }).collect();
        landmarks = Some(Landmarks { points });
    }
    
    let face_box = if let Some(bbox) = proto_res.face_box {
        Some(Rect {
            x: bbox.x,
            y: bbox.y,
            width: bbox.width,
            height: bbox.height,
        })
    } else {
        None
    };
    
    let gaze = if let Some(g) = proto_res.gaze {
        Some((g.yaw, g.pitch))
    } else {
        None
    };
    
    RemoteResult {
        face_mesh: landmarks,
        face_box,
        gaze,
        timestamp: proto_res.timestamp_us as u64,
        stream_id: proto_res.stream_id,
        trace_timestamps: proto_res.trace_timestamps,
    }
}
