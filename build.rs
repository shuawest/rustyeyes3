fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false) // We only need the client
        .compile(&["remote_server/gaze_stream.proto"], &["remote_server"])?;
    Ok(())
}
