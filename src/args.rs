use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Camera Index (default 0)
    #[arg(short, long, default_value_t = 0)]
    pub cam_index: u32,

    /// Initial model to load (mesh, detection, pose, gaze)
    #[arg(long)]
    pub model: Option<String>,

    /// Mirror the camera output
    #[arg(long, default_value_t = false)]
    pub mirror: bool,

    /// List available cameras
    #[arg(long)]
    pub list: bool,
}
