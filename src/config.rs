use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use anyhow::Result;

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub defaults: Defaults,
    pub ui: UiConfig,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Defaults {
    pub show_mesh: bool,
    pub show_pose: bool,
    pub show_gaze: bool,
    pub mirror_mode: bool,
    pub show_overlay: bool,
    pub moondream_active: bool,
    // New Params
    pub head_pose_length: f32, 
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub menu_scale: usize,
    pub font_size_pt: u32,
    pub font_family: String, // New: Font Family (e.g. "Arial", "Terminus")
    pub mesh_dot_size: usize,
    pub mesh_color_hex: String, // e.g. "#FF0000"
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            show_mesh: true,
            show_pose: true,
            show_gaze: false,
            show_overlay: true,
            mirror_mode: true,
            moondream_active: false,
            head_pose_length: 150.0,
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            menu_scale: 2,
            font_size_pt: 12,
            font_family: "Monospace".to_string(),
            mesh_dot_size: 2,
            mesh_color_hex: "#FF0000".to_string(),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            defaults: Defaults::default(),
            ui: UiConfig::default(),
        }
    }
}

impl AppConfig {
    const PATH: &'static str = "config.json";

    pub fn load() -> Result<Self> {
        let config = if Path::new(Self::PATH).exists() {
            let content = fs::read_to_string(Self::PATH)?;
            // Attempt to load. serde_json::from_str will use Default for missing fields due to #[serde(default)]
            match serde_json::from_str::<AppConfig>(&content) {
                Ok(c) => {
                    println!("Loaded configuration from {}", Self::PATH);
                    c
                },
                Err(e) => {
                    println!("Error parsing config: {}. Loading defaults.", e);
                    Self::default()
                }
            }
        } else {
            println!("Configuration file not found. Creating default at {}", Self::PATH);
            Self::default()
        };
        
        // Always save back to ensure new fields are populated in the file
        config.save()?;
        
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(Self::PATH, content)?;
        Ok(())
    }
}
