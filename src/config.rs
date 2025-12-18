use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use anyhow::{Result, Context};
use crate::rectification::CalibrationConfig;

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub defaults: Defaults,
    pub ui: UiConfig,
    pub models: Models,
    #[serde(default)]
    pub calibration: CalibrationConfig,
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Models {
    pub l2cs_path: String,
    pub mobile_path: String,
    pub face_detection_path: String,
    pub face_mesh_path: String,
    pub head_pose_path: String,
}

impl Default for Models {
    fn default() -> Self {
        Self {
            l2cs_path: "models/l2cs_net.onnx".to_string(),
            mobile_path: "models/mobile_gaze.onnx".to_string(),
            face_detection_path: "models/face_detection.onnx".to_string(),
            face_mesh_path: "models/face_mesh.onnx".to_string(),
            head_pose_path: "models/head_pose.onnx".to_string(),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            defaults: Defaults::default(),
            ui: UiConfig::default(),
            models: Models::default(),
            calibration: CalibrationConfig::default(), // Added calibration default
        }
    }
}

impl AppConfig {
    pub const DEFAULT_CONFIG_PATH: &str = "config.json";


    pub fn load() -> Result<Self> {
        let config_path = Path::new(Self::DEFAULT_CONFIG_PATH);
        let mut config: AppConfig = if config_path.exists() {
            let data = fs::read_to_string(config_path).context("Failed to read config file")?;
            match serde_json::from_str(&data) {
                Ok(c) => {
                    println!("Loaded configuration from {}", Self::DEFAULT_CONFIG_PATH);
                    c
                },
                Err(e) => {
                    println!("Failed to parse config: {}. Loading defaults.", e);
                    AppConfig::default() 
                }
            }
        } else {
            println!("Configuration file not found. Creating default at {}", Self::DEFAULT_CONFIG_PATH);
            let defaults = AppConfig::default();
            
            // Write defaults to disk
            if let Ok(content) = serde_json::to_string_pretty(&defaults) {
                let _ = fs::write(Self::DEFAULT_CONFIG_PATH, content);
            }
            
            defaults
        };
        
        // Save if we just created defaults (redundant but safe)
        if !Path::new(Self::DEFAULT_CONFIG_PATH).exists() {
             if let Ok(content) = serde_json::to_string_pretty(&config) {
                 let _ = fs::write(Self::DEFAULT_CONFIG_PATH, content);
             }
        }
        
        // Always save back to ensure new fields are populated in the file
        config.save()?;
        
        Ok(config)
    }

    pub fn save(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(Self::DEFAULT_CONFIG_PATH, content)?;
        Ok(())
    }
}
