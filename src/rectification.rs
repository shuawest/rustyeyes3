use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationParams {
    pub yaw_offset: f32,
    pub pitch_offset: f32,
    pub yaw_gain: f32,
    pub pitch_gain: f32,
    #[serde(default)]
    pub yaw_curve: f32, // Coefficient for (deg - off)^2
    #[serde(default)]
    pub pitch_curve: f32, // Coefficient for (deg - off)^2
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            yaw_offset: 0.0,
            pitch_offset: 0.0,
            yaw_gain: 1.0,
            pitch_gain: 1.0,
            yaw_curve: 0.0,
            pitch_curve: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CalibrationConfig {
    pub models: HashMap<String, CalibrationParams>,
}

impl CalibrationConfig {
    pub fn new() -> Self {
        let mut models = HashMap::new();
        models.insert("l2cs".to_string(), CalibrationParams::default());
        models.insert("mobile".to_string(), CalibrationParams::default());
        Self { models }
    }
}
