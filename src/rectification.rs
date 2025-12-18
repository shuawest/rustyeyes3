use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationParams {
    pub yaw_offset: f32,
    pub pitch_offset: f32,
    pub yaw_gain: f32,
    pub pitch_gain: f32,
}

impl Default for CalibrationParams {
    fn default() -> Self {
        Self {
            yaw_offset: 0.0,
            pitch_offset: 12.0, // Default heuristic
            yaw_gain: 5.0,
            pitch_gain: 5.0,
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
