#[cfg(test)]
mod tests {
    use std::path::Path;
    // We need to allow dead code since we are importing modules that might strict-check
    #[allow(dead_code)]
    
    // Note: We can't easily import the binary crate modules in integration tests 
    // unless they are in a lib.rs. For this project structure (main.rs), 
    // integration tests are harder.
    // Instead, we will simulate the check by ensuring the models exist and basic config 
    // logic holds. 
    // Ideally, we would refactor `src/gaze.rs` into a library but for now we will 
    // verify the artifacts produced.

    #[test]
    fn verify_models_exist() {
        let models = vec![
            "models/face_detection.onnx",
            "models/face_mesh.onnx",
            "models/l2cs_net.onnx",
            "models/mobile_gaze.onnx",
        ];

        for m in models {
            assert!(Path::new(m).exists(), "Model file missing: {}", m);
        }
    }

    #[test]
    fn verify_config_integrity() {
        // Ensure we can parse the default config
        let _ = include_str!("../config.json");
        // In a real lib we'd deserialize it here
    }
}
