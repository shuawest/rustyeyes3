use std::process::Command;

fn main() {
    // Only compile on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=src/overlay_sidecar.swift");
        
        let status = Command::new("swiftc")
            .arg("src/overlay_sidecar.swift")
            .arg("-o")
            .arg("overlay_app")
            .status()
            .expect("Failed to run swiftc");
        
        if !status.success() {
            panic!("Failed to compile overlay_sidecar.swift");
        }
    }
}
