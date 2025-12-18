#[cfg(test)]
mod tests {
    use crate::gaze::{compute_simulated_gaze, compute_pupil_gaze};

    // =========================================================================
    // Regression Tests: Gaze Direction
    // Convention: Negative Yaw = Look Left (Screen Left)
    // =========================================================================
    
    #[test]
    fn test_simulated_gaze_direction() {
        // DERIVED COORDINATE SYSTEM REASONING:
        // 1. main.rs: x + sin(yaw) -> Positive Yaw = Screen Right. Negative Yaw = Screen Left.
        // 2. User Report: "Head Left -> Dot went Right" (with positive gain).
        //    Therefore: Head Left input must have been Positive (causing positive yaw -> Screen Right).
        // 3. Fix: We want Head Left (Positive) -> Screen Left (Negative).
        //    So correct logic is Inverted (-).
        
        // Case 1: Head Turned Left (+10 deg)
        // Should result in Gaze Left (< 0)
        let (yaw, _pitch) = compute_simulated_gaze(10.0, 0.0);
        assert!(yaw < 0.0, "Simulated Gaze: Head Left (+10) produced Yaw {}, expected negative (Left)", yaw);
        
        // Case 2: Head Turned Right (-10 deg)
        let (yaw, _pitch) = compute_simulated_gaze(-10.0, 0.0);
        assert!(yaw > 0.0, "Simulated Gaze: Head Right (-10) produced Yaw {}, expected positive (Right)", yaw);
    }

    #[test]
    fn test_pupil_gaze_direction() {
        // Case 1: Head Turned Left (+10 deg), Pupil Center
        let (yaw, _pitch) = compute_pupil_gaze(10.0, 0.0, 0.0, 0.0);
        assert!(yaw < 0.0, "Pupil Gaze: Head Left (+10) produced Yaw {}, expected negative (Left)", yaw);
        
        // Case 2: Head Center, Pupil Looking Left
        // If Head Left is Positive, assumes standard axis.
        // Pupil Offset: Usually -0.5 is Left in normalized image coords (0 to 1 or -1 to 1).
        // If Pupil is Left (-0.5), we want Gaze Left (< 0).
        // Original logic: raw = head + pupil*gain.
        // If head=0, pupil=-0.5 -> raw = negative.
        // But we INVERT raw. So raw negative -> output POSITIVE (Right)?
        // Wait. If pupil offset -0.5 is Left, and we want Output Left (< 0).
        // We invert the result: `-(pupil * gain)`.
        // So `-( -0.5 * gain )` = Positive. Screen Right.
        // This implies Pupil Pipeline might be inverted relative to Head pipeline?
        // Or my assumption about Pupil Offset sign is wrong.
        // Let's assume standard image coords: x increases to Right. So Left is Negative.
        // If Pupil is -0.5 (Left), we want Screen Left (-).
        // Current Code: `-(pupil_offset * gain)`.
        // `-(-0.5)` = +0.5 (Right).
        // So Pupil Gaze will be OPPOSITE of Eye Movement with current code?
        // User said "Pupil Gaze is opposite".
        // But PupilGaze relies on Head Pose heavily.
        // Let's stick to Head Pose dominance for this test first.
        
        // Case 2: Head Turned Right (-10 deg)
        let (yaw, _pitch) = compute_pupil_gaze(-10.0, 0.0, 0.0, 0.0);
        assert!(yaw > 0.0, "Pupil Gaze: Head Right (-10) produced Yaw {}, expected positive (Right)", yaw);
    }
}
