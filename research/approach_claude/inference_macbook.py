#!/usr/bin/env python3
"""
Real-time Gaze Estimation Inference for MacBook Pro
====================================================

This module provides 60Hz+ gaze estimation using ONNX models trained on DGX Spark.
Designed to integrate with your existing PINTO zoo face detection pipeline.

Features:
- Screen-size adaptive (works on any resolution)
- Kalman smoothing for stable output
- Async Moondream2 validation (optional)
- CoreML acceleration on M-series chips

Usage:
    from inference_macbook import GazeEstimationPipeline, ScreenConfig
    
    # Initialize
    pipeline = GazeEstimationPipeline(
        face_detector_path='models/face_detector.onnx',  # Your existing PINTO model
        gaze_model_path='models/gaze_model_finetuned.onnx',
        calibration_path='models/calibration_mlp.onnx',
        screen=ScreenConfig(width=1920, height=1080),
    )
    
    # Real-time loop
    while True:
        frame = get_webcam_frame()
        result = pipeline.process(frame)
        if result:
            print(f"Gaze: ({result.x:.0f}, {result.y:.0f}) @ {result.fps:.0f}Hz")

Requirements:
    pip install numpy onnxruntime opencv-python
    # For CoreML acceleration (recommended on Mac):
    pip install coremltools
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import time
import threading
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScreenConfig:
    """Screen configuration for coordinate mapping."""
    width: int = 1920
    height: int = 1080
    
    # Optional: physical dimensions for distance-based corrections
    physical_width_cm: float = 34.5
    camera_offset_y_cm: float = 1.0  # Camera above screen center
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
    
    def update(self, width: int, height: int):
        """Update screen dimensions (e.g., on display change)."""
        self.width = width
        self.height = height


@dataclass
class GazeResult:
    """Gaze estimation result."""
    x: float                    # Screen X in pixels
    y: float                    # Screen Y in pixels
    raw_x: float               # Unsmoothed X
    raw_y: float               # Unsmoothed Y
    gaze_pitch: float          # Pitch angle (radians)
    gaze_yaw: float            # Yaw angle (radians)
    confidence: float          # Detection confidence [0, 1]
    latency_ms: float          # Processing time
    fps: float                 # Current throughput
    face_detected: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'screen_x': self.x,
            'screen_y': self.y,
            'raw_x': self.raw_x,
            'raw_y': self.raw_y,
            'gaze_pitch': self.gaze_pitch,
            'gaze_yaw': self.gaze_yaw,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms,
            'fps': self.fps,
        }


# =============================================================================
# Kalman Filter for Smoothing
# =============================================================================

class KalmanFilter1D:
    """
    1D Kalman filter for coordinate smoothing.
    
    Uses constant velocity model for smooth cursor movement.
    """
    
    def __init__(
        self, 
        process_noise: float = 0.01, 
        measurement_noise: float = 1.0,
        initial_value: float = 0.0
    ):
        self.q = process_noise      # Process noise
        self.r = measurement_noise  # Measurement noise
        
        # State: [position, velocity]
        self.x = np.array([initial_value, 0.0])
        
        # State covariance
        self.P = np.eye(2) * 100.0
        
        # State transition matrix
        self.F = np.array([[1.0, 1.0],
                          [0.0, 1.0]])
        
        # Measurement matrix
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.array([[self.q, 0.0],
                          [0.0, self.q]])
        
        # Measurement noise covariance
        self.R = np.array([[self.r]])
        
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement.
        Returns smoothed position.
        """
        if not self.initialized:
            self.x[0] = measurement
            self.initialized = True
            return measurement
        
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update
        y = measurement - self.H @ x_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = x_pred + K.flatten() * y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x[0]
    
    def reset(self, initial_value: float = 0.0):
        """Reset filter state."""
        self.x = np.array([initial_value, 0.0])
        self.P = np.eye(2) * 100.0
        self.initialized = False
    
    def set_measurement_noise(self, r: float):
        """Adjust measurement noise (lower = trust measurements more)."""
        self.r = r
        self.R = np.array([[r]])


class GazeSmoother:
    """Dual Kalman filter for X/Y smoothing with outlier rejection."""
    
    def __init__(
        self,
        process_noise: float = 0.005,
        measurement_noise: float = 0.5,
        outlier_threshold: float = 150.0  # pixels
    ):
        self.kf_x = KalmanFilter1D(process_noise, measurement_noise)
        self.kf_y = KalmanFilter1D(process_noise, measurement_noise)
        
        self.outlier_threshold = outlier_threshold
        self.history = deque(maxlen=10)
    
    def update(self, x: float, y: float, confidence: float = 1.0) -> Tuple[float, float]:
        """
        Update with new measurement, return smoothed coordinates.
        """
        # Outlier detection
        if len(self.history) >= 3:
            recent = list(self.history)[-5:]
            mean_x = np.mean([p[0] for p in recent])
            mean_y = np.mean([p[1] for p in recent])
            
            dist = np.sqrt((x - mean_x)**2 + (y - mean_y)**2)
            
            if dist > self.outlier_threshold:
                # Reject outlier, return prediction only
                return self.kf_x.x[0], self.kf_y.x[0]
        
        # Adjust noise based on confidence
        noise_mult = 1.0 / max(confidence, 0.1)
        self.kf_x.set_measurement_noise(0.5 * noise_mult)
        self.kf_y.set_measurement_noise(0.5 * noise_mult)
        
        # Update filters
        smooth_x = self.kf_x.update(x)
        smooth_y = self.kf_y.update(y)
        
        self.history.append((smooth_x, smooth_y))
        
        return smooth_x, smooth_y
    
    def reset(self):
        """Reset smoother state."""
        self.kf_x.reset()
        self.kf_y.reset()
        self.history.clear()


# =============================================================================
# ONNX Model Wrapper
# =============================================================================

class ONNXModel:
    """Wrapper for ONNX model inference."""
    
    def __init__(self, model_path: str, use_coreml: bool = True):
        import onnxruntime as ort
        
        # Select execution provider
        providers = []
        
        if use_coreml:
            # Try CoreML first (best for M-series)
            if 'CoreMLExecutionProvider' in ort.get_available_providers():
                providers.append('CoreMLExecutionProvider')
        
        # Fallback to CPU
        providers.append('CPUExecutionProvider')
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Warmup
        dummy = np.zeros([1] + list(self.input_shape[1:]), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
    
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        return self.session.run(None, {self.input_name: input_data})[0]


# =============================================================================
# Face Detection Interface
# =============================================================================

class FaceDetectorInterface:
    """
    Interface to your existing PINTO face detection model.
    
    Override this class to match your actual implementation.
    """
    
    def __init__(self, model_path: str, use_coreml: bool = True):
        """
        Initialize with your face detection ONNX model.
        
        Expected output:
            - Face bounding box [x1, y1, x2, y2]
            - Head pose [yaw, pitch, roll] in degrees
            - Eye positions (optional)
        """
        self.model = ONNXModel(model_path, use_coreml)
        
        # Add your preprocessing parameters here
        self.input_size = (640, 480)  # Adjust to match your model
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect face and extract head pose.
        
        Returns:
            Dict with:
                'bbox': [x1, y1, x2, y2]
                'head_pose': [yaw, pitch, roll] in degrees
                'confidence': float
                'left_eye': (x, y) or None
                'right_eye': (x, y) or None
            Or None if no face detected
        """
        # Preprocess
        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)
        
        # Run inference
        output = self.model(input_tensor)
        
        # Parse output (customize for your model)
        result = self._parse_output(output, w, h)
        
        return result
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for face detection model."""
        # Resize
        resized = cv2.resize(frame, self.input_size)
        
        # Normalize (adjust for your model)
        normalized = resized.astype(np.float32) / 255.0
        
        # CHW format
        chw = normalized.transpose(2, 0, 1)
        
        # Add batch dimension
        return np.expand_dims(chw, 0)
    
    def _parse_output(self, output: np.ndarray, orig_w: int, orig_h: int) -> Optional[Dict]:
        """
        Parse model output to extract face info.
        
        CUSTOMIZE THIS METHOD for your specific PINTO model output format.
        """
        # Placeholder implementation
        # Replace with your actual output parsing
        
        # Example: assuming output format [batch, num_faces, 5+3]
        # where 5 = [x1, y1, x2, y2, confidence]
        # and 3 = [yaw, pitch, roll]
        
        if output.shape[1] == 0:
            return None
        
        # Get best detection
        best_idx = 0  # or np.argmax(output[0, :, 4])
        det = output[0, best_idx]
        
        # Scale bbox to original image
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]
        
        return {
            'bbox': [
                det[0] * scale_x,
                det[1] * scale_y,
                det[2] * scale_x,
                det[3] * scale_y,
            ],
            'confidence': float(det[4]) if len(det) > 4 else 0.9,
            'head_pose': [
                det[5] if len(det) > 5 else 0.0,  # yaw
                det[6] if len(det) > 6 else 0.0,  # pitch
                det[7] if len(det) > 7 else 0.0,  # roll
            ],
            'left_eye': None,
            'right_eye': None,
        }


class SimpleFaceDetector:
    """
    Simple face detector using MediaPipe (fallback option).
    Use this if you don't have a custom PINTO model.
    """
    
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            raise ImportError("MediaPipe not installed. Run: pip install mediapipe")
    
    def detect(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect face using MediaPipe."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get bounding box from landmarks
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        
        # Eye landmarks (MediaPipe indices)
        left_eye_idx = [33, 133]  # Inner/outer corners
        right_eye_idx = [362, 263]
        
        left_eye = (
            np.mean([landmarks.landmark[i].x * w for i in left_eye_idx]),
            np.mean([landmarks.landmark[i].y * h for i in left_eye_idx]),
        )
        right_eye = (
            np.mean([landmarks.landmark[i].x * w for i in right_eye_idx]),
            np.mean([landmarks.landmark[i].y * h for i in right_eye_idx]),
        )
        
        # Estimate head pose from landmarks
        head_pose = self._estimate_head_pose(landmarks, w, h)
        
        return {
            'bbox': [min(xs), min(ys), max(xs), max(ys)],
            'confidence': 0.9,
            'head_pose': head_pose,
            'left_eye': left_eye,
            'right_eye': right_eye,
        }
    
    def _estimate_head_pose(self, landmarks, w, h) -> List[float]:
        """Rough head pose estimation from landmarks."""
        # Simplified: use nose tip and face orientation
        nose = landmarks.landmark[1]
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        
        # Yaw from ear positions
        yaw = np.arctan2(
            (right_ear.z - left_ear.z),
            (right_ear.x - left_ear.x)
        ) * 57.3  # rad to deg
        
        # Pitch from nose position (very rough)
        pitch = (nose.y - 0.5) * -60  # degrees
        
        # Roll from ear y-positions
        roll = np.arctan2(
            (right_ear.y - left_ear.y),
            (right_ear.x - left_ear.x)
        ) * 57.3
        
        return [yaw, pitch, roll]


# =============================================================================
# Main Pipeline
# =============================================================================

class GazeEstimationPipeline:
    """
    Real-time gaze estimation pipeline.
    
    Combines face detection, gaze model, and calibration for
    60Hz+ screen coordinate estimation.
    """
    
    def __init__(
        self,
        face_detector_path: Optional[str] = None,
        gaze_model_path: str = 'models/gaze_model_finetuned.onnx',
        calibration_path: str = 'models/calibration_mlp.onnx',
        screen: Optional[ScreenConfig] = None,
        use_coreml: bool = True,
        use_mediapipe: bool = False,
    ):
        """
        Initialize gaze estimation pipeline.
        
        Args:
            face_detector_path: Path to PINTO face detection ONNX (or None for MediaPipe)
            gaze_model_path: Path to trained gaze model ONNX
            calibration_path: Path to personal calibration ONNX
            screen: Screen configuration
            use_coreml: Use CoreML acceleration on Mac
            use_mediapipe: Use MediaPipe for face detection (if no PINTO model)
        """
        self.screen = screen or ScreenConfig()
        
        # Face detector
        if use_mediapipe or face_detector_path is None:
            self.face_detector = SimpleFaceDetector()
        else:
            self.face_detector = FaceDetectorInterface(face_detector_path, use_coreml)
        
        # Gaze model
        self.gaze_model = ONNXModel(gaze_model_path, use_coreml)
        
        # Calibration layer
        self.calibration = ONNXModel(calibration_path, use_coreml)
        
        # Smoother
        self.smoother = GazeSmoother(
            process_noise=0.005,
            measurement_noise=0.5,
            outlier_threshold=150.0,
        )
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)
        
        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def _preprocess_face(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Crop and preprocess face for gaze model."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Add margin
        face_w, face_h = x2 - x1, y2 - y1
        margin = 0.3
        
        x1 = max(0, int(x1 - face_w * margin))
        y1 = max(0, int(y1 - face_h * margin))
        x2 = min(w, int(x2 + face_w * margin))
        y2 = min(h, int(y2 + face_h * margin))
        
        # Crop
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to model input size
        face = cv2.resize(face, (224, 224))
        
        # Normalize
        face = face.astype(np.float32) / 255.0
        face = (face - self.mean) / self.std
        
        # CHW format + batch dimension
        face = face.transpose(2, 0, 1)
        face = np.expand_dims(face, 0)
        
        return face.astype(np.float32)
    
    def process(self, frame: np.ndarray) -> Optional[GazeResult]:
        """
        Process single frame.
        
        Args:
            frame: BGR image from webcam
        
        Returns:
            GazeResult with screen coordinates, or None if no face
        """
        start_time = time.perf_counter()
        
        # Step 1: Face detection
        detection = self.face_detector.detect(frame)
        
        if detection is None:
            return GazeResult(
                x=0, y=0, raw_x=0, raw_y=0,
                gaze_pitch=0, gaze_yaw=0,
                confidence=0, latency_ms=0, fps=0,
                face_detected=False,
            )
        
        # Step 2: Preprocess face
        face_input = self._preprocess_face(frame, detection['bbox'])
        
        if face_input is None:
            return None
        
        # Step 3: Gaze model inference
        gaze_output = self.gaze_model(face_input)
        pitch, yaw = gaze_output[0]  # radians
        
        # Step 4: Build calibration features
        h, w = frame.shape[:2]
        face_center_x = (detection['bbox'][0] + detection['bbox'][2]) / 2 / w
        face_center_y = (detection['bbox'][1] + detection['bbox'][3]) / 2 / h
        
        calib_features = np.array([[
            pitch,                              # gaze pitch
            yaw,                                # gaze yaw
            np.radians(detection['head_pose'][0]),  # head yaw (convert to rad)
            np.radians(detection['head_pose'][1]),  # head pitch
            np.radians(detection['head_pose'][2]),  # head roll
            face_center_x,                      # face x (normalized)
            face_center_y,                      # face y (normalized)
            self.screen.aspect_ratio,           # screen aspect ratio
        ]], dtype=np.float32)
        
        # Step 5: Calibration mapping
        screen_coords = self.calibration(calib_features)
        
        # Denormalize to pixels
        raw_x = float(screen_coords[0, 0] * self.screen.width)
        raw_y = float(screen_coords[0, 1] * self.screen.height)
        
        # Clamp to screen bounds
        raw_x = np.clip(raw_x, 0, self.screen.width)
        raw_y = np.clip(raw_y, 0, self.screen.height)
        
        # Step 6: Kalman smoothing
        smooth_x, smooth_y = self.smoother.update(
            raw_x, raw_y, 
            confidence=detection['confidence']
        )
        
        # Performance tracking
        elapsed = time.perf_counter() - start_time
        self.frame_times.append(elapsed)
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return GazeResult(
            x=smooth_x,
            y=smooth_y,
            raw_x=raw_x,
            raw_y=raw_y,
            gaze_pitch=float(pitch),
            gaze_yaw=float(yaw),
            confidence=detection['confidence'],
            latency_ms=elapsed * 1000,
            fps=fps,
            face_detected=True,
        )
    
    def update_screen(self, width: int, height: int):
        """Update screen dimensions."""
        self.screen.update(width, height)
        self.smoother.reset()
    
    def reset(self):
        """Reset smoother state."""
        self.smoother.reset()


# =============================================================================
# Async Moondream2 Validator (Optional)
# =============================================================================

class AsyncMoondreamValidator:
    """
    Run Moondream2 asynchronously for validation and drift detection.
    
    Does NOT block the real-time pipeline.
    """
    
    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        check_interval: int = 30,  # frames between checks
        drift_threshold: float = 100.0,  # pixels
    ):
        """
        Initialize Moondream2 validator.
        
        This runs in background and flags when recalibration may be needed.
        """
        self.check_interval = check_interval
        self.drift_threshold = drift_threshold
        self.frame_count = 0
        
        self._model = None
        self._tokenizer = None
        self._device = None
        
        # State
        self.last_validation_result = None
        self.drift_detected = False
        self.validation_thread = None
        
        # Lazy load model
        self._model_id = model_id
        self._revision = revision
    
    def _load_model(self):
        """Lazy load Moondream2 (heavy model)."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._device = "mps" if torch.backends.mps.is_available() else "cpu"
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, 
                revision=self._revision
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                revision=self._revision,
                trust_remote_code=True,
            ).to(self._device)
            
            print(f"Moondream2 loaded on {self._device}")
            
        except Exception as e:
            print(f"Warning: Could not load Moondream2: {e}")
            self._model = None
    
    def should_validate(self) -> bool:
        """Check if we should run validation this frame."""
        self.frame_count += 1
        return self.frame_count % self.check_interval == 0
    
    def validate_async(
        self, 
        frame: np.ndarray, 
        current_estimate: GazeResult,
        callback=None
    ):
        """
        Run Moondream2 validation in background thread.
        
        Args:
            frame: Current webcam frame
            current_estimate: Real-time pipeline estimate
            callback: Optional callback(drift_detected, moondream_estimate)
        """
        if self.validation_thread and self.validation_thread.is_alive():
            return  # Previous validation still running
        
        def _validate():
            self._load_model()
            
            if self._model is None:
                return
            
            try:
                import torch
                from PIL import Image
                
                # Convert frame to PIL
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Query Moondream2
                prompt = "Where is this person looking on their computer screen? Estimate the approximate x,y position as a percentage of the screen width and height. Answer in format: x%, y%"
                
                enc_image = self._model.encode_image(pil_image)
                answer = self._model.answer_question(enc_image, prompt, self._tokenizer)
                
                # Parse response (rough)
                moondream_x, moondream_y = self._parse_response(answer)
                
                if moondream_x is not None:
                    # Compare with real-time estimate
                    rt_x_pct = current_estimate.x / current_estimate.raw_x * 100 if current_estimate.raw_x > 0 else 50
                    rt_y_pct = current_estimate.y / current_estimate.raw_y * 100 if current_estimate.raw_y > 0 else 50
                    
                    drift = np.sqrt(
                        (moondream_x - rt_x_pct)**2 + 
                        (moondream_y - rt_y_pct)**2
                    )
                    
                    self.drift_detected = drift > self.drift_threshold / 10  # percentage threshold
                    
                    self.last_validation_result = {
                        'moondream_x_pct': moondream_x,
                        'moondream_y_pct': moondream_y,
                        'realtime_x_pct': rt_x_pct,
                        'realtime_y_pct': rt_y_pct,
                        'drift_pct': drift,
                        'drift_detected': self.drift_detected,
                    }
                    
                    if callback:
                        callback(self.drift_detected, self.last_validation_result)
                        
            except Exception as e:
                print(f"Moondream validation error: {e}")
        
        self.validation_thread = threading.Thread(target=_validate)
        self.validation_thread.start()
    
    def _parse_response(self, response: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse Moondream2 response for percentages."""
        import re
        
        # Look for patterns like "50%, 30%" or "50% x, 30% y"
        matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', response)
        
        if len(matches) >= 2:
            return float(matches[0]), float(matches[1])
        
        return None, None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_pipeline_simple(
    gaze_model_path: str,
    calibration_path: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> GazeEstimationPipeline:
    """
    Create a simple pipeline using MediaPipe for face detection.
    Use this if you don't have a custom PINTO model yet.
    """
    return GazeEstimationPipeline(
        face_detector_path=None,
        gaze_model_path=gaze_model_path,
        calibration_path=calibration_path,
        screen=ScreenConfig(width=screen_width, height=screen_height),
        use_mediapipe=True,
    )


def benchmark_pipeline(pipeline: GazeEstimationPipeline, n_frames: int = 100):
    """Benchmark pipeline latency."""
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        pipeline.process(dummy_frame)
    
    # Benchmark
    times = []
    for _ in range(n_frames):
        start = time.perf_counter()
        pipeline.process(dummy_frame)
        times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # to ms
    
    print(f"\nPipeline Benchmark ({n_frames} frames):")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std: {np.std(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")
    print(f"  Max: {np.max(times):.2f} ms")
    print(f"  Throughput: {1000 / np.mean(times):.1f} Hz")


# =============================================================================
# Demo
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Gaze Estimation')
    parser.add_argument('--gaze-model', type=str, default='models/gaze_model_finetuned.onnx')
    parser.add_argument('--calibration', type=str, default='models/calibration_mlp.onnx')
    parser.add_argument('--screen-width', type=int, default=1920)
    parser.add_argument('--screen-height', type=int, default=1080)
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark only')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    args = parser.parse_args()
    
    # Create pipeline
    print("Initializing pipeline...")
    
    pipeline = create_pipeline_simple(
        gaze_model_path=args.gaze_model,
        calibration_path=args.calibration,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )
    
    if args.benchmark:
        benchmark_pipeline(pipeline)
    else:
        # Real-time demo
        print("\nStarting real-time demo. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = pipeline.process(frame)
            
            if result and result.face_detected:
                # Draw gaze point on frame (scaled down for display)
                scale = frame.shape[1] / args.screen_width
                gaze_x = int(result.x * scale)
                gaze_y = int(result.y * scale * args.screen_height / args.screen_width)
                
                cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 0), -1)
                
                # Display info
                cv2.putText(
                    frame, 
                    f"Gaze: ({result.x:.0f}, {result.y:.0f}) @ {result.fps:.0f}Hz",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"Latency: {result.latency_ms:.1f}ms",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    frame, "No face detected",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            cv2.imshow('Gaze Estimation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()