#!/usr/bin/env python3
"""
Gaze Calibration Starter - Phase 1 Implementation
================================================

This script implements the calibration layer approach using your existing
ONNX model's output. Run this to establish a baseline before moving to
more sophisticated approaches.

Usage:
    python gaze_calibration_starter.py --calibration-dir ./calibration_data
    python gaze_calibration_starter.py --calibration-dir ./calibration_data --evaluate

Requirements:
    pip install numpy scikit-learn matplotlib joblib
"""

import json
import glob
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ML imports
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class CalibrationSample:
    """Structured calibration data point."""
    timestamp: int
    screen_x: float
    screen_y: float
    yaw: float
    pitch: float
    roll: float
    left_eye_x: float
    left_eye_y: float
    right_eye_x: float
    right_eye_y: float
    image_path: Optional[str] = None


def load_calibration_data(calibration_dir: str) -> List[CalibrationSample]:
    """Load all calibration samples from directory."""
    samples = []
    json_files = sorted(glob.glob(f"{calibration_dir}/*.json"))
    
    print(f"Found {len(json_files)} calibration files")
    
    for json_path in json_files:
        with open(json_path) as f:
            data = json.load(f)
        
        gaze = data.get('inference', {}).get('Gaze', {})
        
        # Skip if missing required fields
        if not gaze:
            print(f"  Warning: Missing Gaze data in {json_path}")
            continue
        
        sample = CalibrationSample(
            timestamp=data.get('timestamp', 0),
            screen_x=data.get('screen_x', 0),
            screen_y=data.get('screen_y', 0),
            yaw=gaze.get('yaw', 0),
            pitch=gaze.get('pitch', 0),
            roll=gaze.get('roll', 0),
            left_eye_x=gaze.get('left_eye', {}).get('x', 0),
            left_eye_y=gaze.get('left_eye', {}).get('y', 0),
            right_eye_x=gaze.get('right_eye', {}).get('x', 0),
            right_eye_y=gaze.get('right_eye', {}).get('y', 0),
            image_path=json_path.replace('.json', '.jpg'),
        )
        samples.append(sample)
    
    return samples


def extract_features(sample: CalibrationSample) -> np.ndarray:
    """
    Extract feature vector from calibration sample.
    
    Feature engineering is critical here. We include:
    1. Raw head pose angles
    2. Eye positions in frame
    3. Derived features (trigonometric transforms for nonlinearity)
    4. Inter-eye relationships
    """
    features = [
        # Head pose (raw)
        sample.yaw,
        sample.pitch,
        sample.roll,
        
        # Eye positions (raw)
        sample.left_eye_x,
        sample.left_eye_y,
        sample.right_eye_x,
        sample.right_eye_y,
        
        # Derived: Inter-pupillary relationships
        sample.right_eye_x - sample.left_eye_x,  # IPD (inter-pupillary distance)
        sample.right_eye_y - sample.left_eye_y,  # Eye tilt
        
        # Derived: Eye center
        (sample.left_eye_x + sample.right_eye_x) / 2,  # Face center X
        (sample.left_eye_y + sample.right_eye_y) / 2,  # Eye line Y
        
        # Derived: Trigonometric transforms (captures nonlinear relationships)
        np.sin(np.radians(sample.yaw)),
        np.cos(np.radians(sample.yaw)),
        np.sin(np.radians(sample.pitch)),
        np.cos(np.radians(sample.pitch)),
        
        # Interaction terms
        sample.yaw * sample.pitch,
        sample.yaw * (sample.right_eye_x - sample.left_eye_x),
    ]
    
    return np.array(features)


class GazeCalibrator:
    """
    SVR-based calibration layer that maps ONNX model outputs to screen coordinates.
    """
    
    def __init__(self, kernel='rbf', C=100.0):
        self.scaler = StandardScaler()
        self.model_x = SVR(kernel=kernel, C=C, gamma='scale')
        self.model_y = SVR(kernel=kernel, C=C, gamma='scale')
        self.is_fitted = False
        
    def fit(self, samples: List[CalibrationSample]) -> Dict[str, float]:
        """Train calibration models and return cross-validation metrics."""
        X = np.array([extract_features(s) for s in samples])
        y_x = np.array([s.screen_x for s in samples])
        y_y = np.array([s.screen_y for s in samples])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Leave-one-out cross-validation (best for small datasets)
        loo = LeaveOneOut()
        errors_x, errors_y, euclidean_errors = [], [], []
        
        print("\nRunning leave-one-out cross-validation...")
        
        for train_idx, test_idx in loo.split(X_scaled):
            # Train on all but one
            self.model_x.fit(X_scaled[train_idx], y_x[train_idx])
            self.model_y.fit(X_scaled[train_idx], y_y[train_idx])
            
            # Predict held-out sample
            pred_x = self.model_x.predict(X_scaled[test_idx])[0]
            pred_y = self.model_y.predict(X_scaled[test_idx])[0]
            
            # Calculate errors
            err_x = abs(pred_x - y_x[test_idx][0])
            err_y = abs(pred_y - y_y[test_idx][0])
            err_euclidean = np.sqrt(err_x**2 + err_y**2)
            
            errors_x.append(err_x)
            errors_y.append(err_y)
            euclidean_errors.append(err_euclidean)
        
        # Final fit on all data
        self.model_x.fit(X_scaled, y_x)
        self.model_y.fit(X_scaled, y_y)
        self.is_fitted = True
        
        metrics = {
            'mae_x': np.mean(errors_x),
            'mae_y': np.mean(errors_y),
            'mae_euclidean': np.mean(euclidean_errors),
            'median_euclidean': np.median(euclidean_errors),
            'std_euclidean': np.std(euclidean_errors),
            'max_euclidean': np.max(euclidean_errors),
            'p90_euclidean': np.percentile(euclidean_errors, 90),
        }
        
        return metrics
    
    def predict(self, sample: CalibrationSample) -> Tuple[float, float]:
        """Predict screen coordinates from sample."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = extract_features(sample).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        pred_x = self.model_x.predict(X_scaled)[0]
        pred_y = self.model_y.predict(X_scaled)[0]
        
        return pred_x, pred_y
    
    def predict_from_inference(self, inference_dict: Dict) -> Tuple[float, float]:
        """Predict from raw inference dictionary (for real-time use)."""
        gaze = inference_dict.get('Gaze', {})
        
        sample = CalibrationSample(
            timestamp=0,
            screen_x=0,
            screen_y=0,
            yaw=gaze.get('yaw', 0),
            pitch=gaze.get('pitch', 0),
            roll=gaze.get('roll', 0),
            left_eye_x=gaze.get('left_eye', {}).get('x', 0),
            left_eye_y=gaze.get('left_eye', {}).get('y', 0),
            right_eye_x=gaze.get('right_eye', {}).get('x', 0),
            right_eye_y=gaze.get('right_eye', {}).get('y', 0),
        )
        
        return self.predict(sample)
    
    def save(self, path: str):
        """Save calibration models."""
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'model_x': self.model_x,
            'model_y': self.model_y,
        }, path)
        print(f"Models saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'GazeCalibrator':
        """Load calibration models."""
        import joblib
        data = joblib.load(path)
        
        calibrator = cls()
        calibrator.scaler = data['scaler']
        calibrator.model_x = data['model_x']
        calibrator.model_y = data['model_y']
        calibrator.is_fitted = True
        
        return calibrator


def analyze_data_quality(samples: List[CalibrationSample]):
    """Analyze calibration data quality and distribution."""
    print("\n" + "="*60)
    print("CALIBRATION DATA ANALYSIS")
    print("="*60)
    
    # Screen coordinate distribution
    screen_xs = [s.screen_x for s in samples]
    screen_ys = [s.screen_y for s in samples]
    
    print(f"\nScreen X range: {min(screen_xs):.1f} - {max(screen_xs):.1f}")
    print(f"Screen Y range: {min(screen_ys):.1f} - {max(screen_ys):.1f}")
    
    # Head pose distribution
    yaws = [s.yaw for s in samples]
    pitches = [s.pitch for s in samples]
    
    print(f"\nYaw range: {min(yaws):.2f}° - {max(yaws):.2f}°")
    print(f"Pitch range: {min(pitches):.2f}° - {max(pitches):.2f}°")
    
    # Check for sufficient coverage
    coverage_warning = False
    
    x_range = max(screen_xs) - min(screen_xs)
    y_range = max(screen_ys) - min(screen_ys)
    
    if x_range < 500:
        print("\n⚠️  WARNING: Limited X coverage. Consider adding points across full screen width.")
        coverage_warning = True
    
    if y_range < 500:
        print("⚠️  WARNING: Limited Y coverage. Consider adding points across full screen height.")
        coverage_warning = True
    
    if len(samples) < 9:
        print(f"⚠️  WARNING: Only {len(samples)} samples. Recommend 9-25 for robust calibration.")
        coverage_warning = True
    
    if not coverage_warning:
        print("\n✓ Data coverage looks adequate for calibration")
    
    return {
        'screen_x_range': (min(screen_xs), max(screen_xs)),
        'screen_y_range': (min(screen_ys), max(screen_ys)),
        'yaw_range': (min(yaws), max(yaws)),
        'pitch_range': (min(pitches), max(pitches)),
        'n_samples': len(samples),
    }


def visualize_results(samples: List[CalibrationSample], calibrator: GazeCalibrator, save_path: str = None):
    """Create visualization of calibration results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Ground truth vs predicted scatter
    ax1 = axes[0, 0]
    gt_x = [s.screen_x for s in samples]
    gt_y = [s.screen_y for s in samples]
    
    pred_x, pred_y = [], []
    for s in samples:
        px, py = calibrator.predict(s)
        pred_x.append(px)
        pred_y.append(py)
    
    ax1.scatter(gt_x, gt_y, c='blue', label='Ground Truth', s=100, alpha=0.7)
    ax1.scatter(pred_x, pred_y, c='red', label='Predicted', s=100, alpha=0.7, marker='x')
    
    # Draw lines connecting GT to prediction
    for gx, gy, px, py in zip(gt_x, gt_y, pred_x, pred_y):
        ax1.plot([gx, px], [gy, py], 'k-', alpha=0.3)
    
    ax1.set_xlabel('Screen X (pixels)')
    ax1.set_ylabel('Screen Y (pixels)')
    ax1.set_title('Ground Truth vs Predicted Gaze Points')
    ax1.legend()
    ax1.invert_yaxis()  # Screen coordinates have Y increasing downward
    ax1.set_aspect('equal')
    
    # 2. Error distribution histogram
    ax2 = axes[0, 1]
    errors = [np.sqrt((gx-px)**2 + (gy-py)**2) for gx, gy, px, py in zip(gt_x, gt_y, pred_x, pred_y)]
    
    ax2.hist(errors, bins=min(15, len(errors)), edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}px')
    ax2.axvline(np.median(errors), color='green', linestyle='--', label=f'Median: {np.median(errors):.1f}px')
    ax2.set_xlabel('Euclidean Error (pixels)')
    ax2.set_ylabel('Count')
    ax2.set_title('Error Distribution')
    ax2.legend()
    
    # 3. Error by screen position
    ax3 = axes[1, 0]
    scatter = ax3.scatter(gt_x, gt_y, c=errors, cmap='RdYlGn_r', s=200, alpha=0.8)
    plt.colorbar(scatter, ax=ax3, label='Error (pixels)')
    ax3.set_xlabel('Screen X (pixels)')
    ax3.set_ylabel('Screen Y (pixels)')
    ax3.set_title('Error by Screen Position')
    ax3.invert_yaxis()
    
    # 4. Feature importance (approximate via coefficient magnitude for scaled features)
    ax4 = axes[1, 1]
    
    feature_names = [
        'yaw', 'pitch', 'roll', 
        'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
        'ipd', 'eye_tilt', 'face_center_x', 'eye_line_y',
        'sin(yaw)', 'cos(yaw)', 'sin(pitch)', 'cos(pitch)',
        'yaw*pitch', 'yaw*ipd'
    ]
    
    # Get support vector coefficients as proxy for importance
    X = np.array([extract_features(s) for s in samples])
    X_scaled = calibrator.scaler.transform(X)
    
    # Use variance of scaled features as simple importance proxy
    feature_variance = np.var(X_scaled, axis=0)
    
    y_pos = np.arange(len(feature_names))
    ax4.barh(y_pos, feature_variance, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(feature_names)
    ax4.set_xlabel('Feature Variance (scaled)')
    ax4.set_title('Feature Variance (higher = more informative)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def print_metrics(metrics: Dict[str, float]):
    """Pretty print evaluation metrics."""
    print("\n" + "="*60)
    print("CALIBRATION RESULTS (Leave-One-Out Cross-Validation)")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'MAE X:':<25} {metrics['mae_x']:.2f} px")
    print(f"{'MAE Y:':<25} {metrics['mae_y']:.2f} px")
    print(f"{'MAE Euclidean:':<25} {metrics['mae_euclidean']:.2f} px")
    print(f"{'Median Euclidean:':<25} {metrics['median_euclidean']:.2f} px")
    print(f"{'Std Euclidean:':<25} {metrics['std_euclidean']:.2f} px")
    print(f"{'Max Error:':<25} {metrics['max_euclidean']:.2f} px")
    print(f"{'90th Percentile:':<25} {metrics['p90_euclidean']:.2f} px")
    
    # Quality assessment
    print("\n" + "-"*40)
    mae = metrics['mae_euclidean']
    
    if mae < 30:
        print("✓ EXCELLENT: Error < 30px - Production ready")
    elif mae < 50:
        print("✓ GOOD: Error < 50px - Usable for most applications")
    elif mae < 100:
        print("◐ FAIR: Error < 100px - Consider L2CS-Net upgrade")
    else:
        print("✗ POOR: Error > 100px - Need better gaze features")
        print("  Recommendation: Implement L2CS-Net (Phase 2)")


def main():
    parser = argparse.ArgumentParser(description='Gaze Calibration Training')
    parser.add_argument('--calibration-dir', type=str, default='./calibration_data',
                        help='Directory containing calibration images and JSON files')
    parser.add_argument('--output-model', type=str, default='./gaze_calibration.pkl',
                        help='Path to save trained calibration model')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of results')
    parser.add_argument('--viz-output', type=str, default='./calibration_results.png',
                        help='Path to save visualization')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear', 'poly'],
                        help='SVR kernel type')
    parser.add_argument('--C', type=float, default=100.0,
                        help='SVR regularization parameter')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nLoading calibration data from: {args.calibration_dir}")
    samples = load_calibration_data(args.calibration_dir)
    
    if len(samples) == 0:
        print("ERROR: No valid calibration samples found!")
        return
    
    # Analyze data quality
    analyze_data_quality(samples)
    
    # Train calibration model
    print(f"\nTraining calibration model (kernel={args.kernel}, C={args.C})...")
    calibrator = GazeCalibrator(kernel=args.kernel, C=args.C)
    metrics = calibrator.fit(samples)
    
    # Print results
    print_metrics(metrics)
    
    # Save model
    calibrator.save(args.output_model)
    
    # Visualize if requested
    if args.visualize:
        visualize_results(samples, calibrator, args.viz_output)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
1. If MAE < 50px: Success! Integrate into your real-time pipeline
   
2. If MAE > 50px: Consider these improvements:
   - Collect more calibration points (aim for 25+)
   - Include varied head poses while looking at same point
   - Upgrade to L2CS-Net for appearance-based gaze (see spec)

3. For real-time use, load the model and call predict_from_inference():
   
   calibrator = GazeCalibrator.load('./gaze_calibration.pkl')
   screen_x, screen_y = calibrator.predict_from_inference(inference_dict)
   
4. Add temporal smoothing (Kalman filter) for stable output
""")


if __name__ == '__main__':
    main()