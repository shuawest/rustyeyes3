# Precision Gaze Estimation System

## DGX Spark Training → MacBook Pro Inference Architecture

**Version:** 2.0  
**Target:** 60Hz+ real-time inference, screen-size adaptive, personal→multi-user scaling

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING (DGX Spark GB10)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  ETH-XGaze /     │    │  Personal        │    │  MobileNetV2     │      │
│  │  Gaze360 Data    │───▶│  Calibration     │───▶│  Gaze Model      │      │
│  │  (1M+ images)    │    │  Fine-tuning     │    │  (Export ONNX)   │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                                                           │                 │
│                                                           ▼                 │
│                                                  ┌──────────────────┐      │
│                                                  │  Calibration     │      │
│                                                  │  MLP Layer       │      │
│                                                  │  (Per-User)      │      │
│                                                  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ ONNX Export
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INFERENCE (MacBook Pro M-series)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐        │
│  │  Webcam    │   │  Face/Eye  │   │  Gaze      │   │ Calibration│        │
│  │  Frame     │──▶│  Detection │──▶│  Model     │──▶│ + Smooth   │──▶ X,Y │
│  │  (60 FPS)  │   │  (PINTO)   │   │  (ONNX)    │   │ (Kalman)   │        │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘        │
│       │                                                                     │
│       │ Async (every Nth frame)                                            │
│       ▼                                                                     │
│  ┌────────────┐                                                            │
│  │ Moondream2 │──▶ Confidence scoring / Drift detection / Recalibration   │
│  └────────────┘                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Screen-Size Adaptive Design

### 2.1 Normalized Coordinate System

All models output **normalized coordinates [0, 1]**, not pixel values:

```python
# Training targets
normalized_x = screen_x / screen_width   # 0.0 = left edge, 1.0 = right edge
normalized_y = screen_y / screen_height  # 0.0 = top edge, 1.0 = bottom edge

# Inference conversion (any screen size)
pixel_x = normalized_x * current_screen_width
pixel_y = normalized_y * current_screen_height
```

### 2.2 Aspect Ratio Handling

Include aspect ratio as model input to handle different screen shapes:

```python
aspect_ratio = screen_width / screen_height  # e.g., 16/9 = 1.78, 16/10 = 1.6

# Feature vector includes aspect ratio
features = [gaze_pitch, gaze_yaw, head_pose..., aspect_ratio]
```

### 2.3 Camera Position Compensation

Camera offset from screen center affects mapping:

```python
@dataclass
class ScreenConfig:
    width_px: int
    height_px: int
    width_cm: float  # Physical width for distance calculations
    camera_offset_x: float  # Camera X offset from screen center (cm)
    camera_offset_y: float  # Camera Y offset (typically above screen)

    @property
    def aspect_ratio(self):
        return self.width_px / self.height_px
```

---

## 3. Model Architecture for 60Hz+ Inference

### 3.1 Lightweight Gaze Model (Primary)

**Architecture:** MobileNetV2 backbone + Gaze head
**Target latency:** <8ms on M-series ANE

```
Input: 224x224 RGB face crop
       ↓
MobileNetV2 (width_mult=0.5)  # Reduced width for speed
       ↓
Global Average Pool
       ↓
FC(512) → ReLU → Dropout(0.5)
       ↓
FC(2) → [pitch, yaw]  # Gaze angles in radians
```

**Why MobileNetV2:**

- Optimized for Apple Neural Engine (ANE)
- Inverted residuals = efficient on mobile
- 0.5x width = ~1.5M params, <5ms inference

### 3.2 Alternative: EfficientNet-B0 (Higher accuracy)

```
Input: 224x224 RGB
       ↓
EfficientNet-B0 (5.3M params)
       ↓
FC(256) → FC(2)

Latency: ~10ms on ANE (still 100Hz capable)
```

### 3.3 Screen Calibration Layer

Lightweight MLP that runs after gaze model:

```
Input: [pitch, yaw, head_yaw, head_pitch, head_roll, eye_x, eye_y, aspect_ratio]
       ↓
FC(64) → ReLU
       ↓
FC(32) → ReLU
       ↓
FC(2) → Sigmoid → [norm_x, norm_y]  # Normalized screen coords
```

**Latency:** <0.5ms (negligible)

---

## 4. Training Pipeline (DGX Spark)

### 4.1 Phase 1: Base Gaze Model Training

Train on large-scale gaze datasets to learn general gaze estimation:

```python
# datasets.py - Data loading for DGX Spark training

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class ETHXGazeDataset(Dataset):
    """
    ETH-XGaze: 1M+ images, extreme head poses
    Download: https://ait.ethz.ch/xgaze
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['face_patch'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            face = f['face_patch'][idx]  # 224x224x3
            gaze = f['face_gaze'][idx]   # [pitch, yaw] in radians
            head_pose = f['face_head_pose'][idx]  # [pitch, yaw] head

        if self.transform:
            face = self.transform(face)

        return {
            'face': torch.FloatTensor(face).permute(2, 0, 1) / 255.0,
            'gaze': torch.FloatTensor(gaze),
            'head_pose': torch.FloatTensor(head_pose),
        }


class Gaze360Dataset(Dataset):
    """
    Gaze360: 238K images, full 360° gaze range
    Download: http://gaze360.csail.mit.edu/
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Load annotations...


class CombinedGazeDataset(Dataset):
    """Combine multiple datasets for robust training."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumulative = np.cumsum([0] + self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(zip(self.cumulative[:-1], self.cumulative[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]
```

### 4.2 Model Definition

```python
# models.py - Lightweight gaze model for ANE deployment

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class LightweightGazeNet(nn.Module):
    """
    MobileNetV2-based gaze estimation model.
    Optimized for Apple Neural Engine deployment.
    """
    def __init__(self, pretrained=True, width_mult=0.5):
        super().__init__()

        # Backbone: MobileNetV2 with reduced width
        if pretrained:
            # Start from ImageNet weights, then slim down
            full_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            # Extract features (remove classifier)
            self.features = full_model.features
        else:
            from torchvision.models.mobilenetv2 import MobileNetV2
            backbone = MobileNetV2(width_mult=width_mult)
            self.features = backbone.features

        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.features(dummy).shape[1]

        # Gaze estimation head
        self.gaze_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),  # [pitch, yaw]
        )

    def forward(self, x):
        features = self.features(x)
        gaze = self.gaze_head(features)
        return gaze  # [batch, 2] - pitch, yaw in radians

    def export_onnx(self, path, input_shape=(1, 3, 224, 224)):
        """Export to ONNX for cross-platform deployment."""
        self.eval()
        dummy_input = torch.randn(*input_shape)

        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=['face_image'],
            output_names=['gaze_angles'],
            dynamic_axes={
                'face_image': {0: 'batch'},
                'gaze_angles': {0: 'batch'},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"Exported ONNX model to {path}")


class ScreenCalibrationMLP(nn.Module):
    """
    Per-user calibration layer: gaze angles → screen coordinates.
    Lightweight enough to retrain on-device if needed.
    """
    def __init__(self, input_dim=8):
        super().__init__()

        # Input: [pitch, yaw, head_pitch, head_yaw, head_roll,
        #         face_center_x, face_center_y, aspect_ratio]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),  # Output in [0, 1] for normalized coords
        )

    def forward(self, x):
        return self.mlp(x)
```

### 4.3 Training Script for DGX Spark

```python
# train_dgx.py - Training script optimized for DGX Spark GB10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from pathlib import Path

from models import LightweightGazeNet
from datasets import ETHXGazeDataset, Gaze360Dataset, CombinedGazeDataset


def angular_loss(pred, target):
    """
    Angular error loss in radians.
    More meaningful than MSE for gaze estimation.
    """
    # Convert to 3D vectors
    pred_vec = angles_to_vector(pred)
    target_vec = angles_to_vector(target)

    # Cosine similarity → angular error
    cos_sim = torch.sum(pred_vec * target_vec, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angular_error = torch.acos(cos_sim)

    return angular_error.mean()


def angles_to_vector(angles):
    """Convert pitch/yaw to unit gaze vector."""
    pitch, yaw = angles[:, 0], angles[:, 1]

    x = -torch.sin(yaw) * torch.cos(pitch)
    y = -torch.sin(pitch)
    z = torch.cos(yaw) * torch.cos(pitch)

    return torch.stack([x, y, z], dim=1)


def train_base_model(config):
    """
    Phase 1: Train base gaze model on large-scale datasets.

    DGX Spark GB10 optimizations:
    - Mixed precision (FP16/BF16)
    - Multi-GPU DataParallel
    - Large batch sizes (leverage 128GB+ VRAM)
    """

    # Initialize wandb for experiment tracking
    wandb.init(project="gaze-estimation", config=config)

    device = torch.device('cuda')

    # Model
    model = LightweightGazeNet(pretrained=True, width_mult=config['width_mult'])

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datasets
    eth_xgaze = ETHXGazeDataset(config['eth_xgaze_path'])
    gaze360 = Gaze360Dataset(config['gaze360_path'])

    train_dataset = CombinedGazeDataset([eth_xgaze, gaze360])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],  # Large batch: 256-512 on DGX
        shuffle=True,
        num_workers=16,  # DGX has many cores
        pin_memory=True,
        prefetch_factor=4,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
    )

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            face = batch['face'].to(device)
            gaze_target = batch['gaze'].to(device)

            optimizer.zero_grad()

            # Mixed precision forward
            with autocast():
                gaze_pred = model(face)
                loss = angular_loss(gaze_pred, gaze_target)

            # Backward with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                wandb.log({
                    'batch_loss': loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                })

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        wandb.log({'epoch_loss': avg_loss, 'epoch': epoch})

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = Path(config['output_dir']) / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    # Export final model to ONNX
    model_to_export = model.module if hasattr(model, 'module') else model
    model_to_export.export_onnx(Path(config['output_dir']) / "gaze_model.onnx")

    return model


if __name__ == "__main__":
    config = {
        'eth_xgaze_path': '/data/eth_xgaze/train.h5',
        'gaze360_path': '/data/gaze360/',
        'output_dir': '/output/gaze_model/',

        'width_mult': 0.5,  # MobileNetV2 width multiplier
        'batch_size': 512,  # Large batch for DGX
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 50,
        'save_every': 5,
    }

    train_base_model(config)
```

### 4.4 Phase 2: Personal Calibration Fine-tuning

```python
# finetune_personal.py - Fine-tune for specific user

import torch
import torch.nn as nn
import json
import glob
from PIL import Image
import torchvision.transforms as T

from models import LightweightGazeNet, ScreenCalibrationMLP


class PersonalCalibrationDataset(torch.utils.data.Dataset):
    """Load personal calibration data (your format)."""

    def __init__(self, calibration_dir, screen_width=1920, screen_height=1080, transform=None):
        self.samples = []
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for json_path in sorted(glob.glob(f"{calibration_dir}/*.json")):
            with open(json_path) as f:
                data = json.load(f)

            img_path = json_path.replace('.json', '.jpg')

            self.samples.append({
                'image_path': img_path,
                'screen_x': data['screen_x'],
                'screen_y': data['screen_y'],
                'inference': data.get('inference', {}),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)

        # Normalized screen coordinates (0-1)
        norm_x = sample['screen_x'] / self.screen_w
        norm_y = sample['screen_y'] / self.screen_h

        # Head pose from existing inference
        gaze_data = sample['inference'].get('Gaze', {})
        head_pose = torch.FloatTensor([
            gaze_data.get('yaw', 0),
            gaze_data.get('pitch', 0),
            gaze_data.get('roll', 0),
        ])

        return {
            'image': image,
            'target': torch.FloatTensor([norm_x, norm_y]),
            'head_pose': head_pose,
        }


def finetune_for_user(base_model_path, calibration_dir, output_dir, config):
    """
    Fine-tune base model + train calibration layer for specific user.

    Strategy:
    1. Freeze most of base model, fine-tune last layers
    2. Train calibration MLP from scratch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    # Load base model
    base_model = LightweightGazeNet(pretrained=False)
    checkpoint = torch.load(base_model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze backbone, unfreeze head
    for param in base_model.features.parameters():
        param.requires_grad = False
    for param in base_model.gaze_head.parameters():
        param.requires_grad = True

    base_model = base_model.to(device)

    # Calibration MLP
    calibration_mlp = ScreenCalibrationMLP(input_dim=8).to(device)

    # Dataset
    dataset = PersonalCalibrationDataset(
        calibration_dir,
        screen_width=config['screen_width'],
        screen_height=config['screen_height'],
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(len(dataset), 8),
        shuffle=True,
    )

    # Combined forward pass
    def forward_combined(images, head_pose, aspect_ratio):
        # Get gaze angles from base model
        gaze_angles = base_model(images)  # [batch, 2] - pitch, yaw

        # Build calibration input
        batch_size = images.shape[0]
        aspect = torch.full((batch_size, 1), aspect_ratio, device=device)

        calib_input = torch.cat([
            gaze_angles,          # pitch, yaw
            head_pose,            # head yaw, pitch, roll
            torch.zeros(batch_size, 2, device=device),  # placeholder for face position
            aspect,
        ], dim=1)

        # Get screen coordinates
        screen_coords = calibration_mlp(calib_input)

        return screen_coords, gaze_angles

    # Optimizer for both models
    optimizer = torch.optim.Adam([
        {'params': base_model.gaze_head.parameters(), 'lr': 1e-5},
        {'params': calibration_mlp.parameters(), 'lr': 1e-3},
    ])

    aspect_ratio = config['screen_width'] / config['screen_height']

    # Training loop
    for epoch in range(config['epochs']):
        total_loss = 0

        for batch in loader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            head_pose = batch['head_pose'].to(device)

            optimizer.zero_grad()

            pred_coords, gaze_angles = forward_combined(images, head_pose, aspect_ratio)

            # MSE loss on normalized coordinates
            loss = nn.functional.mse_loss(pred_coords, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # Convert to pixel error for interpretability
        pixel_error_x = avg_loss ** 0.5 * config['screen_width']
        pixel_error_y = avg_loss ** 0.5 * config['screen_height']

        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f}, ~Error={pixel_error_x:.1f}px x {pixel_error_y:.1f}px")

    # Save models
    torch.save(base_model.state_dict(), f"{output_dir}/gaze_model_finetuned.pt")
    torch.save(calibration_mlp.state_dict(), f"{output_dir}/calibration_mlp.pt")

    # Export to ONNX
    base_model.export_onnx(f"{output_dir}/gaze_model_finetuned.onnx")

    # Export calibration MLP to ONNX
    calibration_mlp.eval()
    dummy_input = torch.randn(1, 8)
    torch.onnx.export(
        calibration_mlp,
        dummy_input.to(device),
        f"{output_dir}/calibration_mlp.onnx",
        input_names=['features'],
        output_names=['screen_coords'],
        opset_version=17,
    )

    print(f"\nModels saved to {output_dir}")


if __name__ == "__main__":
    config = {
        'screen_width': 1920,
        'screen_height': 1080,
        'epochs': 100,
    }

    finetune_for_user(
        base_model_path='/output/gaze_model/checkpoint_epoch50.pt',
        calibration_dir='./calibration_data',
        output_dir='./personal_model',
        config=config,
    )
```

---

## 5. Inference Pipeline (MacBook Pro)

### 5.1 Real-time Pipeline (60Hz+)

```python
# inference_realtime.py - MacBook Pro optimized inference

import numpy as np
import onnxruntime as ort
import cv2
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class ScreenConfig:
    """Screen configuration for coordinate mapping."""
    width_px: int
    height_px: int

    @property
    def aspect_ratio(self):
        return self.width_px / self.height_px


class KalmanSmoother:
    """1D Kalman filter for coordinate smoothing."""

    def __init__(self, process_noise=0.01, measurement_noise=1.0):
        self.q = process_noise
        self.r = measurement_noise
        self.x = 0.0  # State estimate
        self.p = 1.0  # Estimate covariance
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Predict
        self.p += self.q

        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)

        return self.x

    def reset(self):
        self.initialized = False


class RealtimeGazeEstimator:
    """
    Real-time gaze estimation pipeline.
    Target: 60Hz+ on MacBook Pro M-series.
    """

    def __init__(
        self,
        face_detector_path: str,      # PINTO zoo face detection ONNX
        gaze_model_path: str,          # Trained gaze model ONNX
        calibration_path: str,         # Personal calibration ONNX
        screen_config: ScreenConfig,
        use_coreml: bool = True,       # Use CoreML for Apple Neural Engine
    ):
        self.screen = screen_config

        # Select execution provider for Apple Silicon
        if use_coreml:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Load models
        self.face_detector = ort.InferenceSession(face_detector_path, providers=providers)
        self.gaze_model = ort.InferenceSession(gaze_model_path, providers=providers)
        self.calibration = ort.InferenceSession(calibration_path, providers=providers)

        # Smoothing
        self.smoother_x = KalmanSmoother(process_noise=0.005, measurement_noise=0.5)
        self.smoother_y = KalmanSmoother(process_noise=0.005, measurement_noise=0.5)

        # Performance tracking
        self.frame_times = deque(maxlen=60)

        # Input specs
        self.gaze_input_name = self.gaze_model.get_inputs()[0].name
        self.gaze_input_shape = self.gaze_model.get_inputs()[0].shape
        self.calib_input_name = self.calibration.get_inputs()[0].name

    def preprocess_face(self, frame, bbox):
        """Crop and preprocess face for gaze model."""
        x1, y1, x2, y2 = map(int, bbox)

        # Add margin
        w, h = x2 - x1, y2 - y1
        margin = 0.2
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(frame.shape[1], int(x2 + w * margin))
        y2 = min(frame.shape[0], int(y2 + h * margin))

        # Crop and resize
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))

        # Normalize (ImageNet stats)
        face = face.astype(np.float32) / 255.0
        face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        face = face.transpose(2, 0, 1)  # HWC → CHW
        face = np.expand_dims(face, 0)  # Add batch dim

        return face.astype(np.float32)

    def detect_face(self, frame):
        """Run face detection (using your existing PINTO model)."""
        # This should match your existing face detection preprocessing
        # Returns: bbox [x1, y1, x2, y2], head_pose [yaw, pitch, roll]

        # Placeholder - replace with your actual detection
        h, w = frame.shape[:2]

        # Your PINTO model inference here
        # ...

        # Return format
        return {
            'bbox': [w*0.3, h*0.2, w*0.7, h*0.8],  # Face bbox
            'head_pose': [0.0, 0.0, 0.0],           # [yaw, pitch, roll]
            'face_center': [w/2, h*0.4],            # Face center in frame
        }

    def estimate_gaze(self, frame) -> dict:
        """
        Full pipeline: frame → screen coordinates.
        Returns dict with x, y in pixels and metadata.
        """
        start_time = time.perf_counter()

        # Step 1: Face detection
        detection = self.detect_face(frame)

        if detection is None:
            return None

        # Step 2: Gaze angle estimation
        face_input = self.preprocess_face(frame, detection['bbox'])
        gaze_output = self.gaze_model.run(None, {self.gaze_input_name: face_input})[0]

        pitch, yaw = gaze_output[0]  # Radians

        # Step 3: Calibration mapping
        # Build feature vector: [pitch, yaw, head_yaw, head_pitch, head_roll, face_x, face_y, aspect]
        calib_features = np.array([[
            pitch,
            yaw,
            detection['head_pose'][0],  # head yaw
            detection['head_pose'][1],  # head pitch
            detection['head_pose'][2],  # head roll
            detection['face_center'][0] / frame.shape[1],  # normalized face x
            detection['face_center'][1] / frame.shape[0],  # normalized face y
            self.screen.aspect_ratio,
        ]], dtype=np.float32)

        screen_coords = self.calibration.run(None, {self.calib_input_name: calib_features})[0]

        # Denormalize to pixels
        raw_x = screen_coords[0, 0] * self.screen.width_px
        raw_y = screen_coords[0, 1] * self.screen.height_px

        # Step 4: Kalman smoothing
        smooth_x = self.smoother_x.update(raw_x)
        smooth_y = self.smoother_y.update(raw_y)

        # Track performance
        elapsed = time.perf_counter() - start_time
        self.frame_times.append(elapsed)

        return {
            'x': smooth_x,
            'y': smooth_y,
            'raw_x': raw_x,
            'raw_y': raw_y,
            'gaze_pitch': float(pitch),
            'gaze_yaw': float(yaw),
            'latency_ms': elapsed * 1000,
            'fps': 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0,
        }

    def reset_smoothing(self):
        """Reset Kalman filters (e.g., after detected jump)."""
        self.smoother_x.reset()
        self.smoother_y.reset()


class AsyncMoondreamValidator:
    """
    Run Moondream2 asynchronously for validation/drift detection.
    Doesn't block real-time pipeline.
    """

    def __init__(self, model_path: str, check_interval: int = 30):
        """
        Args:
            model_path: Path to Moondream2 model
            check_interval: Run validation every N frames
        """
        self.check_interval = check_interval
        self.frame_count = 0
        self.last_validation = None

        # Load Moondream2 (heavy model, runs async)
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = ...

    def should_validate(self) -> bool:
        """Check if we should run validation this frame."""
        self.frame_count += 1
        return self.frame_count % self.check_interval == 0

    def validate_async(self, frame, gaze_estimate):
        """
        Run Moondream2 validation in background thread.

        Use cases:
        1. Confidence scoring - does VLM agree with estimate?
        2. Drift detection - has calibration degraded?
        3. Trigger recalibration if needed
        """
        import threading

        def _validate():
            # Query Moondream2 about gaze target
            # Compare with real-time estimate
            # Flag if significant disagreement
            pass

        thread = threading.Thread(target=_validate)
        thread.start()
```

### 5.2 Integration with Your Existing App

```python
# integration_example.py - How to integrate with your existing harness

class GazeTrackingApp:
    """Example integration with your existing Rust app via FFI or subprocess."""

    def __init__(self, screen_width=1920, screen_height=1080):
        # Initialize real-time pipeline
        self.gaze_estimator = RealtimeGazeEstimator(
            face_detector_path='models/face_detector.onnx',  # Your existing PINTO model
            gaze_model_path='models/gaze_model_finetuned.onnx',
            calibration_path='models/calibration_mlp.onnx',
            screen_config=ScreenConfig(screen_width, screen_height),
            use_coreml=True,
        )

        # Async validator
        self.validator = AsyncMoondreamValidator(
            model_path='models/moondream2',
            check_interval=30,
        )

        # Screen change detection
        self._current_screen = (screen_width, screen_height)

    def on_screen_change(self, new_width, new_height):
        """Handle screen resolution change."""
        self.gaze_estimator.screen = ScreenConfig(new_width, new_height)
        self.gaze_estimator.reset_smoothing()
        self._current_screen = (new_width, new_height)

    def process_frame(self, frame):
        """
        Main processing function - call this at 60Hz.
        Returns (x, y) screen coordinates or None.
        """
        result = self.gaze_estimator.estimate_gaze(frame)

        if result is None:
            return None

        # Async validation (non-blocking)
        if self.validator.should_validate():
            self.validator.validate_async(frame, result)

        return result['x'], result['y']


# For Rust FFI (if your app is in Rust)
# Export functions that can be called from Rust via ctypes/cffi

def create_gaze_estimator(screen_width: int, screen_height: int) -> int:
    """Create estimator, return handle."""
    # ... implementation for FFI
    pass

def estimate_gaze(handle: int, frame_ptr: int, width: int, height: int) -> tuple:
    """Process frame, return (x, y) or (-1, -1) if no face."""
    # ... implementation for FFI
    pass
```

---

## 6. Data Collection Strategy for Multi-User

### 6.1 Calibration Protocol

```python
# calibration_collector.py - Systematic calibration data collection

import cv2
import numpy as np
import json
import time
from pathlib import Path


class CalibrationCollector:
    """Collect calibration data with systematic coverage."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        output_dir: str,
        grid_size: int = 5,  # 5x5 = 25 points
        samples_per_point: int = 3,
        include_head_variations: bool = True,
    ):
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.grid_size = grid_size
        self.samples_per_point = samples_per_point
        self.include_head_variations = include_head_variations

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.points = self._generate_calibration_points()

    def _generate_calibration_points(self):
        """Generate calibration point grid."""
        points = []
        margin_x = self.screen_w * 0.1
        margin_y = self.screen_h * 0.1

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = margin_x + (self.screen_w - 2*margin_x) * i / (self.grid_size - 1)
                y = margin_y + (self.screen_h - 2*margin_y) * j / (self.grid_size - 1)
                points.append((int(x), int(y)))

        return points

    def collect(self):
        """Run interactive calibration collection."""
        cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        head_poses = ["center"]
        if self.include_head_variations:
            head_poses = ["center", "left", "right", "up", "down"]

        sample_idx = 0

        for point_idx, (target_x, target_y) in enumerate(self.points):
            for head_pose in head_poses:
                for sample in range(self.samples_per_point):
                    # Display target
                    screen = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                    cv2.circle(screen, (target_x, target_y), 25, (0, 255, 0), -1)
                    cv2.circle(screen, (target_x, target_y), 5, (255, 255, 255), -1)

                    # Instructions
                    instruction = f"Look at the green dot"
                    if head_pose != "center":
                        instruction += f"\nHead turned {head_pose}, eyes on dot"

                    cv2.putText(screen, instruction, (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(screen, f"Point {point_idx+1}/{len(self.points)}",
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.putText(screen, "Press SPACE when fixating, ESC to quit",
                               (50, self.screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                    cv2.imshow("Calibration", screen)

                    # Wait for keypress
                    while True:
                        key = cv2.waitKey(1) & 0xFF

                        if key == 27:  # ESC
                            cv2.destroyAllWindows()
                            return
                        elif key == 32:  # SPACE
                            # Capture frame
                            ret, frame = self.cap.read()
                            if ret:
                                self._save_sample(
                                    frame,
                                    target_x,
                                    target_y,
                                    head_pose,
                                    sample_idx
                                )
                                sample_idx += 1
                            break

        cv2.destroyAllWindows()
        self.cap.release()
        print(f"\nCalibration complete! Collected {sample_idx} samples.")

    def _save_sample(self, frame, target_x, target_y, head_pose, idx):
        """Save captured sample."""
        timestamp = int(time.time() * 1000)

        # Save image
        img_path = self.output_dir / f"img_{timestamp}.jpg"
        cv2.imwrite(str(img_path), frame)

        # Save metadata
        json_path = self.output_dir / f"img_{timestamp}.json"
        metadata = {
            'timestamp': timestamp,
            'screen_x': target_x,
            'screen_y': target_y,
            'screen_width': self.screen_w,
            'screen_height': self.screen_h,
            'head_pose_instruction': head_pose,
            'sample_index': idx,
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved sample {idx}: ({target_x}, {target_y}), head: {head_pose}")
```

---

## 7. Performance Benchmarks

### Expected Performance on MacBook Pro

| Component                     | M1 Pro    | M2 Pro     | M3 Pro     |
| ----------------------------- | --------- | ---------- | ---------- |
| Face Detection (PINTO)        | ~5ms      | ~4ms       | ~3ms       |
| Gaze Model (MobileNetV2 0.5x) | ~6ms      | ~5ms       | ~4ms       |
| Calibration MLP               | <1ms      | <1ms       | <1ms       |
| **Total Pipeline**            | **~12ms** | **~10ms**  | **~8ms**   |
| **Max FPS**                   | **~83Hz** | **~100Hz** | **~125Hz** |

With ANE (Apple Neural Engine) acceleration, expect 20-40% improvement.

### Accuracy Targets

| Metric          | Acceptable | Good  | Excellent |
| --------------- | ---------- | ----- | --------- |
| MAE (pixels)    | <80px      | <40px | <20px     |
| MAE (degrees)   | <2.5°      | <1.5° | <0.8°     |
| 90th percentile | <120px     | <60px | <35px     |

---

## 8. Summary: Your Implementation Roadmap

### Week 1: Base Model Training (DGX Spark)

1. **Download datasets:**

   - ETH-XGaze (apply for access)
   - Gaze360 (public)

2. **Train base MobileNetV2 gaze model:**

   ```bash
   # On DGX Spark
   python train_dgx.py --config configs/base_training.yaml
   ```

3. **Export to ONNX**

### Week 2: Personal Calibration

1. **Collect calibration data** (25+ points with head variations)

   ```bash
   python calibration_collector.py --grid-size 5 --samples-per-point 3
   ```

2. **Fine-tune + train calibration layer:**

   ```bash
   python finetune_personal.py --calibration-dir ./calibration_data
   ```

3. **Export models to ONNX**

### Week 3: Integration & Testing

1. **Integrate with your existing Rust app**
2. **Benchmark latency on MacBook Pro**
3. **Tune Kalman smoothing parameters**
4. **Setup Moondream2 async validation**

### Future: Multi-User Scaling

1. **Collect data from multiple users**
2. **Train person-independent base model**
3. **Implement few-shot calibration (3-9 points per new user)**
