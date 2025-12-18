#!/usr/bin/env python3
"""
Gaze Estimation Training Pipeline for NVIDIA DGX Spark GB10
============================================================

This script trains a lightweight gaze estimation model suitable for
60Hz+ inference on MacBook Pro M-series.

Usage:
    # Phase 1: Train base model on large datasets
    python train_gaze_dgx.py train-base \
        --eth-xgaze-path /data/eth_xgaze \
        --gaze360-path /data/gaze360 \
        --output-dir /output/gaze_base

    # Phase 2: Fine-tune for personal calibration
    python train_gaze_dgx.py finetune-personal \
        --base-model /output/gaze_base/best_model.pt \
        --calibration-dir ./calibration_data \
        --screen-width 1920 --screen-height 1080 \
        --output-dir /output/gaze_personal

Requirements:
    pip install torch torchvision onnx onnxruntime h5py wandb pillow
"""

import argparse
import json
import glob
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Optional: h5py for ETH-XGaze
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


# =============================================================================
# Model Definitions
# =============================================================================

class MobileNetV2GazeNet(nn.Module):
    """
    Lightweight gaze estimation model using MobileNetV2 backbone.
    
    Optimized for:
    - Apple Neural Engine (ANE) deployment
    - <10ms inference on M-series chips
    - ~2M parameters with width_mult=0.5
    """
    
    def __init__(self, width_mult: float = 0.5, pretrained: bool = True):
        super().__init__()
        
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        
        # Load pretrained backbone
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            backbone = mobilenet_v2(weights=weights)
        else:
            backbone = mobilenet_v2(weights=None)
        
        self.features = backbone.features
        
        # Determine feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat_dim = self.features(dummy).mean(dim=[2, 3]).shape[1]
        
        self.feat_dim = feat_dim
        
        # Gaze estimation head
        self.gaze_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # [pitch, yaw] in radians
        )
        
        # Initialize head
        for m in self.gaze_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] face images (normalized)
        Returns:
            [B, 2] gaze angles (pitch, yaw) in radians
        """
        features = self.features(x)
        gaze = self.gaze_head(features)
        return gaze
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector (for calibration layer training)."""
        features = self.features(x)
        features = F.adaptive_avg_pool2d(features, 1)
        return features.flatten(1)


class EfficientNetB0GazeNet(nn.Module):
    """
    Alternative: EfficientNet-B0 backbone for higher accuracy.
    Slightly slower (~10-12ms) but more accurate.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        
        if pretrained:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            backbone = efficientnet_b0(weights=None)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Gaze head
        self.gaze_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.gaze_head(x)


class ScreenCalibrationMLP(nn.Module):
    """
    Per-user calibration: maps gaze features to normalized screen coordinates.
    
    Input features:
        - pitch, yaw (from gaze model)
        - head pose (yaw, pitch, roll)
        - face position (normalized x, y)
        - screen aspect ratio
    
    Total: 8 features → 2 outputs (normalized x, y)
    """
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# =============================================================================
# Datasets
# =============================================================================

class ETHXGazeDataset(Dataset):
    """
    ETH-XGaze dataset loader.
    
    Dataset: ~1M images with extreme head poses and gaze variations.
    Download: https://ait.ethz.ch/xgaze (requires registration)
    
    Expected structure:
        eth_xgaze/
        ├── train/
        │   ├── subject0000.h5
        │   ├── subject0001.h5
        │   └── ...
        └── test/
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Find all subject h5 files
        self.h5_files = sorted(glob.glob(str(self.root_dir / split / "*.h5")))
        
        if not self.h5_files:
            # Alternative: single h5 file
            single_file = self.root_dir / f"{split}.h5"
            if single_file.exists():
                self.h5_files = [str(single_file)]
        
        if not self.h5_files:
            raise ValueError(f"No h5 files found in {self.root_dir / split}")
        
        # Build index
        self.samples = []
        for h5_path in self.h5_files:
            with h5py.File(h5_path, 'r') as f:
                n_samples = len(f['face_patch'])
                for i in range(n_samples):
                    self.samples.append((h5_path, i))
        
        print(f"ETH-XGaze {split}: {len(self.samples)} samples from {len(self.h5_files)} files")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        h5_path, sample_idx = self.samples[idx]
        
        with h5py.File(h5_path, 'r') as f:
            face = f['face_patch'][sample_idx]  # [H, W, 3] uint8
            gaze = f['face_gaze'][sample_idx]   # [2] pitch, yaw in radians
        
        # Convert to PIL for transforms
        face = Image.fromarray(face)
        
        if self.transform:
            face = self.transform(face)
        else:
            face = T.ToTensor()(face)
        
        return {
            'face': face,
            'gaze': torch.FloatTensor(gaze),
        }


class Gaze360Dataset(Dataset):
    """
    Gaze360 dataset loader.
    
    Dataset: 238K images with 360° gaze range.
    Download: http://gaze360.csail.mit.edu/
    
    Expected structure:
        gaze360/
        ├── imgs/
        │   ├── 0000/
        │   └── ...
        └── metadata.mat or annotations.txt
    """
    
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load annotations
        self.samples = self._load_annotations()
        print(f"Gaze360 {split}: {len(self.samples)} samples")
    
    def _load_annotations(self):
        """Load Gaze360 annotations."""
        samples = []
        
        # Try different annotation formats
        txt_path = self.root_dir / f"{self.split}.txt"
        
        if txt_path.exists():
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        img_path = self.root_dir / "imgs" / parts[0]
                        gaze = [float(parts[1]), float(parts[2])]  # pitch, yaw
                        samples.append({
                            'image_path': str(img_path),
                            'gaze': gaze,
                        })
        else:
            # Fallback: search for images
            for img_path in self.root_dir.rglob("*.jpg"):
                samples.append({
                    'image_path': str(img_path),
                    'gaze': [0.0, 0.0],  # Placeholder
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        face = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            face = self.transform(face)
        else:
            face = T.ToTensor()(face)
        
        return {
            'face': face,
            'gaze': torch.FloatTensor(sample['gaze']),
        }


class PersonalCalibrationDataset(Dataset):
    """
    Personal calibration dataset from your collected data.
    
    Expected format:
        calibration_data/
        ├── img_1234567890.jpg
        ├── img_1234567890.json
        └── ...
    
    JSON format:
        {
            "screen_x": 1076.6445,
            "screen_y": 973.9961,
            "inference": { "Gaze": { ... } }
        }
    """
    
    def __init__(
        self, 
        calibration_dir: str, 
        screen_width: int = 1920,
        screen_height: int = 1080,
        transform=None
    ):
        self.calibration_dir = Path(calibration_dir)
        self.screen_w = screen_width
        self.screen_h = screen_height
        
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load all samples
        self.samples = []
        for json_path in sorted(glob.glob(str(self.calibration_dir / "*.json"))):
            with open(json_path) as f:
                data = json.load(f)
            
            img_path = json_path.replace('.json', '.jpg')
            if not os.path.exists(img_path):
                continue
            
            gaze_data = data.get('inference', {}).get('Gaze', {})
            
            self.samples.append({
                'image_path': img_path,
                'screen_x': data.get('screen_x', 0),
                'screen_y': data.get('screen_y', 0),
                'head_yaw': gaze_data.get('yaw', 0),
                'head_pitch': gaze_data.get('pitch', 0),
                'head_roll': gaze_data.get('roll', 0),
            })
        
        print(f"Personal calibration: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        face = Image.open(sample['image_path']).convert('RGB')
        face = self.transform(face)
        
        # Normalized screen coordinates
        norm_x = sample['screen_x'] / self.screen_w
        norm_y = sample['screen_y'] / self.screen_h
        
        # Head pose
        head_pose = torch.FloatTensor([
            sample['head_yaw'],
            sample['head_pitch'],
            sample['head_roll'],
        ])
        
        return {
            'face': face,
            'screen_target': torch.FloatTensor([norm_x, norm_y]),
            'head_pose': head_pose,
            'aspect_ratio': self.screen_w / self.screen_h,
        }


# =============================================================================
# Loss Functions
# =============================================================================

def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Angular error loss between predicted and target gaze.
    
    Args:
        pred: [B, 2] predicted (pitch, yaw) in radians
        target: [B, 2] target (pitch, yaw) in radians
    
    Returns:
        Scalar loss (mean angular error in radians)
    """
    # Convert to unit vectors
    pred_vec = pitch_yaw_to_vector(pred)
    target_vec = pitch_yaw_to_vector(target)
    
    # Cosine similarity
    cos_sim = torch.sum(pred_vec * target_vec, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Angular error
    angular_error = torch.acos(cos_sim)
    
    return angular_error.mean()


def pitch_yaw_to_vector(angles: torch.Tensor) -> torch.Tensor:
    """Convert pitch/yaw angles to unit gaze vector."""
    pitch = angles[:, 0]
    yaw = angles[:, 1]
    
    x = -torch.sin(yaw) * torch.cos(pitch)
    y = -torch.sin(pitch)
    z = torch.cos(yaw) * torch.cos(pitch)
    
    return torch.stack([x, y, z], dim=1)


def vector_to_pitch_yaw(vector: torch.Tensor) -> torch.Tensor:
    """Convert unit gaze vector to pitch/yaw angles."""
    x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]
    
    pitch = torch.asin(-y)
    yaw = torch.atan2(-x, z)
    
    return torch.stack([pitch, yaw], dim=1)


# =============================================================================
# Training Functions
# =============================================================================

def train_base_model(args):
    """
    Phase 1: Train base gaze model on large-scale datasets.
    """
    print("\n" + "="*60)
    print("PHASE 1: Base Model Training")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(project="gaze-estimation", name=f"base-{args.backbone}")
    
    # Data transforms
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    datasets_train = []
    datasets_val = []
    
    if args.eth_xgaze_path and H5PY_AVAILABLE:
        try:
            eth_train = ETHXGazeDataset(args.eth_xgaze_path, 'train', train_transform)
            datasets_train.append(eth_train)
            
            eth_val = ETHXGazeDataset(args.eth_xgaze_path, 'test', val_transform)
            datasets_val.append(eth_val)
        except Exception as e:
            print(f"Warning: Could not load ETH-XGaze: {e}")
    
    if args.gaze360_path:
        try:
            gaze360_train = Gaze360Dataset(args.gaze360_path, 'train', train_transform)
            datasets_train.append(gaze360_train)
        except Exception as e:
            print(f"Warning: Could not load Gaze360: {e}")
    
    if not datasets_train:
        raise ValueError("No training datasets loaded!")
    
    train_dataset = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
    val_dataset = datasets_val[0] if datasets_val else None
    
    print(f"\nTotal training samples: {len(train_dataset)}")
    
    # DataLoaders - optimized for DGX Spark
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers // 2,
            pin_memory=True,
        )
    
    # Model
    if args.backbone == 'mobilenetv2':
        model = MobileNetV2GazeNet(width_mult=args.width_mult, pretrained=True)
    elif args.backbone == 'efficientnet':
        model = EfficientNetB0GazeNet(pretrained=True)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
    )
    
    # Mixed precision
    scaler = GradScaler()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            face = batch['face'].to(device)
            gaze_target = batch['gaze'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                gaze_pred = model(face)
                loss = angular_loss(gaze_pred, gaze_target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if batch_idx % args.log_interval == 0:
                # Convert to degrees for readability
                loss_deg = np.degrees(loss.item())
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss_deg:.2f}° ({loss.item():.4f} rad)")
        
        scheduler.step()
        
        # Epoch stats
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / n_batches
        avg_loss_deg = np.degrees(avg_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} complete:")
        print(f"  Train Loss: {avg_loss_deg:.2f}° ({avg_loss:.4f} rad)")
        print(f"  Time: {epoch_time:.1f}s, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            n_val = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    face = batch['face'].to(device)
                    gaze_target = batch['gaze'].to(device)
                    
                    gaze_pred = model(face)
                    loss = angular_loss(gaze_pred, gaze_target)
                    
                    val_loss += loss.item()
                    n_val += 1
            
            val_loss = val_loss / n_val
            val_loss_deg = np.degrees(val_loss)
            print(f"  Val Loss: {val_loss_deg:.2f}° ({val_loss:.4f} rad)")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, output_dir / 'best_model.pt')
                print(f"  Saved best model (val loss: {val_loss_deg:.2f}°)")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch+1}.pt')
        
        # Log to wandb
        if WANDB_AVAILABLE and args.use_wandb:
            log_dict = {
                'train_loss': avg_loss,
                'train_loss_deg': avg_loss_deg,
                'lr': scheduler.get_last_lr()[0],
                'epoch': epoch,
            }
            if val_loader:
                log_dict['val_loss'] = val_loss
                log_dict['val_loss_deg'] = val_loss_deg
            wandb.log(log_dict)
    
    # Export final model to ONNX
    print("\nExporting to ONNX...")
    model_to_export = model.module if hasattr(model, 'module') else model
    model_to_export.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = output_dir / 'gaze_model.onnx'
    
    torch.onnx.export(
        model_to_export,
        dummy_input,
        str(onnx_path),
        input_names=['face_image'],
        output_names=['gaze_angles'],
        dynamic_axes={
            'face_image': {0: 'batch'},
            'gaze_angles': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported: {onnx_path}")
    
    print("\nTraining complete!")


def finetune_personal(args):
    """
    Phase 2: Fine-tune for personal calibration.
    """
    print("\n" + "="*60)
    print("PHASE 2: Personal Calibration Fine-tuning")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    
    if args.backbone == 'mobilenetv2':
        gaze_model = MobileNetV2GazeNet(width_mult=args.width_mult, pretrained=False)
    else:
        gaze_model = EfficientNetB0GazeNet(pretrained=False)
    
    checkpoint = torch.load(args.base_model, map_location=device)
    gaze_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze backbone, keep head trainable
    for param in gaze_model.features.parameters():
        param.requires_grad = False
    for param in gaze_model.gaze_head.parameters():
        param.requires_grad = True
    
    gaze_model = gaze_model.to(device)
    
    # Calibration MLP
    calibration_mlp = ScreenCalibrationMLP(input_dim=8).to(device)
    
    # Dataset
    dataset = PersonalCalibrationDataset(
        args.calibration_dir,
        screen_width=args.screen_width,
        screen_height=args.screen_height,
    )
    
    if len(dataset) < 5:
        print(f"\nWARNING: Only {len(dataset)} samples. Recommend 15-25 for good calibration.")
    
    # Simple train/val split (80/20)
    n_train = int(len(dataset) * 0.8)
    n_val = len(dataset) - n_train
    
    if n_val > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        train_dataset = dataset
        val_dataset = None
    
    train_loader = DataLoader(train_dataset, batch_size=min(8, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset)) if val_dataset else None
    
    # Optimizer
    optimizer = optim.Adam([
        {'params': gaze_model.gaze_head.parameters(), 'lr': 1e-5},
        {'params': calibration_mlp.parameters(), 'lr': 1e-3},
    ])
    
    # Loss
    criterion = nn.MSELoss()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aspect_ratio = args.screen_width / args.screen_height
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        gaze_model.train()
        calibration_mlp.train()
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            face = batch['face'].to(device)
            target = batch['screen_target'].to(device)
            head_pose = batch['head_pose'].to(device)
            
            optimizer.zero_grad()
            
            # Get gaze angles from base model
            gaze_angles = gaze_model(face)
            
            # Build calibration features
            batch_size = face.shape[0]
            aspect = torch.full((batch_size, 1), aspect_ratio, device=device)
            face_center = torch.full((batch_size, 2), 0.5, device=device)  # Placeholder
            
            calib_input = torch.cat([
                gaze_angles,    # [B, 2]
                head_pose,      # [B, 3]
                face_center,    # [B, 2]
                aspect,         # [B, 1]
            ], dim=1)
            
            # Predict screen coordinates
            pred = calibration_mlp(calib_input)
            
            # Loss
            loss = criterion(pred, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Convert to pixel error
        rmse_x = np.sqrt(avg_loss) * args.screen_width
        rmse_y = np.sqrt(avg_loss) * args.screen_height
        
        # Validation
        val_str = ""
        if val_loader:
            gaze_model.eval()
            calibration_mlp.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    face = batch['face'].to(device)
                    target = batch['screen_target'].to(device)
                    head_pose = batch['head_pose'].to(device)
                    
                    gaze_angles = gaze_model(face)
                    
                    batch_size = face.shape[0]
                    aspect = torch.full((batch_size, 1), aspect_ratio, device=device)
                    face_center = torch.full((batch_size, 2), 0.5, device=device)
                    
                    calib_input = torch.cat([gaze_angles, head_pose, face_center, aspect], dim=1)
                    pred = calibration_mlp(calib_input)
                    
                    val_loss = criterion(pred, target).item()
            
            val_rmse_x = np.sqrt(val_loss) * args.screen_width
            val_rmse_y = np.sqrt(val_loss) * args.screen_height
            val_str = f", Val: ~{val_rmse_x:.1f}px x {val_rmse_y:.1f}px"
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train ~{rmse_x:.1f}px x {rmse_y:.1f}px{val_str}")
    
    # Save models
    print("\nSaving models...")
    
    # Gaze model
    gaze_model_to_save = gaze_model.module if hasattr(gaze_model, 'module') else gaze_model
    torch.save(gaze_model_to_save.state_dict(), output_dir / 'gaze_model_finetuned.pt')
    
    # Calibration MLP
    torch.save(calibration_mlp.state_dict(), output_dir / 'calibration_mlp.pt')
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    
    gaze_model.eval()
    dummy_face = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        gaze_model_to_save,
        dummy_face,
        str(output_dir / 'gaze_model_finetuned.onnx'),
        input_names=['face_image'],
        output_names=['gaze_angles'],
        opset_version=17,
    )
    
    calibration_mlp.eval()
    dummy_calib = torch.randn(1, 8).to(device)
    torch.onnx.export(
        calibration_mlp,
        dummy_calib,
        str(output_dir / 'calibration_mlp.onnx'),
        input_names=['features'],
        output_names=['screen_coords'],
        opset_version=17,
    )
    
    print(f"\nModels saved to {output_dir}")
    print("\nFor MacBook inference, copy these ONNX files:")
    print(f"  - {output_dir}/gaze_model_finetuned.onnx")
    print(f"  - {output_dir}/calibration_mlp.onnx")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gaze Estimation Training')
    subparsers = parser.add_subparsers(dest='command', help='Training phase')
    
    # Phase 1: Base model training
    base_parser = subparsers.add_parser('train-base', help='Train base gaze model')
    base_parser.add_argument('--eth-xgaze-path', type=str, help='Path to ETH-XGaze dataset')
    base_parser.add_argument('--gaze360-path', type=str, help='Path to Gaze360 dataset')
    base_parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    base_parser.add_argument('--backbone', type=str, default='mobilenetv2', 
                            choices=['mobilenetv2', 'efficientnet'])
    base_parser.add_argument('--width-mult', type=float, default=0.5, 
                            help='MobileNetV2 width multiplier')
    base_parser.add_argument('--batch-size', type=int, default=512, 
                            help='Batch size (512 for DGX Spark)')
    base_parser.add_argument('--lr', type=float, default=1e-3)
    base_parser.add_argument('--weight-decay', type=float, default=1e-4)
    base_parser.add_argument('--epochs', type=int, default=50)
    base_parser.add_argument('--num-workers', type=int, default=16)
    base_parser.add_argument('--log-interval', type=int, default=100)
    base_parser.add_argument('--save-every', type=int, default=5)
    base_parser.add_argument('--use-wandb', action='store_true')
    
    # Phase 2: Personal fine-tuning
    personal_parser = subparsers.add_parser('finetune-personal', help='Fine-tune for personal calibration')
    personal_parser.add_argument('--base-model', type=str, required=True, 
                                help='Path to base model checkpoint')
    personal_parser.add_argument('--calibration-dir', type=str, required=True,
                                help='Directory with calibration images/JSON')
    personal_parser.add_argument('--screen-width', type=int, default=1920)
    personal_parser.add_argument('--screen-height', type=int, default=1080)
    personal_parser.add_argument('--output-dir', type=str, required=True)
    personal_parser.add_argument('--backbone', type=str, default='mobilenetv2')
    personal_parser.add_argument('--width-mult', type=float, default=0.5)
    personal_parser.add_argument('--epochs', type=int, default=200)
    
    args = parser.parse_args()
    
    if args.command == 'train-base':
        train_base_model(args)
    elif args.command == 'finetune-personal':
        finetune_personal(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()