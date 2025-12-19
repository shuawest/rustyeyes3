import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- CONFIG ---
IMG_SIZE = 448
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4

class GazeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            # Handle corrupt images gracefully
            return self.__getitem__((idx + 1) % len(self))
            
        # Labels: Yaw, Pitch
        yaw = self.data_frame.iloc[idx, 1]
        pitch = self.data_frame.iloc[idx, 2]
        labels = torch.tensor([yaw, pitch], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels

class L2CS_Regression(nn.Module):
    def __init__(self, pretrained=True):
        super(L2CS_Regression, self).__init__()
        # Backbone: ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace FC layer (originally 2048 -> 1000)
        # We want 2048 -> 2 (Yaw, Pitch)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)
        
    def forward(self, x):
        return self.backbone(x)

def main():
    parser = argparse.ArgumentParser(description="Train Custom Gaze Model")
    parser.add_argument("--data_dir", type=str, default="dataset_clean", help="Path to dataset folder")
    parser.add_argument("--output", type=str, default="custom_gaze.onnx", help="Output ONNX filename")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights (optional)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (Normalization matches L2CS standard)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset
    csv_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    dataset = GazeDataset(csv_path, args.data_dir, transform=transform)
    
    # Simple Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = L2CS_Regression(pretrained=True)
    if args.weights:
        print(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights))
    model.to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f}")
        
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Saved Best Model")

    print("Training Complete. Exporting ONNX...")
    
    # Export ONNX
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    model.to("cpu") # Export on CPU usually safer
    
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    
    # Dynamic axes for batch size support
    torch.onnx.export(
        model, 
        dummy_input, 
        args.output,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"Exported to {args.output}")

if __name__ == "__main__":
    main()
