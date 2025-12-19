import os
import json
import cv2
import glob
import numpy as np
import pandas as pd
import mediapipe as mp
import shutil
import random

# --- CONFIG ---
INPUT_DIR = "calibration_data"
OUTPUT_DIR = "dataset_clean"
IMG_SIZE = 448
DEG_PER_PIXEL = 0.0307
SCREEN_CENTER_X = 864
SCREEN_CENTER_Y = 558

# Augmentation Settings
AUGMENT_FACTOR = 50  # Generate 50 versions of each image

def setup_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

def get_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )


def compute_crop_rect(landmarks, img_w, img_h):
    # Convert normalized landmarks to pixels
    px_points = []
    for lm in landmarks.landmark:
        px_points.append((lm.x * img_w, lm.y * img_h))
    
    # Bounding Box
    xs = [p[0] for p in px_points]
    ys = [p[1] for p in px_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    w = max_x - min_x
    h = max_y - min_y
    cx = min_x + w / 2.0
    cy = min_y + h / 2.0
    
    # 1.5x Expansion (Matching Rust L2CS Logic)
    size = max(w, h) * 1.5
    
    # Box
    x = cx - size / 2.0
    y = cy - size / 2.0
    
    return int(x), int(y), int(size)

def augment_image(img):
    """Apply random augmentations: brightness, noise, blur"""
    rows, cols, ch = img.shape
    
    # 1. Random Brightness/Contrast
    alpha = 1.0 + random.uniform(-0.3, 0.3) # Contrast
    beta = random.uniform(-30, 30)          # Brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # 2. Gaussian Blur (Simulate motion)
    if random.random() < 0.3:
        k = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (k, k), 0)
        
    # 3. Noise
    if random.random() < 0.3:
        gauss = np.random.normal(0, 15, (rows, cols, ch))
        img = img + gauss
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    # 4. Small Shift (Jitter crop slightly)
    # We do this by shift-warping instead of re-cropping for simplicity
    if random.random() < 0.5:
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (cols, rows))
        
    return img

def main():
    setup_dirs()
    print(f"Scanning {INPUT_DIR}...")
    
    json_files = glob.glob(os.path.join(INPUT_DIR, "img_*.json"))
    print(f"Found {len(json_files)} samples.")
    
    data_records = []
    
    with get_face_mesh() as face_mesh:
        for json_path in json_files:
            # 1. Load Data
            base_name = os.path.basename(json_path).replace(".json", "")
            img_path = os.path.join(INPUT_DIR, base_name + ".jpg")
            
            if not os.path.exists(img_path):
                print(f"Missing Image: {img_path}")
                continue
                
            with open(json_path, 'r') as f:
                meta = json.load(f)
                
            target_x = meta.get("target_x", 0)
            target_y = meta.get("target_y", 0)
            
            # 2. Process Image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load: {img_path}")
                continue
                
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Detect & Crop
            results = face_mesh.process(rgb_img)
            if not results.multi_face_landmarks:
                print(f"No face detected: {base_name}")
                continue
                
            # Use first face
            cx, cy, size = compute_crop_rect(results.multi_face_landmarks[0], w, h)
            
            # Safe Crop logic
            # Handle out of bounds by padding? Or just clipping. Use cv2 handling?
            # Creating a large canvas to handle out-of-bounds crops cleanly
            canvas_size = max(w, h) * 3
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            pad_x = (canvas_size - w) // 2
            pad_y = (canvas_size - h) // 2
            canvas[pad_y:pad_y+h, pad_x:pad_x+w] = img
            
            # Adjust crop coords to canvas
            cx += pad_x
            cy += pad_y
            
            # Crop
            x1 = cx
            y1 = cy
            x2 = cx + size
            y2 = cy + size
            
            # Ensure valid
            if size < 10: 
                print(f"Crop too small: {base_name}")
                continue
                
            crop = canvas[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"Empty crop: {base_name}")
                continue
                
            # Resize
            final_img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            
            # 4. Compute Labels (Angles)
            # Head Yaw (Left is +) corresponds to Target X vs Center
            # Screen Right (High X) -> Head looks Right (Negative Yaw)
            # Wait, our runtime uses: Right Head = Positive Yaw (Screen Right). 
            # Step 2468: "Positive Yaw = Screen Right".
            # So: yaw = (target_x - center) * deg_per_px.
            
            yaw = (target_x - SCREEN_CENTER_X) * DEG_PER_PIXEL
            
            # Pitch: Look Up (Negative Pitch). Screen Top (Low Y).
            # So Low Y -> Neg Pitch.
            # pitch = (target_y - SCREEN_CENTER_Y) * DEG_PER_PIXEL
            # If Y=0 (Top): (0 - 558) = -558. times 0.03 = -16. Good.
            # If Y=1000 (Bot): (1000 - 558) = +442. times 0.03 = +13. Good.
            pitch = (target_y - SCREEN_CENTER_Y) * DEG_PER_PIXEL
            
            # 5. Generate Augmented Versions
            for i in range(AUGMENT_FACTOR):
                if i == 0:
                    aug_img = final_img # Originals
                    suffix = "orig"
                else:
                    aug_img = augment_image(final_img.copy())
                    suffix = f"aug_{i}"
                    
                filename = f"{base_name}_{suffix}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(save_path, aug_img)
                
                data_records.append({
                    "filename": filename,
                    "yaw": yaw,
                    "pitch": pitch,
                    "orig_x": target_x,
                    "orig_y": target_y
                })
                
    # 6. Save Labels
    df = pd.DataFrame(data_records)
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"Created {len(df)} samples in {OUTPUT_DIR}/labels.csv")

if __name__ == "__main__":
    main()
