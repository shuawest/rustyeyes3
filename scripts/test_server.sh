#!/bin/bash
# Quick test script for Moondream server
# Creates a test image and sends it to the server

echo "Creating test image..."
# Use Python to create a simple test image
./venv/bin/python3 -c "
from PIL import Image, ImageDraw
import sys

# Create a simple face-like image
img = Image.new('RGB', (640, 480), color='white')
draw = ImageDraw.Draw(img)

# Draw a simple face
draw.ellipse([220, 180, 420, 340], fill='beige', outline='black')  # Face
draw.ellipse([270, 220, 310, 260], fill='white', outline='black')  # Left eye
draw.ellipse([330, 220, 370, 260], fill='white', outline='black')  # Right eye
draw.ellipse([280, 230, 300, 250], fill='black')  # Left pupil
draw.ellipse([340, 230, 360, 250], fill='black')  # Right pupil
draw.arc([280, 280, 360, 320], 0, 180, fill='black', width=3)  # Smile

img.save('/tmp/test_face.jpg')
print('Test image created at /tmp/test_face.jpg')
"

echo ""
echo "Testing Moondream server..."
echo '{"image_path": "/tmp/test_face.jpg", "timestamp": 0}' | ./venv/bin/python3 scripts/moondream_server.py

echo ""
echo "Test complete! If you saw a JSON response above, the server is working."
