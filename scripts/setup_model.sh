#!/bin/bash
# Setup script for DINO V2 model conversion and deployment

set -e

echo "🚀 Setting up DINO V2 model for Triton inference..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p model_repository/dinov2_vit/1
mkdir -p /tmp/trt_cache
mkdir -p data/sample_images

# Create and activate virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies in virtual environment
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Convert model to TensorRT
echo "🔄 Converting DINO V2 model to TensorRT..."
./venv/bin/python scripts/convert_to_tensorrt.py \
    --model-name facebook/dinov2-base \
    --output-dir ./model_repository/dinov2_vit/1 \
    --max-batch-size 8 \
    --precision fp16

# Validate model repository structure
echo "✅ Validating model repository structure..."
if [ -f "model_repository/dinov2_vit/config.pbtxt" ] && [ -f "model_repository/dinov2_vit/1/model.plan" ]; then
    echo "✓ Model repository structure is valid"
else
    echo "❌ Model repository structure is invalid"
    exit 1
fi

# Download sample images for testing
echo "🖼️ Downloading sample images..."
./venv/bin/python -c "
import urllib.request
import os

# Sample images URLs
urls = [
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png'
]

os.makedirs('data/sample_images', exist_ok=True)

for i, url in enumerate(urls):
    try:
        urllib.request.urlretrieve(url, f'data/sample_images/sample_{i+1}.png')
        print(f'Downloaded sample_{i+1}.png')
    except Exception as e:
        print(f'Failed to download {url}: {e}')
"

echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Start Triton server: docker-compose up triton-server"
echo "3. Test inference: ./venv/bin/python client/triton_client.py --image data/sample_images/sample_1.png"
echo ""
echo "Model repository structure:"
echo "model_repository/"
echo "└── dinov2_vit/"
echo "    ├── config.pbtxt"
echo "    └── 1/"
echo "        └── model.plan" 