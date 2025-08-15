# DINOv2 Triton Inference Server

This repository contains a complete setup for running DINOv2 models with NVIDIA Triton Inference Server using Docker Compose.

## Architecture

The system consists of three models in an ensemble:

1. **dinov2_preprocess**: Python-based preprocessing (PIL/Pillow image operations)
2. **dinov2_trt**: TensorRT optimized DINOv2 model
3. **dinov2_postprocess**: Python-based postprocessing (L2 normalization)

## Quick Start

### 1. Build the Models

First, build the ONNX and TensorRT models:

```bash
# Build and run the model builder
docker compose --profile build up model-builder

# Or run manually:
docker compose build model-builder
docker compose run --rm model-builder
```

### 2. Start Triton Server

Once models are built, start the Triton server:

```bash
docker compose up triton-server
```

The server will be available at:

- HTTP: http://localhost:8000
- GRPC: localhost:8001
- Metrics: http://localhost:8002

### 3. Test the Server

Check server status:

```bash
curl http://localhost:8000/v2/health/ready
```

List available models:

```bash
curl http://localhost:8000/v2/models
```

## Manual Setup (Alternative)

If you prefer to run without Docker Compose:

### 1. Build Models

```bash
# Export ONNX model
python export_model_onnx.py

# Convert to TensorRT
chmod +x convert_to_tensorrt.sh
./convert_to_tensorrt.sh
```

### 2. Build and Run Container

```bash
# Build the image
docker build -t dinov2-triton .

# Run the container
docker run --gpus=all --rm -it \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  dinov2-triton
```

## Usage

### API Endpoint

Send POST requests to `http://localhost:8000/v2/models/dinov2_ensemble/infer`

### Example Request

```python
import requests
import base64
import json

# Read image file
with open("your_image.jpg", "rb") as f:
    image_bytes = f.read()

# Prepare request
data = {
    "inputs": [
        {
            "name": "IMAGE_BYTES",
            "shape": [1, 1],
            "datatype": "BYTES",
            "data": [base64.b64encode(image_bytes).decode()]
        }
    ]
}

# Send request
response = requests.post(
    "http://localhost:8000/v2/models/dinov2_ensemble/infer",
    json=data
)

# Get embeddings
result = response.json()
embeddings = result["outputs"][0]["data"]  # 768-dimensional vector
```

## Troubleshooting

### Common Issues

1. **PIL Module Error**: Ensure the Dockerfile installs Pillow correctly
2. **TensorRT Model Missing**: Run the model builder first
3. **GPU Not Available**: Ensure NVIDIA Docker runtime is installed

### Logs

Check container logs:

```bash
docker-compose logs triton-server
```

### Health Check

The service includes health checks. Check status:

```bash
docker-compose ps
```

## Configuration

### Model Configuration

- **Batch Size**: Max 32 (configurable in config.pbtxt files)
- **Image Size**: 224x224 (DINOv2 standard)
- **Precision**: FP16 for TensorRT model
- **Output**: 768-dimensional embeddings (DINOv2-base)

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control GPU visibility
- Custom Triton arguments can be added to docker-compose.yml

## Files Structure

```
triton/
├── Dockerfile              # Main Triton server image
├── Dockerfile.builder      # Model building image
├── docker-compose.yml      # Orchestration
├── export_model_onnx.py    # ONNX export script
├── convert_to_tensorrt.sh  # TensorRT conversion
├── models/                 # Model repository
│   ├── dinov2_ensemble/
│   ├── dinov2_preprocess/
│   ├── dinov2_postprocess/
│   └── dinov2_trt/
└── requirements.txt        # Python dependencies
```
