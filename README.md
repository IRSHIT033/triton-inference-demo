# DINO V2 ViT Triton Inference Pipeline

A high-performance inference pipeline for DINO V2 Vision Transformer models using NVIDIA Triton Inference Server with TensorRT backend optimization.

## Features

- 🚀 **High Performance**: TensorRT optimization for fast inference
- 🔄 **Batch Processing**: Efficient batch inference with dynamic batching
- 🐳 **Containerized**: Docker-based deployment for easy scaling
- 🎯 **Production Ready**: Triton server with health checks and monitoring
- 📊 **Feature Extraction**: Extract rich visual features from images
- 🔍 **Similarity Computation**: Built-in cosine similarity calculations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│  Triton Server   │───▶│  TensorRT       │
│                 │    │                  │    │  Engine         │
│ - Image Input   │    │ - Load Balancing │    │                 │
│ - Preprocessing │    │ - Batching       │    │ - DINO V2 ViT   │
│ - Post-process  │    │ - Health Checks  │    │ - FP16 Precision│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Runtime
- Python 3.8+

### 1. Setup and Model Conversion

```bash
# Clone the repository
git clone <your-repo-url>
cd triton-dinov2

# Run setup script
chmod +x scripts/setup_model.sh
./scripts/setup_model.sh
```

This script will:

- Install dependencies
- Download and convert DINO V2 model to TensorRT
- Create model repository structure
- Download sample images for testing

### 2. Start Triton Server

```bash
# Using Docker Compose (recommended)
docker-compose up triton-server

# Or using Docker directly
docker build -f docker/Dockerfile.triton -t triton-dinov2 .
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  triton-dinov2
```

### 3. Run Inference

```bash
# Single image inference
python client/triton_client.py --image data/sample_images/sample_1.png

# Compare two images
python client/triton_client.py \
  --image data/sample_images/sample_1.png \
  --compare-image data/sample_images/sample_2.png

# Batch inference
python client/triton_client.py \
  --batch-images data/sample_images/*.png \
  --output features.npy

# Batch inference example
python examples/batch_inference.py --image-dir data/sample_images/
```

## Model Configuration

The DINO V2 model is configured with the following specifications:

- **Model**: facebook/dinov2-base (768-dimensional features)
- **Input**: RGB images, 224x224 pixels
- **Batch Size**: Up to 8 images
- **Precision**: FP16 for optimal performance
- **Backend**: TensorRT for GPU acceleration

### Configuration File

```protobuf
name: "dinov2_vit"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "features"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
```

## API Usage

### Python Client

```python
from client.triton_client import DinoV2TritonClient

# Initialize client
client = DinoV2TritonClient(triton_url="localhost:8000")

# Single image inference
features = client.infer_single("path/to/image.jpg")
print(f"Feature shape: {features.shape}")  # (768,)

# Batch inference
batch_features = client.infer(preprocessed_batch)
print(f"Batch features shape: {batch_features.shape}")  # (batch_size, 768)

# Compute similarity
similarity = client.compute_similarity(features1, features2)
print(f"Cosine similarity: {similarity:.4f}")
```

### HTTP API

```bash
# Health check
curl http://localhost:8000/v2/health/ready

# Model metadata
curl http://localhost:8000/v2/models/dinov2_vit

# Inference (using tritonclient or custom HTTP client)
```

### gRPC API

```python
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient(url="localhost:8001")
# ... inference code
```

## Performance Optimization

### TensorRT Optimizations

- **FP16 Precision**: Reduces memory usage and increases throughput
- **Dynamic Batching**: Automatically batches requests for efficiency
- **Engine Caching**: Reuses compiled engines across restarts
- **Workspace Optimization**: 1GB workspace for optimal performance

### Scaling Options

1. **Horizontal Scaling**: Deploy multiple Triton instances
2. **Model Instances**: Configure multiple model instances per GPU
3. **Multi-GPU**: Distribute models across multiple GPUs

```protobuf
instance_group [
  {
    count: 2  # Multiple instances
    kind: KIND_GPU
    gpus: [ 0, 1 ]  # Multiple GPUs
  }
]
```

## Monitoring and Metrics

Triton provides comprehensive metrics on port 8002:

```bash
# Prometheus metrics
curl http://localhost:8002/metrics

# Server statistics
curl http://localhost:8000/v2/models/dinov2_vit/stats
```

Key metrics to monitor:

- Inference latency
- Throughput (requests/second)
- Queue time
- GPU utilization
- Memory usage

## Directory Structure

```
triton-dinov2/
├── model_repository/           # Triton model repository
│   └── dinov2_vit/
│       ├── config.pbtxt       # Model configuration
│       └── 1/
│           └── model.plan     # TensorRT engine
├── client/
│   └── triton_client.py       # Python client
├── scripts/
│   ├── convert_to_tensorrt.py # Model conversion
│   └── setup_model.sh         # Setup script
├── docker/
│   ├── Dockerfile.triton      # Server container
│   └── Dockerfile.client      # Client container
├── examples/
│   └── batch_inference.py     # Batch processing example
├── docker-compose.yml         # Multi-container setup
└── requirements.txt           # Python dependencies
```

## Advanced Usage

### Custom Model Variants

To use different DINO V2 variants:

```bash
# DINO V2 Small (384-dim features)
python scripts/convert_to_tensorrt.py --model-name facebook/dinov2-small

# DINO V2 Large (1024-dim features)
python scripts/convert_to_tensorrt.py --model-name facebook/dinov2-large

# DINO V2 Giant (1536-dim features)
python scripts/convert_to_tensorrt.py --model-name facebook/dinov2-giant
```

### Integration with Vector Databases

```python
# Example: Store features in a vector database
import faiss

# Extract features
features = client.infer_single("image.jpg")

# Add to FAISS index
index = faiss.IndexFlatIP(768)  # Inner product for cosine similarity
index.add(features.reshape(1, -1))

# Search similar images
similarities, indices = index.search(query_features.reshape(1, -1), k=10)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size in config.pbtxt
   - Use INT8 precision instead of FP16

2. **Model Loading Fails**

   - Check TensorRT version compatibility
   - Verify CUDA driver version
   - Check model file permissions

3. **Slow Inference**
   - Enable engine caching
   - Optimize batch size
   - Check GPU utilization

### Debug Mode

```bash
# Enable verbose logging
docker run --gpus all -p 8000:8000 \
  -e TRITON_LOG_VERBOSE=2 \
  triton-dinov2
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for the DINO V2 model
- NVIDIA for Triton Inference Server and TensorRT
- Hugging Face for model hosting and transformers library
