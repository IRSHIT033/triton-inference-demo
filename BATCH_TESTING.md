# Batch Image Testing for DinoV2 Triton Server

## Overview

Your Triton setup **fully supports batch image testing** with up to **32 images per batch**. This can significantly improve throughput compared to processing images individually.

## Quick Start

### 1. Basic Batch Testing

```bash
# Process images from a directory
python batch_client.py --images /path/to/your/images --batch-size 8

# Create sample images and test
python batch_client.py --create-samples 10 --batch-size 4

# Use maximum batch size for best performance
python batch_client.py --create-samples 32 --batch-size 32
```

### 2. Performance Comparison

```bash
# Run benchmark to compare single vs batch processing
python benchmark_batch.py
```

## Batch Processing Benefits

### Performance Improvements

- **2-5x faster** throughput compared to single image processing
- **Better GPU utilization** through batching
- **Reduced network overhead** per image
- **Optimized memory usage** with dynamic batching

### Your Current Configuration

- **Max batch size**: 32 images
- **Dynamic batching**: Enabled with preferred sizes [4, 8, 16]
- **Queue delay**: 2ms max for batching optimization
- **GPU acceleration**: TensorRT with FP16 precision

## Usage Examples

### Process Directory of Images

```bash
python batch_client.py --images ./my_photos --batch-size 16
```

### Create Test Images

```bash
python batch_client.py --create-samples 20 --batch-size 8
```

### Custom Model or Server

```bash
python batch_client.py --model dinov2_ensemble --images ./data --batch-size 12
```

## Expected Performance

Based on your TensorRT configuration with dynamic batching:

| Batch Size | Expected Throughput | Use Case             |
| ---------- | ------------------- | -------------------- |
| 1          | ~2-5 img/sec        | Single image testing |
| 4          | ~8-15 img/sec       | Small batches        |
| 8          | ~15-25 img/sec      | Balanced performance |
| 16         | ~25-35 img/sec      | High throughput      |
| 32         | ~30-40 img/sec      | Maximum throughput   |

_Actual performance depends on your GPU, image sizes, and server load._

## Features

### Batch Client (`batch_client.py`)

- ✅ Process multiple images simultaneously
- ✅ Auto-discover images in directories
- ✅ Support for JPG, PNG, BMP, TIFF formats
- ✅ Configurable batch sizes (1-32)
- ✅ Detailed timing and throughput metrics
- ✅ Error handling and recovery
- ✅ Progress tracking

### Benchmark Tool (`benchmark_batch.py`)

- ✅ Compare single vs batch performance
- ✅ Test multiple batch sizes automatically
- ✅ Performance recommendations
- ✅ Detailed timing analysis

## Troubleshooting

### Common Issues

1. **Batch size too large**

   ```
   Error: Resource exhausted
   Solution: Reduce batch size (try 16 or 8)
   ```

2. **Server not ready**

   ```bash
   # Start Triton server
   docker-compose up triton
   ```

3. **Out of memory**
   ```
   Solution: Reduce batch size or use smaller images
   ```

### Performance Tips

1. **Optimal batch size**: Usually 8-16 for best balance
2. **Image preprocessing**: Already optimized in your setup
3. **GPU memory**: Monitor with `nvidia-smi`
4. **Network latency**: Batch processing reduces per-image overhead

## Integration Examples

### Python Script Integration

```python
from batch_client import process_images_in_batches

# Process your images
image_paths = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
features = process_images_in_batches(image_paths, batch_size=8)
print(f"Extracted features shape: {features.shape}")
```

### REST API Alternative

```bash
# Your setup also supports REST API for batching
curl -X POST http://localhost:8000/v2/models/dinov2_ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "IMAGE", "shape": [2, 1], "datatype": "BYTES", "data": [...]}]}'
```

## Next Steps

1. **Test with your images**: `python batch_client.py --images /your/image/directory`
2. **Find optimal batch size**: `python benchmark_batch.py`
3. **Integrate into your workflow**: Use the batch processing functions in your applications
4. **Monitor performance**: Use the timing metrics to optimize your setup

Your Triton server is already configured for excellent batch performance! 🚀
