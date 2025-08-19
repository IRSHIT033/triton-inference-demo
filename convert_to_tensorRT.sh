#!/bin/bash

echo "Converting ONNX model to TensorRT..."

# Check if ONNX model exists
if [ ! -f "dinov2_base_cls.onnx" ]; then
    echo "Error: dinov2_base_cls.onnx not found. Please run export_model_onnx.py first."
    exit 1
fi

# Create the TensorRT model directory
mkdir -p models/dinov2/1

# Convert ONNX to TensorRT
trtexec \
  --onnx=dinov2_base_cls.onnx \
  --saveEngine=models/dinov2/1/model.plan \
  --fp16 \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:8x3x224x224 \
  --maxShapes=pixel_values:32x3x224x224 \
  --shapes=pixel_values:8x3x224x224 \
  --profilingVerbosity=detailed \
  --timingCacheFile=dinov2.timing

# Check if conversion was successful
if [ -f "models/dinov2/1/model.plan" ]; then
    echo "TensorRT model created successfully at models/dinov2/1/model.plan"
    ls -la models/dinov2/1/
else
    echo "Error: TensorRT conversion failed"
    exit 1
fi
