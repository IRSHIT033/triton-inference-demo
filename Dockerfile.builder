FROM nvcr.io/nvidia/tensorrt:24.08-py3

# Install Python dependencies for model export
RUN pip install --no-cache-dir \
    torch \
    transformers \
    onnx \
    onnxruntime \
    safetensors \
    huggingface-hub \
    onnxscript \
    torchvision

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"] 