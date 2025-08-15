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

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy conversion script and make it executable
COPY convert_to_tensorRT.sh .
RUN chmod +x convert_to_tensorRT.sh

# Default command
CMD ["bash"] 