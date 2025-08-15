FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install Python dependencies required by the models
RUN pip install --no-cache-dir \
    Pillow \
    numpy \
    torch \
    transformers \
    onnx \
    onnxruntime \
    safetensors \
    huggingface-hub \
    onnxscript \
    torchvision \
    tritonclient[http]

# Set working directory
WORKDIR /workspace

# Copy requirements for reference
COPY requirements.txt .

# Copy model repository
COPY models/ /models/

# Copy setup scripts
COPY export_model_onnx.py .
COPY setup_models.sh .

# Make scripts executable
RUN chmod +x setup_models.sh

# Expose Triton server ports
EXPOSE 8000 8001 8002

# Default command to run Triton server
CMD ["tritonserver", "--model-repository=/models", "--backend-config=tensorrt,coalesce-request-input=true"] 