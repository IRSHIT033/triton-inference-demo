#!/bin/bash

# First, install required Python packages in the container
docker run --gpus=all --rm -it \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/models:/models \
  -v $(pwd)/requirements.txt:/workspace/requirements.txt \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  bash -c "pip install Pillow && tritonserver \
    --model-repository=/models \
    --backend-config=tensorrt,coalesce-request-input=true"
