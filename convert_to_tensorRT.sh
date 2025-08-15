# From the folder that contains dinov2_base_cls.onnx
docker run --gpus=all --rm -it -v $PWD:/workspace nvcr.io/nvidia/tensorrt:24.08-py3 bash
# inside the container:
cd /workspace
trtexec \
  --onnx=dinov2_base_cls.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --profilingVerbosity=detailed \
  --timingCacheFile=dinov2.timing


trtexec \
  --onnx=dinov2_base_cls.onnx \
  --saveEngine=model.plan \
  --fp16 \
  --minShapes=pixel_values:1x3x224x224 \
  --optShapes=pixel_values:8x3x224x224 \
  --maxShapes=pixel_values:32x3x224x224 \
  --shapes=pixel_values:8x3x224x224 \
  --profilingVerbosity=detailed \
  --timingCacheFile=dinov2.timing
