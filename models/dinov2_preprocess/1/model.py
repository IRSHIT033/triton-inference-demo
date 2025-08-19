import io
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

IM_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            # Receive bytes tensor shaped [batch, 1]
            img_bytes = pb_utils.get_input_tensor_by_name(req, "IMAGE").as_numpy()
            bs = img_bytes.shape[0]
            out = np.empty((bs, 3, IM_SIZE, IM_SIZE), dtype=np.float16)  # FP16

            for i in range(bs):
                b = img_bytes[i, 0].tobytes()
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img = img.resize((IM_SIZE, IM_SIZE), Image.BILINEAR)
                x = np.asarray(img, dtype=np.float32) / 255.0  # HWC float32
                x = (x - MEAN) / STD
                x = np.transpose(x, (2, 0, 1))  # to CHW
                out[i] = x.astype(np.float16)

            # Ensure the output tensor is explicitly FP16
            # Double-check the dtype and make sure it's contiguous
            out_fp16 = np.ascontiguousarray(out, dtype=np.float16)
            print(f"DEBUG: Output tensor dtype: {out_fp16.dtype}, shape: {out_fp16.shape}")
            out_tensor = pb_utils.Tensor("input", out_fp16)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
