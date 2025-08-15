import io
import numpy as np
from PIL import Image
import triton_python_backend_utils as pb_utils

# ImageNet mean/std used by DINOv2 preprocessor
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def resize_short_side(img, short=256):
    w, h = img.size
    if w <= h:
        new_w = short
        new_h = int(round(h * short / w))
    else:
        new_h = short
        new_w = int(round(w * short / h))
    return img.resize((new_w, new_h), Image.BILINEAR)

def center_crop(img, size=224):
    w, h = img.size
    left = int(round((w - size) / 2.0))
    top  = int(round((h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            inp = pb_utils.get_input_tensor_by_name(req, "IMAGE_BYTES")
            bstr = inp.as_numpy()[0]  # dtype=object
            img = Image.open(io.BytesIO(bstr.tobytes())).convert("RGB")

            img = resize_short_side(img, 256)
            img = center_crop(img, 224)

            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
            arr = (arr - MEAN) / STD
            arr = arr.transpose(2, 0, 1)  # [3,224,224]
            arr = arr.astype(np.float32)

            out = pb_utils.Tensor("pixel_values", arr)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
