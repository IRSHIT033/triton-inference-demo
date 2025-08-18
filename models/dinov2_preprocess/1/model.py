import io
import base64
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
    return img.resize((new_w, new_h), Image.LANCZOS)

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
            try:
                inp = pb_utils.get_input_tensor_by_name(req, "IMAGE_BYTES")
                input_data = inp.as_numpy()
                
                # Handle different input shapes - could be [1] or [batch_size]
                if input_data.ndim == 1:
                    batch_size = len(input_data)
                    processed_images = []
                    
                    for i in range(batch_size):
                        bstr = input_data[i]
                        
                        # Handle both base64-encoded strings (HTTP) and raw bytes (gRPC)
                        if isinstance(bstr, (str, np.str_)):
                            # HTTP client sends base64-encoded strings
                            image_bytes = base64.b64decode(bstr)
                        elif isinstance(bstr, bytes):
                            # Already bytes
                            image_bytes = bstr
                        else:
                            # gRPC client sends raw bytes as numpy array
                            image_bytes = bstr.tobytes() if hasattr(bstr, 'tobytes') else bytes(bstr)
                        
                        # Load and process image
                        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        img = resize_short_side(img, 256)
                        img = center_crop(img, 224)
                        
                        # Convert to array and normalize
                        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
                        arr = (arr - MEAN) / STD
                        arr = arr.transpose(2, 0, 1)  # [3,224,224]
                        processed_images.append(arr)
                    
                    # Stack into batch: [batch_size, 3, 224, 224]
                    if batch_size == 1:
                        output_array = processed_images[0][np.newaxis, ...]  # Add batch dimension
                    else:
                        output_array = np.stack(processed_images, axis=0)
                else:
                    # Handle single image case (shouldn't happen with current config but good to be safe)
                    bstr = input_data.item()
                    
                    if isinstance(bstr, (str, np.str_)):
                        image_bytes = base64.b64decode(bstr)
                    elif isinstance(bstr, bytes):
                        image_bytes = bstr
                    else:
                        image_bytes = bstr.tobytes() if hasattr(bstr, 'tobytes') else bytes(bstr)
                    
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img = resize_short_side(img, 256)
                    img = center_crop(img, 224)
                    
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    arr = (arr - MEAN) / STD
                    arr = arr.transpose(2, 0, 1)
                    output_array = arr[np.newaxis, ...]  # Add batch dimension
                
                output_array = output_array.astype(np.float32)
                out = pb_utils.Tensor("pixel_values", output_array)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
                
            except Exception as e:
                # Create error response
                error_msg = f"Preprocessing failed: {str(e)}"
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(error_response)
        
        return responses
