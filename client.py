# client_ensemble.py
import base64
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

img_path = "image.jpg"
with open(img_path, "rb") as f:
    data = f.read()

# Triton expects raw bytes as string for TYPE_STRING
# Use latin1 encoding to preserve all byte values when converting to string
str_data = data.decode('latin1')  # Convert bytes to string preserving all byte values
arr = np.array([[str_data]], dtype=object)  # shape [1,1] - string data

cli = InferenceServerClient("localhost:8000")
inp = InferInput("IMAGE", arr.shape, "STRING")
inp.set_data_from_numpy(arr)

out = InferRequestedOutput("features")
resp = cli.infer("dinov2_ensemble", [inp], outputs=[out])
feats = resp.as_numpy("features")
print("features:", feats.shape, feats.dtype)
