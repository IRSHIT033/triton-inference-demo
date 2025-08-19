# client_ensemble.py
import base64
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

img_path = "image.jpg"
with open(img_path, "rb") as f:
    data = f.read()

# Triton expects bytes in an object array for TYPE_BYTES
arr = np.array([[np.frombuffer(data, dtype=np.uint8)]], dtype=object)  # shape [1,1]

cli = InferenceServerClient("localhost:8000")
inp = InferInput("IMAGE", arr.shape, "BYTES")
inp.set_data_from_numpy(arr)

out = InferRequestedOutput("features")
resp = cli.infer("dinov2_ensemble", [inp], outputs=[out])
feats = resp.as_numpy("features")
print("features:", feats.shape, feats.dtype)
