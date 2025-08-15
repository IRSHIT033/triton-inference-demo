#!/usr/bin/env python3
# infer_dinov2_ensemble.py
#
# pip install "tritonclient[grpc]" numpy
#
# Example:
#   python infer_dinov2_ensemble.py \
#       --url localhost:8001 \
#       --model dinov2_ensemble \
#       --out embeddings.npy \
#       path/to/img1.jpg path/to/img2.png ...

import argparse
import os
import sys
from typing import List

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

def load_image_bytes(paths: List[str]) -> np.ndarray:
    """
    Returns a numpy array of shape [batch, 1] with dtype=object,
    where each element is the raw bytes of an image file.
    """
    if not paths:
        raise ValueError("No image paths provided.")
    if len(paths) > 32:
        raise ValueError(f"Batch size {len(paths)} exceeds model max_batch_size=32.")

    samples = []
    for p in paths:
        with open(p, "rb") as f:
            img_bytes = f.read()
        # Each sample is a length-1 vector per your dims: [1]
        samples.append([img_bytes])

    arr = np.array(samples, dtype=object)  # shape [B, 1]
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8001", help="Triton gRPC endpoint host:port")
    parser.add_argument("--model", default="dinov2_ensemble", help="Triton model name")
    parser.add_argument("--model-version", default="", help="Model version (empty = latest)")
    parser.add_argument("--out", default="", help="Optional path to save embeddings as .npy")
    parser.add_argument("images", nargs="+", help="Image file paths (batch size ≤ 32)")
    args = parser.parse_args()

    # Validate inputs exist
    missing = [p for p in args.images if not os.path.isfile(p)]
    if missing:
        print(f"Error: the following files do not exist:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(1)

    # Prepare input tensor
    image_bytes = load_image_bytes(args.images)  # shape [B,1], dtype=object

    # Create Triton client
    try:
        triton = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    except Exception as e:
        print(f"Failed to create Triton client: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare inputs/outputs
    infer_inputs = []
    infer_outputs = []

    # NOTE: For BYTES/string tensors, Triton uses datatype "BYTES"
    inp = grpcclient.InferInput("IMAGE_BYTES", image_bytes.shape, "BYTES")
    inp.set_data_from_numpy(image_bytes)
    infer_inputs.append(inp)

    # Request the ensemble output
    out = grpcclient.InferOutput("EMBEDDINGS", binary_data=True)
    infer_outputs.append(out)

    # Send request
    try:
        result = triton.infer(
            model_name=args.model,
            inputs=infer_inputs,
            outputs=infer_outputs,
            model_version=args.model_version if args.model_version else None,
        )
    except InferenceServerException as e:
        print(f"Inference failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Fetch and inspect output
    embeds = result.as_numpy("EMBEDDINGS")  # Expect shape [B, 768], dtype float32
    if embeds is None:
        print("No 'EMBEDDINGS' output found.", file=sys.stderr)
        sys.exit(1)

    print(f"Inference OK. Got embeddings array with shape {embeds.shape}, dtype {embeds.dtype}")
    for i, p in enumerate(args.images):
        # Print a tiny summary per item
        v = embeds[i]
        norm = float(np.linalg.norm(v))
        print(f"[{i:02d}] {os.path.basename(p)} -> dim={v.size}, norm={norm:.4f}, first3={v[:3].tolist()}")

    if args.out:
        np.save(args.out, embeds)
        print(f"Saved embeddings to: {args.out}")

if __name__ == "__main__":
    main()
