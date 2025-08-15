#!/usr/bin/env python3
"""
Simple HTTP example for Triton inference with DINOv2

This is a minimal example showing how to use the HTTP API directly
without the full client class.

Usage:
    python simple_http_example.py path/to/image.jpg
"""

import base64
import json
import sys
import requests


def simple_triton_inference(image_path: str, server_url: str = "http://localhost:8000"):
    """
    Simple function to perform DINOv2 inference via HTTP
    
    Args:
        image_path: Path to the image file
        server_url: Triton server URL
        
    Returns:
        Numpy array of embeddings
    """
    # 1. Read and encode the image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 2. Prepare the request payload
    payload = {
        "inputs": [
            {
                "name": "IMAGE_BYTES",
                "shape": [1, 1],
                "datatype": "BYTES", 
                "data": [image_b64]
            }
        ]
    }
    
    # 3. Send the request
    response = requests.post(
        f"{server_url}/v2/models/dinov2_ensemble/infer",
        json=payload,
        timeout=30
    )
    
    # 4. Check response
    if response.status_code != 200:
        raise RuntimeError(f"Request failed with status {response.status_code}: {response.text}")
    
    result = response.json()
    
    # 5. Extract embeddings
    embeddings_data = result["outputs"][0]["data"]
    
    return embeddings_data


def main():
    if len(sys.argv) != 2:
        print("Usage: python simple_http_example.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # Perform inference
        embeddings = simple_triton_inference(image_path)
        
        print(f"✓ Inference successful!")
        print(f"Got {len(embeddings)} embedding values")
        print(f"First 5 values: {embeddings[:5]}")
        
        # Calculate L2 norm
        import math
        norm = math.sqrt(sum(x*x for x in embeddings))
        print(f"L2 norm: {norm:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Error: Request failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 