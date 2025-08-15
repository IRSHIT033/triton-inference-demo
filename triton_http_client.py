#!/usr/bin/env python3
"""
Triton HTTP Inference Client for DINOv2 Ensemble

This script provides a simple HTTP client for inferencing with the DINOv2 ensemble model
running on Triton Inference Server.

Usage:
    python triton_http_client.py --image path/to/image.jpg
    python triton_http_client.py --images path/to/img1.jpg path/to/img2.png --batch
    python triton_http_client.py --url http://localhost:8000 --image test.jpg --save embeddings.npy

Requirements:
    pip install requests numpy pillow
"""

import argparse
import base64
import json
import os
import sys
from typing import List, Optional, Dict, Any

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class TritonHTTPClient:
    """HTTP client for Triton Inference Server"""
    
    def __init__(self, url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the Triton HTTP client.
        
        Args:
            url: Triton server HTTP URL
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def health_check(self) -> bool:
        """Check if the server is ready"""
        try:
            response = self.session.get(f"{self.url}/v2/health/ready", timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            response = self.session.get(f"{self.url}/v2/models", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to list models: {e}")
    
    def get_model_metadata(self, model_name: str, model_version: str = "") -> Dict[str, Any]:
        """Get model metadata"""
        try:
            version_part = f"/versions/{model_version}" if model_version else ""
            response = self.session.get(
                f"{self.url}/v2/models/{model_name}{version_part}", 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get model metadata: {e}")
    
    def infer(
        self, 
        model_name: str, 
        inputs: List[Dict[str, Any]], 
        outputs: Optional[List[Dict[str, str]]] = None,
        model_version: str = ""
    ) -> Dict[str, Any]:
        """
        Send inference request to Triton server.
        
        Args:
            model_name: Name of the model
            inputs: List of input dictionaries
            outputs: Optional list of output specifications
            model_version: Model version (empty for latest)
            
        Returns:
            Inference response as dictionary
        """
        # Prepare request payload
        payload = {
            "inputs": inputs
        }
        
        if outputs:
            payload["outputs"] = outputs
        
        # Send request
        try:
            url = f"{self.url}/v2/models/{model_name}"
            if model_version:
                url += f"/versions/{model_version}"
            url += "/infer"
            
            response = self.session.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            raise RuntimeError(f"Inference request failed: {e}")
    
    def infer_dinov2(self, image_paths: List[str], model_name: str = "dinov2_ensemble") -> np.ndarray:
        """
        High-level method to infer DINOv2 embeddings from image files.
        
        Args:
            image_paths: List of paths to image files
            model_name: Name of the DINOv2 ensemble model
            
        Returns:
            Numpy array of embeddings with shape [batch_size, 768]
        """
        if not image_paths:
            raise ValueError("No image paths provided")
        
        if len(image_paths) > 32:
            raise ValueError(f"Batch size {len(image_paths)} exceeds max batch size of 32")
        
        # Validate all files exist
        missing_files = [path for path in image_paths if not os.path.isfile(path)]
        if missing_files:
            raise FileNotFoundError(f"Image files not found: {missing_files}")
        
        # Load and encode images
        image_data = []
        for path in image_paths:
            with open(path, "rb") as f:
                image_bytes = f.read()
            image_data.append(base64.b64encode(image_bytes).decode('utf-8'))
        
        # Prepare inputs for batch processing
        inputs = [
            {
                "name": "IMAGE_BYTES",
                "shape": [len(image_paths), 1],
                "datatype": "BYTES",
                "data": image_data
            }
        ]
        
        # Send inference request
        result = self.infer(model_name, inputs)
        
        # Extract embeddings
        if "outputs" not in result or not result["outputs"]:
            raise RuntimeError("No outputs received from inference")
        
        embeddings_output = None
        for output in result["outputs"]:
            if output["name"] == "EMBEDDINGS":
                embeddings_output = output
                break
        
        if embeddings_output is None:
            raise RuntimeError("EMBEDDINGS output not found in response")
        
        # Convert to numpy array
        embeddings_data = embeddings_output["data"]
        shape = embeddings_output["shape"]
        
        embeddings = np.array(embeddings_data, dtype=np.float32).reshape(shape)
        return embeddings


def load_single_image_bytes(image_path: str) -> str:
    """Load and base64 encode a single image file"""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    return base64.b64encode(image_bytes).decode('utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Triton HTTP Client for DINOv2 Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single image inference
    python triton_http_client.py --image test.jpg
    
    # Batch inference
    python triton_http_client.py --images img1.jpg img2.png img3.jpg --batch
    
    # Custom server URL and save results
    python triton_http_client.py --url http://localhost:8000 --image test.jpg --save embeddings.npy
    
    # Check server health
    python triton_http_client.py --health
    
    # List available models
    python triton_http_client.py --list-models
        """
    )
    
    # Server configuration
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Triton server HTTP URL (default: http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Request timeout in seconds (default: 30)")
    
    # Model configuration
    parser.add_argument("--model", default="dinov2_ensemble",
                       help="Model name (default: dinov2_ensemble)")
    parser.add_argument("--model-version", default="",
                       help="Model version (default: latest)")
    
    # Image inputs
    parser.add_argument("--image", type=str,
                       help="Single image file path")
    parser.add_argument("--images", nargs="+",
                       help="Multiple image file paths for batch processing")
    parser.add_argument("--batch", action="store_true",
                       help="Use batch processing mode for multiple images")
    
    # Output options
    parser.add_argument("--save", type=str,
                       help="Save embeddings to .npy file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # Utility options
    parser.add_argument("--health", action="store_true",
                       help="Check server health and exit")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # Create client
    try:
        client = TritonHTTPClient(url=args.url, timeout=args.timeout)
    except Exception as e:
        print(f"Error: Failed to create Triton client: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Health check
    if args.health:
        if client.health_check():
            print("✓ Triton server is ready")
            sys.exit(0)
        else:
            print("✗ Triton server is not ready", file=sys.stderr)
            sys.exit(1)
    
    # List models
    if args.list_models:
        try:
            models = client.list_models()
            print("Available models:")
            for model in models:
                print(f"  - {model['name']} (versions: {model.get('versions', 'N/A')})")
        except Exception as e:
            print(f"Error: Failed to list models: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    
    # Determine image inputs
    image_paths = []
    if args.image:
        image_paths = [args.image]
    elif args.images:
        image_paths = args.images
    else:
        print("Error: No image inputs provided. Use --image or --images", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Check server health before inference
    if not client.health_check():
        print("Warning: Triton server health check failed. Proceeding anyway...", file=sys.stderr)
    
    # Perform inference
    try:
        if args.verbose:
            print(f"Processing {len(image_paths)} image(s)...")
            for i, path in enumerate(image_paths):
                print(f"  [{i+1}] {path}")
        
        # Use batch processing or single image processing
        if args.batch or len(image_paths) > 1:
            embeddings = client.infer_dinov2(image_paths, args.model)
        else:
            # Single image inference
            embeddings = client.infer_dinov2(image_paths, args.model)
        
        # Display results
        print(f"\n✓ Inference successful!")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        
        # Show summary for each image
        for i, path in enumerate(image_paths):
            embed_vector = embeddings[i] if len(embeddings.shape) > 1 else embeddings
            norm = float(np.linalg.norm(embed_vector))
            print(f"[{i+1:02d}] {os.path.basename(path)}: norm={norm:.4f}, first3={embed_vector[:3].tolist()}")
        
        # Save if requested
        if args.save:
            np.save(args.save, embeddings)
            print(f"\n💾 Embeddings saved to: {args.save}")
            
    except Exception as e:
        print(f"Error: Inference failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 