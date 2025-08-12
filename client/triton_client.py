#!/usr/bin/env python3
"""
Triton Inference Client for DINO V2 ViT model.
"""

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import argparse
import time
from pathlib import Path
from typing import List, Union, Tuple
import json

class DinoV2TritonClient:
    """Client for DINO V2 model inference using Triton."""
    
    def __init__(self, 
                 triton_url: str = "localhost:8000",
                 model_name: str = "dinov2_vit",
                 protocol: str = "http"):
        """
        Initialize Triton client.
        
        Args:
            triton_url: Triton server URL
            model_name: Name of the model in Triton
            protocol: Communication protocol ("http" or "grpc")
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.protocol = protocol
        
        # Initialize client
        if protocol == "grpc":
            self.client = grpcclient.InferenceServerClient(url=triton_url)
        else:
            self.client = httpclient.InferenceServerClient(url=triton_url)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Check server health
        self._check_server_health()
        
    def _check_server_health(self):
        """Check if Triton server is healthy and model is loaded."""
        try:
            if self.client.is_server_live():
                print("✓ Triton server is live")
            else:
                raise Exception("Triton server is not live")
                
            if self.client.is_server_ready():
                print("✓ Triton server is ready")
            else:
                raise Exception("Triton server is not ready")
                
            if self.client.is_model_ready(self.model_name):
                print(f"✓ Model '{self.model_name}' is ready")
            else:
                raise Exception(f"Model '{self.model_name}' is not ready")
                
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for DINO V2 inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image array
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Assume PIL Image
        
        # Apply transformations
        tensor = self.transform(image)
        
        # Convert to numpy and add batch dimension
        return tensor.numpy()
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batched preprocessed images
        """
        batch = []
        for path in image_paths:
            img_array = self.preprocess_image(path)
            batch.append(img_array)
        
        return np.stack(batch, axis=0)
    
    def infer(self, images: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed images.
        
        Args:
            images: Preprocessed image batch [batch_size, 3, 224, 224]
            
        Returns:
            Feature embeddings [batch_size, 768]
        """
        # Prepare input
        if self.protocol == "grpc":
            inputs = [grpcclient.InferInput("images", images.shape, "FP32")]
            outputs = [grpcclient.InferRequestedOutput("features")]
        else:
            inputs = [httpclient.InferInput("images", images.shape, "FP32")]
            outputs = [httpclient.InferRequestedOutput("features")]
        
        inputs[0].set_data_from_numpy(images)
        
        # Run inference
        start_time = time.time()
        result = self.client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs
        )
        inference_time = time.time() - start_time
        
        # Extract features
        features = result.as_numpy("features")
        
        print(f"Inference completed in {inference_time:.4f}s")
        print(f"Output shape: {features.shape}")
        
        return features
    
    def infer_single(self, image_path: str) -> np.ndarray:
        """
        Convenient method for single image inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Feature embedding [768]
        """
        # Preprocess
        image_array = self.preprocess_image(image_path)
        batch = np.expand_dims(image_array, axis=0)
        
        # Infer
        features = self.infer(batch)
        
        return features[0]  # Return single feature vector
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return float(similarity)
    
    def get_model_metadata(self):
        """Get model metadata from Triton server."""
        try:
            metadata = self.client.get_model_metadata(self.model_name)
            return metadata
        except Exception as e:
            print(f"Failed to get model metadata: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="DINO V2 Triton Client")
    parser.add_argument("--triton-url", default="localhost:8000",
                       help="Triton server URL")
    parser.add_argument("--model-name", default="dinov2_vit",
                       help="Model name in Triton")
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http",
                       help="Communication protocol")
    parser.add_argument("--image", required=True,
                       help="Path to input image")
    parser.add_argument("--compare-image", 
                       help="Optional second image for similarity comparison")
    parser.add_argument("--batch-images", nargs="+",
                       help="Multiple images for batch inference")
    parser.add_argument("--output", 
                       help="Output file to save features")
    
    args = parser.parse_args()
    
    # Initialize client
    client = DinoV2TritonClient(
        triton_url=args.triton_url,
        model_name=args.model_name,
        protocol=args.protocol
    )
    
    # Print model metadata
    metadata = client.get_model_metadata()
    if metadata:
        print(f"\nModel: {metadata.name}")
        print(f"Platform: {metadata.platform}")
        print(f"Versions: {[v.name for v in metadata.versions]}")
    
    if args.batch_images:
        # Batch inference
        print(f"\nRunning batch inference on {len(args.batch_images)} images...")
        batch_array = client.preprocess_batch(args.batch_images)
        features = client.infer(batch_array)
        
        print(f"Extracted features for {len(features)} images")
        
        if args.output:
            np.save(args.output, features)
            print(f"Features saved to {args.output}")
            
    else:
        # Single image inference
        print(f"\nRunning inference on: {args.image}")
        features = client.infer_single(args.image)
        
        print(f"Feature vector shape: {features.shape}")
        print(f"Feature norm: {np.linalg.norm(features):.4f}")
        
        if args.compare_image:
            print(f"\nComparing with: {args.compare_image}")
            features2 = client.infer_single(args.compare_image)
            
            similarity = client.compute_similarity(features, features2)
            print(f"Cosine similarity: {similarity:.4f}")
        
        if args.output:
            np.save(args.output, features)
            print(f"Features saved to {args.output}")

if __name__ == "__main__":
    main() 