#!/usr/bin/env python3
"""
Example: Batch inference with DINO V2 on Triton.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from client.triton_client import DinoV2TritonClient
import numpy as np
import time
from pathlib import Path
import argparse

def run_batch_inference(image_dir: str, triton_url: str = "localhost:8000"):
    """Run batch inference on all images in a directory."""
    
    # Initialize client
    client = DinoV2TritonClient(triton_url=triton_url)
    
    # Find all images in directory
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = [
        str(p) for p in image_dir.iterdir() 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_paths)} images for batch inference")
    
    # Process in batches
    batch_size = 4
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}: {len(batch_paths)} images")
        
        # Preprocess batch
        start_time = time.time()
        batch_array = client.preprocess_batch(batch_paths)
        preprocess_time = time.time() - start_time
        
        # Run inference
        features = client.infer(batch_array)
        
        # Store results
        all_features.append(features)
        
        print(f"Preprocessing time: {preprocess_time:.4f}s")
        print(f"Batch shape: {batch_array.shape}")
        
        # Print individual image info
        for j, path in enumerate(batch_paths):
            feature_norm = np.linalg.norm(features[j])
            print(f"  {Path(path).name}: feature norm = {feature_norm:.4f}")
    
    # Combine all features
    all_features = np.vstack(all_features)
    print(f"\nTotal features extracted: {all_features.shape}")
    
    # Compute similarity matrix
    print("\nComputing pairwise similarities...")
    similarities = compute_similarity_matrix(all_features)
    
    # Find most similar pairs
    most_similar_pairs = find_most_similar_pairs(similarities, image_paths, top_k=3)
    
    print("\nMost similar image pairs:")
    for (idx1, idx2), similarity in most_similar_pairs:
        name1 = Path(image_paths[idx1]).name
        name2 = Path(image_paths[idx2]).name
        print(f"  {name1} <-> {name2}: {similarity:.4f}")
    
    return all_features, similarities

def compute_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix."""
    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized_features = features / (norms + 1e-8)
    
    # Compute similarity matrix
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    
    return similarity_matrix

def find_most_similar_pairs(similarity_matrix: np.ndarray, 
                          image_paths: list, 
                          top_k: int = 5) -> list:
    """Find the most similar image pairs."""
    n = similarity_matrix.shape[0]
    pairs = []
    
    # Extract upper triangular part (excluding diagonal)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append(((i, j), similarity_matrix[i, j]))
    
    # Sort by similarity and return top k
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Batch inference example")
    parser.add_argument("--image-dir", required=True,
                       help="Directory containing images")
    parser.add_argument("--triton-url", default="localhost:8000",
                       help="Triton server URL")
    parser.add_argument("--output", 
                       help="Output file to save features")
    
    args = parser.parse_args()
    
    # Run batch inference
    features, similarities = run_batch_inference(args.image_dir, args.triton_url)
    
    # Save results if requested
    if args.output:
        np.savez(args.output, 
                features=features, 
                similarities=similarities)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 