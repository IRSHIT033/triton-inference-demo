#!/usr/bin/env python3
"""
Batch Image Testing Client for DinoV2 Triton Server
Supports processing multiple images in batches for efficient inference
"""

import os
import glob
import time
import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.http as httpclient
import argparse
from typing import List, Tuple

def create_sample_images(num_images: int = 5, output_dir: str = "sample_images") -> List[str]:
    """Create sample images for testing if none exist"""
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    
    for i in range(num_images):
        img_path = os.path.join(output_dir, f"sample_{i+1:03d}.jpg")
        if not os.path.exists(img_path):
            print(f"Creating sample image: {img_path}")
            # Create different colored gradient images
            img = Image.new('RGB', (224, 224))
            pixels = img.load()
            for x in range(224):
                for y in range(224):
                    # Create different patterns for each image
                    r = (x + i * 50) % 256
                    g = (y + i * 30) % 256
                    b = (x + y + i * 20) % 256
                    pixels[x, y] = (r, g, b)
            img.save(img_path)
        image_paths.append(img_path)
    
    print(f"Created/found {len(image_paths)} sample images in {output_dir}")
    return image_paths

def load_images_from_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """Load all image files from a directory"""
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, ext)
        image_paths.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern = os.path.join(directory, ext.upper())
        image_paths.extend(glob.glob(pattern, recursive=False))
    
    return sorted(list(set(image_paths)))  # Remove duplicates and sort

def prepare_batch_input(image_paths: List[str]) -> np.ndarray:
    """Prepare batch input from list of image paths"""
    batch_size = len(image_paths)
    input_data = np.empty((batch_size, 1), dtype=object)
    
    for i, img_path in enumerate(image_paths):
        print(f"Loading image {i+1}/{batch_size}: {os.path.basename(img_path)}")
        with open(img_path, "rb") as f:
            image_data = f.read()
        input_data[i, 0] = image_data
    
    return input_data

def run_batch_inference(triton_client: InferenceServerClient, 
                       image_paths: List[str], 
                       model_name: str = "dinov2_ensemble") -> Tuple[np.ndarray, float]:
    """Run batch inference on a list of images"""
    
    print(f"\nProcessing batch of {len(image_paths)} images...")
    
    # Prepare batch input
    input_data = prepare_batch_input(image_paths)
    print(f"Batch input shape: {input_data.shape}")
    
    # Create Triton input tensor
    input_tensor = InferInput("IMAGE", input_data.shape, "BYTES")
    input_tensor.set_data_from_numpy(input_data)
    
    # Prepare output
    output_tensor = InferRequestedOutput("features")
    
    # Run inference with timing
    start_time = time.time()
    response = triton_client.infer(model_name, [input_tensor], outputs=[output_tensor])
    inference_time = time.time() - start_time
    
    # Get results
    features = response.as_numpy("features")
    
    return features, inference_time

def process_images_in_batches(image_paths: List[str], 
                            batch_size: int = 8,
                            model_name: str = "dinov2_ensemble") -> None:
    """Process images in batches and display results"""
    
    # Create Triton client
    triton_client = InferenceServerClient("localhost:8000")
    
    # Check server status
    if not triton_client.is_server_live():
        raise Exception("Triton server is not live")
    
    if not triton_client.is_model_ready(model_name):
        raise Exception(f"Model {model_name} is not ready")
    
    print(f"Server and model '{model_name}' are ready")
    print(f"Processing {len(image_paths)} images in batches of {batch_size}")
    
    total_images = len(image_paths)
    total_time = 0
    all_features = []
    
    # Process images in batches
    for i in range(0, total_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_images + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} - Processing {len(batch_paths)} images")
        print(f"{'='*60}")
        
        try:
            features, inference_time = run_batch_inference(triton_client, batch_paths, model_name)
            total_time += inference_time
            all_features.append(features)
            
            # Display results for this batch
            print(f"✅ Batch {batch_num} completed successfully!")
            print(f"   Inference time: {inference_time:.3f} seconds")
            print(f"   Features shape: {features.shape}")
            print(f"   Features dtype: {features.dtype}")
            print(f"   Images per second: {len(batch_paths)/inference_time:.2f}")
            
            # Show sample features for first image in batch
            if len(features) > 0:
                sample_features = features[0]
                print(f"   Sample features (first 5): {sample_features[:5]}")
                print(f"   Feature vector length: {len(sample_features)}")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Total processing time: {total_time:.3f} seconds")
    print(f"Average time per image: {total_time/total_images:.3f} seconds")
    print(f"Overall throughput: {total_images/total_time:.2f} images/second")
    
    if all_features:
        combined_features = np.vstack(all_features)
        print(f"Combined features shape: {combined_features.shape}")
        return combined_features

def main():
    parser = argparse.ArgumentParser(description="Batch Image Testing for DinoV2 Triton Server")
    parser.add_argument("--images", "-i", type=str, 
                       help="Directory containing images to process")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                       help="Batch size for processing (default: 8, max: 32)")
    parser.add_argument("--create-samples", "-s", type=int, default=0,
                       help="Create N sample images for testing (default: 0)")
    parser.add_argument("--model", "-m", type=str, default="dinov2_ensemble",
                       help="Model name to use (default: dinov2_ensemble)")
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch_size > 32:
        print("Warning: Batch size > 32 may not be supported. Setting to 32.")
        args.batch_size = 32
    
    try:
        image_paths = []
        
        # Create sample images if requested
        if args.create_samples > 0:
            print(f"Creating {args.create_samples} sample images...")
            sample_paths = create_sample_images(args.create_samples)
            image_paths.extend(sample_paths)
        
        # Load images from directory if provided
        if args.images:
            if os.path.isdir(args.images):
                dir_images = load_images_from_directory(args.images)
                print(f"Found {len(dir_images)} images in {args.images}")
                image_paths.extend(dir_images)
            else:
                print(f"Error: {args.images} is not a valid directory")
                return
        
        # If no images specified, create some samples
        if not image_paths:
            print("No images specified. Creating 10 sample images...")
            image_paths = create_sample_images(10)
        
        if not image_paths:
            print("No images to process!")
            return
        
        print(f"\nFound {len(image_paths)} images to process")
        print(f"Using batch size: {args.batch_size}")
        print(f"Using model: {args.model}")
        
        # Process images
        features = process_images_in_batches(
            image_paths, 
            batch_size=args.batch_size,
            model_name=args.model
        )
        
        print(f"\n✅ Batch processing completed successfully!")
        
    except httpclient.InferenceServerException as e:
        print(f"❌ Triton server error: {e}")
        print("Make sure:")
        print("1. Triton server is running on localhost:8000")
        print("2. The dinov2_ensemble model is loaded")
        print("3. Run: docker-compose up triton")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 