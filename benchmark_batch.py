#!/usr/bin/env python3
"""
Benchmark script to compare single vs batch processing performance
"""

import time
import numpy as np
from batch_client import create_sample_images, run_batch_inference
from client import main as single_client_main
from tritonclient.http import InferenceServerClient
import os
import tempfile
import sys

def benchmark_single_processing(image_paths, triton_client):
    """Benchmark processing images one by one"""
    print(f"\n🔄 Benchmarking SINGLE image processing...")
    print(f"Processing {len(image_paths)} images individually")
    
    total_time = 0
    successful_inferences = 0
    
    for i, img_path in enumerate(image_paths):
        try:
            start_time = time.time()
            
            # Read image
            with open(img_path, "rb") as f:
                image_data = f.read()
            
            # Prepare single image input
            input_data = np.array([[image_data]], dtype=object)
            
            # Run inference (similar to original client.py)
            from tritonclient.http import InferInput, InferRequestedOutput
            input_tensor = InferInput("IMAGE", input_data.shape, "BYTES")
            input_tensor.set_data_from_numpy(input_data)
            output_tensor = InferRequestedOutput("features")
            
            response = triton_client.infer("dinov2_ensemble", [input_tensor], outputs=[output_tensor])
            features = response.as_numpy("features")
            
            inference_time = time.time() - start_time
            total_time += inference_time
            successful_inferences += 1
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"  Error processing image {i+1}: {e}")
    
    avg_time = total_time / successful_inferences if successful_inferences > 0 else 0
    throughput = successful_inferences / total_time if total_time > 0 else 0
    
    print(f"✅ Single processing results:")
    print(f"   Total time: {total_time:.3f} seconds")
    print(f"   Successful inferences: {successful_inferences}/{len(image_paths)}")
    print(f"   Average time per image: {avg_time:.3f} seconds")
    print(f"   Throughput: {throughput:.2f} images/second")
    
    return total_time, successful_inferences, throughput

def benchmark_batch_processing(image_paths, triton_client, batch_size=8):
    """Benchmark processing images in batches"""
    print(f"\n🚀 Benchmarking BATCH processing (batch_size={batch_size})...")
    
    total_time = 0
    successful_inferences = 0
    
    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        try:
            features, inference_time = run_batch_inference(triton_client, batch_paths)
            total_time += inference_time
            successful_inferences += len(batch_paths)
            
        except Exception as e:
            print(f"  Error processing batch starting at image {i+1}: {e}")
    
    avg_time = total_time / successful_inferences if successful_inferences > 0 else 0
    throughput = successful_inferences / total_time if total_time > 0 else 0
    
    print(f"✅ Batch processing results:")
    print(f"   Total time: {total_time:.3f} seconds")
    print(f"   Successful inferences: {successful_inferences}/{len(image_paths)}")
    print(f"   Average time per image: {avg_time:.3f} seconds")
    print(f"   Throughput: {throughput:.2f} images/second")
    
    return total_time, successful_inferences, throughput

def main():
    print("🔬 DinoV2 Triton Batch vs Single Processing Benchmark")
    print("=" * 60)
    
    # Configuration
    num_test_images = 20
    batch_sizes_to_test = [4, 8, 16, 32]
    
    try:
        # Create Triton client
        triton_client = InferenceServerClient("localhost:8000")
        
        # Check server status
        if not triton_client.is_server_live():
            raise Exception("Triton server is not live")
        
        if not triton_client.is_model_ready("dinov2_ensemble"):
            raise Exception("Model dinov2_ensemble is not ready")
        
        print(f"✅ Server and model are ready")
        
        # Create test images
        print(f"\n📸 Creating {num_test_images} test images...")
        image_paths = create_sample_images(num_test_images, "benchmark_images")
        
        # Benchmark single processing
        single_time, single_success, single_throughput = benchmark_single_processing(
            image_paths, triton_client
        )
        
        # Benchmark different batch sizes
        batch_results = {}
        for batch_size in batch_sizes_to_test:
            if batch_size <= len(image_paths):
                batch_time, batch_success, batch_throughput = benchmark_batch_processing(
                    image_paths, triton_client, batch_size
                )
                batch_results[batch_size] = {
                    'time': batch_time,
                    'success': batch_success,
                    'throughput': batch_throughput
                }
        
        # Summary comparison
        print(f"\n📊 PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Method':<20} {'Time (s)':<12} {'Throughput (img/s)':<18} {'Speedup':<10}")
        print("-" * 60)
        
        print(f"{'Single':<20} {single_time:<12.3f} {single_throughput:<18.2f} {'1.0x':<10}")
        
        best_throughput = single_throughput
        best_method = "Single"
        
        for batch_size, results in batch_results.items():
            speedup = results['throughput'] / single_throughput if single_throughput > 0 else 0
            print(f"{'Batch-' + str(batch_size):<20} {results['time']:<12.3f} {results['throughput']:<18.2f} {speedup:<10.1f}x")
            
            if results['throughput'] > best_throughput:
                best_throughput = results['throughput']
                best_method = f"Batch-{batch_size}"
        
        print("-" * 60)
        print(f"🏆 Best performance: {best_method} with {best_throughput:.2f} images/second")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 60)
        if best_method != "Single":
            improvement = (best_throughput / single_throughput - 1) * 100
            print(f"✅ Use batch processing for {improvement:.1f}% better performance!")
            print(f"✅ Optimal batch size appears to be: {best_method.split('-')[1] if '-' in best_method else 'N/A'}")
        else:
            print("🤔 Single processing performed best - this might indicate:")
            print("   - Server/GPU not under load (batch benefits minimal)")
            print("   - Network latency dominates (try larger batches)")
            print("   - Model optimization needed")
        
        print(f"\n📋 Usage examples:")
        print(f"   Single image:  python client.py")
        print(f"   Batch images:  python batch_client.py --images /path/to/images --batch-size 8")
        print(f"   Create samples: python batch_client.py --create-samples 20 --batch-size 16")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("\nMake sure:")
        print("1. Triton server is running: docker-compose up triton")
        print("2. Model is loaded properly")
        print("3. Dependencies are installed: pip install -r client_requirements.txt")

if __name__ == "__main__":
    main() 