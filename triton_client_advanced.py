#!/usr/bin/env python3

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import numpy as np
from PIL import Image
import cv2
import io
import time
import argparse
import sys
from pathlib import Path
import json

class AdvancedTritonClient:
    def __init__(self, server_url="localhost:8000", use_grpc=False, model_name="dinov2_ensemble"):
        self.server_url = server_url
        self.use_grpc = use_grpc
        self.model_name = model_name
        
        # Initialize the appropriate client
        try:
            if use_grpc:
                # Remove http:// if present and use gRPC port
                clean_url = server_url.replace("http://", "").replace("https://", "")
                if ":8000" in clean_url:
                    clean_url = clean_url.replace(":8000", ":8001")
                elif ":" not in clean_url:
                    clean_url = f"{clean_url}:8001"
                
                self.client = grpcclient.InferenceServerClient(url=clean_url)
                self.protocol = "gRPC"
                print(f"🔗 Connected to Triton server via gRPC: {clean_url}")
            else:
                # Remove http:// prefix for HTTP client (it doesn't expect scheme)
                clean_url = server_url.replace("http://", "").replace("https://", "")
                
                self.client = httpclient.InferenceServerClient(url=clean_url)
                self.protocol = "HTTP"
                print(f"🔗 Connected to Triton server via HTTP: {clean_url}")
                
        except Exception as e:
            print(f"❌ Failed to connect to Triton server: {e}")
            sys.exit(1)
    
    def check_server_health(self):
        """Check server health and readiness"""
        try:
            if self.client.is_server_live():
                print("✅ Server is live")
            else:
                print("❌ Server is not live")
                return False
                
            if self.client.is_server_ready():
                print("✅ Server is ready")
            else:
                print("❌ Server is not ready")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_server_metadata(self):
        """Get server metadata"""
        try:
            metadata = self.client.get_server_metadata()
            print("📊 Server Metadata:")
            if hasattr(metadata, 'name'):
                print(f"   Name: {metadata.name}")
                print(f"   Version: {metadata.version}")
                print(f"   Extensions: {metadata.extensions}")
            else:
                # Handle dictionary response
                print(f"   Name: {metadata.get('name', 'Unknown')}")
                print(f"   Version: {metadata.get('version', 'Unknown')}")
                print(f"   Extensions: {metadata.get('extensions', [])}")
            return metadata
        except Exception as e:
            print(f"❌ Failed to get server metadata: {e}")
            return None
    
    def list_models(self):
        """List all available models"""
        try:
            models = self.client.get_model_repository_index()
            print("📋 Available models:")
            for model in models:
                if hasattr(model, 'name'):
                    print(f"   - {model.name} (state: {model.state})")
                else:
                    # Handle dictionary response
                    print(f"   - {model.get('name', 'Unknown')} (state: {model.get('state', 'Unknown')})")
            return models
        except Exception as e:
            print(f"❌ Failed to list models: {e}")
            return None
    
    def get_model_metadata(self):
        """Get metadata for the specific model"""
        try:
            metadata = self.client.get_model_metadata(self.model_name)
            print(f"🔍 Model '{self.model_name}' Metadata:")
            
            if hasattr(metadata, 'platform'):
                print(f"   Platform: {metadata.platform}")
                print(f"   Versions: {metadata.versions}")
                
                print("   Inputs:")
                for input_meta in metadata.inputs:
                    print(f"     - {input_meta.name}: {input_meta.datatype} {input_meta.shape}")
                
                print("   Outputs:")
                for output_meta in metadata.outputs:
                    print(f"     - {output_meta.name}: {output_meta.datatype} {output_meta.shape}")
            else:
                # Handle dictionary response
                print(f"   Platform: {metadata.get('platform', 'Unknown')}")
                print(f"   Versions: {metadata.get('versions', [])}")
                
                print("   Inputs:")
                for input_meta in metadata.get('inputs', []):
                    if isinstance(input_meta, dict):
                        print(f"     - {input_meta.get('name')}: {input_meta.get('datatype')} {input_meta.get('shape')}")
                    else:
                        print(f"     - {input_meta.name}: {input_meta.datatype} {input_meta.shape}")
                
                print("   Outputs:")
                for output_meta in metadata.get('outputs', []):
                    if isinstance(output_meta, dict):
                        print(f"     - {output_meta.get('name')}: {output_meta.get('datatype')} {output_meta.get('shape')}")
                    else:
                        print(f"     - {output_meta.name}: {output_meta.datatype} {output_meta.shape}")
                
            return metadata
        except Exception as e:
            print(f"❌ Failed to get model metadata: {e}")
            return None
    
    def get_model_config(self):
        """Get model configuration"""
        try:
            config = self.client.get_model_config(self.model_name)
            print(f"⚙️ Model '{self.model_name}' Configuration:")
            
            if hasattr(config, 'max_batch_size'):
                print(f"   Max batch size: {config.max_batch_size}")
                print(f"   Platform: {config.platform}")
                
                if hasattr(config, 'ensemble_scheduling'):
                    print("   Ensemble steps:")
                    for i, step in enumerate(config.ensemble_scheduling.step):
                        print(f"     {i+1}. {step.model_name} (v{step.model_version})")
            else:
                # Handle dictionary response
                print(f"   Max batch size: {config.get('max_batch_size', 'Unknown')}")
                print(f"   Platform: {config.get('platform', 'Unknown')}")
                
                if 'ensemble_scheduling' in config and 'step' in config['ensemble_scheduling']:
                    print("   Ensemble steps:")
                    for i, step in enumerate(config['ensemble_scheduling']['step']):
                        if isinstance(step, dict):
                            print(f"     {i+1}. {step.get('model_name')} (v{step.get('model_version')})")
                        else:
                            print(f"     {i+1}. {step.model_name} (v{step.model_version})")
            
            return config
        except Exception as e:
            print(f"❌ Failed to get model config: {e}")
            return None
    
    def create_test_image(self, size=(224, 224)):
        """Create a test image with more interesting patterns"""
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Create a more complex test pattern
        for y in range(size[1]):
            for x in range(size[0]):
                # Create circular gradient pattern
                center_x, center_y = size[0] // 2, size[1] // 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                r = int(255 * (distance / max_distance))
                g = int(255 * (1 - distance / max_distance))
                b = int(255 * (x / size[0]) * (y / size[1]))
                
                img[y, x] = [r % 255, g % 255, b % 255]
        
        # Convert to PIL Image and then to bytes
        pil_img = Image.fromarray(img)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='JPEG', quality=95)
        return img_buffer.getvalue()
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            print(f"📸 Loaded image: {image_path}")
            return image_bytes
        except Exception as e:
            print(f"❌ Failed to load image {image_path}: {e}")
            return None
    
    def prepare_inputs(self, image_bytes, batch_size=1):
        """Prepare inputs for inference"""
        import base64
        
        # Based on the README example, we need to base64 encode the image bytes
        # The shape should be [batch_size] for the model with max_batch_size: 32
        image_b64_string = base64.b64encode(image_bytes).decode()
        
        if self.use_grpc:
            input_data = grpcclient.InferInput("IMAGE_BYTES", [batch_size], "BYTES")
        else:
            input_data = httpclient.InferInput("IMAGE_BYTES", [batch_size], "BYTES")
        
        # For batch inference, repeat the base64 string
        if batch_size == 1:
            input_data.set_data_from_numpy(np.array([image_b64_string], dtype=object))
        else:
            batch_data = np.array([image_b64_string] * batch_size, dtype=object)
            input_data.set_data_from_numpy(batch_data)
        
        return [input_data]
    
    def prepare_outputs(self):
        """Prepare output requests"""
        if self.use_grpc:
            output_data = grpcclient.InferRequestedOutput("EMBEDDINGS")
        else:
            output_data = httpclient.InferRequestedOutput("EMBEDDINGS")
        
        return [output_data]
    
    def run_inference(self, image_path=None, batch_size=1):
        """Run inference on the model"""
        try:
            # Get image data
            if image_path and Path(image_path).exists():
                image_bytes = self.load_image(image_path)
                if image_bytes is None:
                    return None
            else:
                image_bytes = self.create_test_image()
                print("📸 Using generated test image")
            
            # Prepare inputs and outputs
            inputs = self.prepare_inputs(image_bytes, batch_size)
            outputs = self.prepare_outputs()
            
            print(f"🚀 Running inference (batch_size={batch_size})...")
            start_time = time.time()
            
            # Run inference
            result = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )
            
            inference_time = time.time() - start_time
            
            # Get output data
            embeddings = result.as_numpy("EMBEDDINGS")
            
            print(f"✅ Inference successful!")
            print(f"   Protocol: {self.protocol}")
            print(f"   Response time: {inference_time:.3f}s")
            print(f"   Batch size: {batch_size}")
            print(f"   Output shape: {embeddings.shape}")
            
            if batch_size == 1:
                embedding_vector = embeddings.flatten()
                print(f"   Embedding dimension: {len(embedding_vector)}")
                print(f"   Embedding range: [{embedding_vector.min():.6f}, {embedding_vector.max():.6f}]")
                print(f"   L2 norm: {np.linalg.norm(embedding_vector):.6f}")
                
                # Check if properly normalized
                norm = np.linalg.norm(embedding_vector)
                if 0.95 <= norm <= 1.05:
                    print("✅ Embeddings are properly normalized")
                else:
                    print(f"⚠️  Embeddings may not be normalized (norm: {norm:.6f})")
            else:
                print(f"   Throughput: {batch_size/inference_time:.2f} images/sec")
                # Check each embedding in the batch
                for i in range(batch_size):
                    embedding = embeddings[i]
                    norm = np.linalg.norm(embedding)
                    print(f"   Batch {i+1} norm: {norm:.6f}")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None
    
    def benchmark_inference(self, num_requests=10, batch_size=1):
        """Benchmark inference performance"""
        print(f"🏃 Running benchmark: {num_requests} requests, batch_size={batch_size}")
        
        # Create test image once
        image_bytes = self.create_test_image()
        inputs = self.prepare_inputs(image_bytes, batch_size)
        outputs = self.prepare_outputs()
        
        response_times = []
        successful_requests = 0
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                result = self.client.infer(
                    model_name=self.model_name,
                    inputs=inputs,
                    outputs=outputs
                )
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                successful_requests += 1
                
                if (i + 1) % 5 == 0:
                    print(f"   Completed {i + 1}/{num_requests} requests")
                    
            except Exception as e:
                print(f"   Request {i + 1} failed: {e}")
        
        if successful_requests > 0:
            avg_time = np.mean(response_times)
            p95_time = np.percentile(response_times, 95)
            p99_time = np.percentile(response_times, 99)
            throughput = (successful_requests * batch_size) / sum(response_times)
            
            print(f"📊 Benchmark Results:")
            print(f"   Successful requests: {successful_requests}/{num_requests}")
            print(f"   Average response time: {avg_time:.3f}s")
            print(f"   95th percentile: {p95_time:.3f}s")
            print(f"   99th percentile: {p99_time:.3f}s")
            print(f"   Throughput: {throughput:.2f} images/sec")
        else:
            print("❌ No successful requests in benchmark")
    
    def run_comprehensive_test(self, image_path=None):
        """Run comprehensive test suite"""
        print(f"🧪 Starting Comprehensive Triton Client Test ({self.protocol})")
        print("=" * 60)
        
        # Test 1: Server Health
        print("\n1️⃣ Server Health Check...")
        if not self.check_server_health():
            return False
        
        # Test 2: Server Metadata
        print("\n2️⃣ Server Metadata...")
        self.get_server_metadata()
        
        # Test 3: List Models
        print("\n3️⃣ Available Models...")
        self.list_models()
        
        # Test 4: Model Metadata
        print(f"\n4️⃣ Model Metadata...")
        self.get_model_metadata()
        
        # Test 5: Model Configuration
        print(f"\n5️⃣ Model Configuration...")
        self.get_model_config()
        
        # Test 6: Single Inference
        print(f"\n6️⃣ Single Image Inference...")
        embeddings = self.run_inference(image_path, batch_size=1)
        if embeddings is None:
            return False
        
        # Test 7: Batch Inference
        print(f"\n7️⃣ Batch Inference...")
        batch_embeddings = self.run_inference(image_path, batch_size=4)
        if batch_embeddings is None:
            print("⚠️  Batch inference failed")
        
        # Test 8: Performance Benchmark
        print(f"\n8️⃣ Performance Benchmark...")
        self.benchmark_inference(num_requests=10, batch_size=1)
        
        print("\n" + "=" * 60)
        print("✅ Comprehensive test completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Advanced Triton Client for DINOv2")
    parser.add_argument("--server-url", default="localhost:8000", 
                       help="Triton server URL")
    parser.add_argument("--use-grpc", action="store_true", 
                       help="Use gRPC instead of HTTP")
    parser.add_argument("--model-name", default="dinov2_ensemble", 
                       help="Model name to test")
    parser.add_argument("--image", help="Path to test image file")
    parser.add_argument("--batch-size", type=int, default=1, 
                       help="Batch size for inference")
    parser.add_argument("--benchmark", action="store_true", 
                       help="Run performance benchmark")
    parser.add_argument("--num-requests", type=int, default=10, 
                       help="Number of requests for benchmark")
    
    args = parser.parse_args()
    
    # Create client
    client = AdvancedTritonClient(
        server_url=args.server_url,
        use_grpc=args.use_grpc,
        model_name=args.model_name
    )
    
    if args.benchmark:
        client.benchmark_inference(args.num_requests, args.batch_size)
    else:
        client.run_comprehensive_test(args.image)

if __name__ == "__main__":
    main() 