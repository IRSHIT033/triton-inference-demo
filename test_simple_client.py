#!/usr/bin/env python3

import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    return img_buffer.getvalue()

def test_simple_inference():
    """Simple inference test"""
    try:
        # Connect to server
        client = httpclient.InferenceServerClient(url="localhost:8000")
        print("✅ Connected to Triton server")
        
        # Check if server is ready
        if not client.is_server_ready():
            print("❌ Server is not ready")
            return
        print("✅ Server is ready")
        
        # Create test image
        image_bytes = create_test_image()
        print("✅ Created test image")
        
        # Prepare input with correct shape [1, 1] for batch_size=1
        input_data = httpclient.InferInput("IMAGE_BYTES", [1, 1], "BYTES")
        input_data.set_data_from_numpy(np.array([[image_bytes]], dtype=object))
        
        # Prepare output
        output_data = httpclient.InferRequestedOutput("EMBEDDINGS")
        
        print("🚀 Running inference...")
        
        # Run inference
        result = client.infer(
            model_name="dinov2_ensemble",
            inputs=[input_data],
            outputs=[output_data]
        )
        
        # Get embeddings
        embeddings = result.as_numpy("EMBEDDINGS")
        
        print("✅ Inference successful!")
        print(f"   Output shape: {embeddings.shape}")
        print(f"   Embedding dimension: {embeddings.size}")
        print(f"   L2 norm: {np.linalg.norm(embeddings):.6f}")
        print(f"   Sample values: {embeddings.flatten()[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple DINOv2 Triton Test")
    print("=" * 30)
    test_simple_inference() 