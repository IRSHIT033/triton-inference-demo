#!/usr/bin/env python3

import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    # Create a more realistic test image
    img = Image.new('RGB', (256, 256), color=(128, 64, 192))
    
    # Add some pattern to make it more realistic
    import numpy as np
    arr = np.array(img)
    for i in range(0, 256, 32):
        arr[i:i+16, :] = [255, 0, 0]  # Red stripes
    img = Image.fromarray(arr)
    
    # Save as JPEG to BytesIO
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG', quality=95)
    img_bytes = img_buffer.getvalue()
    
    # Verify the image can be read back
    test_buffer = io.BytesIO(img_bytes)
    test_img = Image.open(test_buffer)
    print(f"✅ Created test image: {test_img.size}, mode: {test_img.mode}")
    
    return img_bytes

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
        
        # Follow the exact format from README example
        import base64
        
        # The HTTP example uses base64 encoding and shape [1]
        image_b64_string = base64.b64encode(image_bytes).decode()
        input_data = httpclient.InferInput("IMAGE_BYTES", [1], "BYTES")
        input_data.set_data_from_numpy(np.array([image_b64_string], dtype=object))
        
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