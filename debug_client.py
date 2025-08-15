#!/usr/bin/env python3

import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io
import base64

def create_different_test_images():
    """Create test images in different formats to debug"""
    # Create a simple test image
    img = Image.new('RGB', (256, 256), color=(255, 0, 0))
    
    formats = {}
    
    # Format 1: JPEG bytes
    buffer1 = io.BytesIO()
    img.save(buffer1, format='JPEG')
    formats['jpeg_bytes'] = buffer1.getvalue()
    
    # Format 2: PNG bytes  
    buffer2 = io.BytesIO()
    img.save(buffer2, format='PNG')
    formats['png_bytes'] = buffer2.getvalue()
    
    # Format 3: Base64 encoded JPEG
    formats['jpeg_base64'] = base64.b64encode(formats['jpeg_bytes'])
    
    return formats

def test_different_formats():
    """Test different input formats"""
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        print("✅ Connected to Triton server")
        
        formats = create_different_test_images()
        
        for format_name, image_data in formats.items():
            print(f"\n🧪 Testing format: {format_name}")
            print(f"   Data type: {type(image_data)}")
            print(f"   Data length: {len(image_data)}")
            
            try:
                # Test if PIL can read it
                test_buffer = io.BytesIO(image_data)
                test_img = Image.open(test_buffer)
                print(f"   ✅ PIL can read: {test_img.size}, {test_img.mode}")
            except Exception as e:
                print(f"   ❌ PIL cannot read: {e}")
                continue
            
            try:
                # Prepare input
                input_data = httpclient.InferInput("IMAGE_BYTES", [1, 1], "BYTES")
                input_data.set_data_from_numpy(np.array([[image_data]], dtype=object))
                
                output_data = httpclient.InferRequestedOutput("EMBEDDINGS")
                
                # Run inference
                result = client.infer(
                    model_name="dinov2_ensemble", 
                    inputs=[input_data],
                    outputs=[output_data]
                )
                
                embeddings = result.as_numpy("EMBEDDINGS")
                print(f"   ✅ Inference successful! Shape: {embeddings.shape}")
                return True, format_name
                
            except Exception as e:
                print(f"   ❌ Inference failed: {str(e)[:100]}...")
                
        return False, None
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False, None

def test_simple_working_case():
    """Test with a very simple case that should work"""
    try:
        client = httpclient.InferenceServerClient(url="localhost:8000")
        
        # Create the simplest possible valid JPEG
        img = Image.new('RGB', (224, 224), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=100)
        image_bytes = buffer.getvalue()
        
        print(f"Image size: {len(image_bytes)} bytes")
        
        # Verify we can read it back
        verify_buffer = io.BytesIO(image_bytes)
        verify_img = Image.open(verify_buffer)
        print(f"Verified image: {verify_img.size}, {verify_img.mode}")
        
        # Try the inference
        input_data = httpclient.InferInput("IMAGE_BYTES", [1, 1], "BYTES")
        input_data.set_data_from_numpy(np.array([[image_bytes]], dtype=object))
        
        output_data = httpclient.InferRequestedOutput("EMBEDDINGS")
        
        result = client.infer(
            model_name="dinov2_ensemble",
            inputs=[input_data], 
            outputs=[output_data]
        )
        
        embeddings = result.as_numpy("EMBEDDINGS")
        print(f"✅ Success! Embeddings shape: {embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        # Let's see what the actual input looks like
        print(f"Input array shape: {np.array([[image_bytes]], dtype=object).shape}")
        print(f"Input array dtype: {np.array([[image_bytes]], dtype=object).dtype}")

if __name__ == "__main__":
    print("🔍 Debugging DINOv2 Input Format")
    print("=" * 40)
    
    print("\n1️⃣ Testing simple working case...")
    test_simple_working_case()
    
    print("\n2️⃣ Testing different formats...")
    success, working_format = test_different_formats()
    
    if success:
        print(f"\n✅ Working format found: {working_format}")
    else:
        print(f"\n❌ No working format found") 