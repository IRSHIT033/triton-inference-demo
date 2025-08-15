#!/usr/bin/env python3

import io
import numpy as np
from PIL import Image

def test_image_processing():
    """Test the exact same process as the preprocessing model"""
    
    # Create test image (same as our client)
    img = Image.new('RGB', (256, 256), color=(128, 64, 192))
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    image_bytes = buffer.getvalue()
    
    print(f"Created image bytes: {len(image_bytes)} bytes")
    print(f"First few bytes: {image_bytes[:20]}")
    
    # Test 1: Direct PIL reading (this should work)
    try:
        test_buffer = io.BytesIO(image_bytes)
        test_img = Image.open(test_buffer).convert("RGB")
        print(f"✅ Direct PIL read works: {test_img.size}")
    except Exception as e:
        print(f"❌ Direct PIL read failed: {e}")
        return
    
    # Test 2: Simulate what the preprocessing model does
    try:
        # Simulate the triton input processing
        # The model does: inp.as_numpy()[0] -> bstr.tobytes()
        
        # First, let's see what happens if we put bytes in a numpy object array
        np_array = np.array([image_bytes], dtype=object)
        bstr = np_array[0]  # This simulates inp.as_numpy()[0]
        
        print(f"bstr type: {type(bstr)}")
        print(f"bstr is bytes: {isinstance(bstr, bytes)}")
        
        # The model calls bstr.tobytes()
        if hasattr(bstr, 'tobytes'):
            processed_bytes = bstr.tobytes()
        else:
            processed_bytes = bstr  # If it's already bytes
            
        print(f"Processed bytes length: {len(processed_bytes)}")
        print(f"Are they the same? {processed_bytes == image_bytes}")
        
        # Test PIL reading of processed bytes
        final_buffer = io.BytesIO(processed_bytes)
        final_img = Image.open(final_buffer).convert("RGB")
        print(f"✅ Preprocessing simulation works: {final_img.size}")
        
    except Exception as e:
        print(f"❌ Preprocessing simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Image Processing Pipeline")
    print("=" * 40)
    test_image_processing() 