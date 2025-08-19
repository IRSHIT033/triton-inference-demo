# client_ensemble.py
import os
import numpy as np
from PIL import Image
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.http as httpclient

def create_sample_image(img_path="sample_image.jpg", size=(224, 224)):
    """Create a sample image if none exists"""
    if not os.path.exists(img_path):
        print(f"Creating sample image: {img_path}")
        # Create a simple gradient image
        img = Image.new('RGB', size)
        pixels = img.load()
        for i in range(size[0]):
            for j in range(size[1]):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
        img.save(img_path)
        print(f"Sample image saved as {img_path}")
    return img_path

def main():
    try:
        # Create or use existing image
        img_path = "image.jpg"
        if not os.path.exists(img_path):
            img_path = create_sample_image(img_path)
        
        print(f"Using image: {img_path}")
        
        # Read image as raw bytes
        with open(img_path, "rb") as f:
            image_data = f.read()
        
        print(f"Image size: {len(image_data)} bytes")
        
        # Prepare input for Triton
        # The preprocessing model expects raw image bytes
        # Triton TYPE_STRING is used for binary data, stored as numpy object array
        input_data = np.array([[image_data]], dtype=object)  # shape [1,1] for batch_size=1
        
        # Create Triton client
        triton_client = InferenceServerClient("localhost:8000")
        
        # Check if server is live
        if not triton_client.is_server_live():
            raise Exception("Triton server is not live")
        
        # Check if model is ready
        if not triton_client.is_model_ready("dinov2_ensemble"):
            raise Exception("Model dinov2_ensemble is not ready")
        
        print("Server and model are ready")
        
        # Prepare input
        # Use BYTES datatype for binary image data (config shows TYPE_STRING but BYTES works for binary)
        input_tensor = InferInput("IMAGE", input_data.shape, "BYTES")
        input_tensor.set_data_from_numpy(input_data)
        
        # Prepare output
        output_tensor = InferRequestedOutput("features")
        
        # Run inference
        print("Running inference...")
        response = triton_client.infer("dinov2_ensemble", [input_tensor], outputs=[output_tensor])
        
        # Get results
        features = response.as_numpy("features")
        print(f"Features shape: {features.shape}")
        print(f"Features dtype: {features.dtype}")
        print(f"Features sample (first 10 values): {features.flatten()[:10]}")
        
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Make sure the image file exists or let the script create a sample image")
    except httpclient.InferenceServerException as e:
        print(f"Triton server error: {e}")
        print("Make sure Triton server is running on localhost:8000")
        print("And that the dinov2_ensemble model is loaded")
    except Exception as e:
        print(f"Error: {e}")
        print("Check that:")
        print("1. Triton server is running (localhost:8000)")
        print("2. dinov2_ensemble model is loaded")
        print("3. Image file exists or can be created")

if __name__ == "__main__":
    main()
