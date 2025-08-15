import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class TritonPythonModel:
    """DINOv2 Preprocessing Model"""
    
    def initialize(self, args):
        """Initialize the model"""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def execute(self, requests):
        """Execute preprocessing on batch of requests"""
        responses = []
        
        for request in requests:
            # Get input image
            input_image = pb_utils.get_input_tensor_by_name(request, "INPUT_IMAGE")
            image_data = input_image.as_numpy()
            
            # Process image
            processed_images = []
            for img in image_data:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(img.astype('uint8'))
                # Apply transforms
                processed_tensor = self.transform(pil_image)
                processed_images.append(processed_tensor.numpy())
            
            # Stack batch
            batch_tensor = np.stack(processed_images)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("PREPROCESSED_IMAGE", batch_tensor)
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        
        return responses
    
    def finalize(self):
        """Clean up resources"""
        pass 