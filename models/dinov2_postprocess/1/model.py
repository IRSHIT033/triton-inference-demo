import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    """DINOv2 Postprocessing Model"""
    
    def initialize(self, args):
        """Initialize the model"""
        pass
    
    def execute(self, requests):
        """Execute postprocessing on batch of requests"""
        responses = []
        
        for request in requests:
            # Get input features from TensorRT model
            input_features = pb_utils.get_input_tensor_by_name(request, "FEATURES")
            features_data = input_features.as_numpy()
            
            # Apply L2 normalization
            normalized_features = features_data / np.linalg.norm(features_data, axis=1, keepdims=True)
            
            # Create output tensor
            output_tensor = pb_utils.Tensor("NORMALIZED_FEATURES", normalized_features)
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)
        
        return responses
    
    def finalize(self):
        """Clean up resources"""
        pass 