import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args): 
        pass

    def execute(self, requests):
        responses = []
        for req in requests:
            try:
                x = pb_utils.get_input_tensor_by_name(req, "raw_embeds").as_numpy()
                # x: [H] or [N,H] depending on batching
                
                # Ensure we have the right dimensions
                if x.ndim == 1:
                    # Single embedding: [768] -> [1, 768]
                    x = x[np.newaxis, :]
                    was_single = True
                else:
                    # Batch of embeddings: [N, 768]
                    was_single = False
                
                # L2 normalization along the feature dimension
                norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
                y = (x / norm).astype(np.float32)
                
                # Return the appropriate shape
                if was_single:
                    # Return single embedding: [768]
                    output_array = y[0]
                else:
                    # Return batch of embeddings: [N, 768]
                    output_array = y
                
                out = pb_utils.Tensor("embeddings", output_array)
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
                
            except Exception as e:
                # Create error response
                error_msg = f"Postprocessing failed: {str(e)}"
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                )
                responses.append(error_response)
        
        return responses
