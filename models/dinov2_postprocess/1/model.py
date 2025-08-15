import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args): pass

    def execute(self, requests):
        responses = []
        for req in requests:
            x = pb_utils.get_input_tensor_by_name(req, "raw_embeds").as_numpy()
            # x: [H] or [N,H] depending on batching; Triton passes per-request tensors here
            if x.ndim == 1:
                x = x[None, ...]
            norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            y = (x / norm).astype(np.float32)
            out = pb_utils.Tensor("embeddings", y[0])
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses
