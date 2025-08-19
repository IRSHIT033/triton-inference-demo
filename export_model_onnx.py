# export_dinov2_onnx.py
import torch
from torch.onnx import DynamicDim  # if not available in your version, see Option 2 below

# 1) load dinov2
base = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True, source='github').eval().cuda()

# 2) wrapper -> [N, D] (CLS)
class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        out = self.m.forward_features(x)  # dict
        for k in ("x_norm_clstoken", "cls_token", "x_cls", "x"):
            if isinstance(out, dict) and k in out:
                x = out[k]; break
        else:
            x = next(v for v in out.values() if isinstance(v, torch.Tensor))
        return torch.nn.functional.normalize(x, dim=-1)

model = Wrapper(base).eval().cuda()

# 3) sample input
dummy = torch.randn(1, 3, 224, 224, device="cuda")

# 4) dynamic batch (1..32)
dyn = {
    "input":   {0: DynamicDim(name="batch", min=1, max=32)},
    "features":{0: "batch"},   # tie output batch to same symbol
}

# 5) export (no dynamic_axes when dynamo=True)
torch.onnx.export(
    model, dummy, "dinov2.onnx",
    input_names=["input"], output_names=["features"],
    opset_version=18,
    dynamo=True,
    dynamic_shapes=dyn,
)
print("Wrote dinov2.onnx")
