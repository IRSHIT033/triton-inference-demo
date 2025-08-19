# export_dinov2_onnx.py (clean, no TracerWarnings)
import torch

# 1) Load model from hub
m = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True, source='github'
).eval().cuda()

# 2) Wrap to return a single [N, D] tensor (CLS embedding)
class Wrapper(torch.nn.Module):
    def __init__(self, base): 
        super().__init__(); self.base = base
    def forward(self, x):
        out = self.base.forward_features(x)  # dict for DINOv2
        # Prefer normalized CLS token if present
        for k in ("x_norm_clstoken", "cls_token", "x_cls", "x"):
            if isinstance(out, dict) and k in out:
                x = out[k]
                break
        if isinstance(out, dict) and not isinstance(x, torch.Tensor):
            # Fallback: first tensor value in dict
            x = next(v for v in out.values() if isinstance(v, torch.Tensor))
        return torch.nn.functional.normalize(x, dim=-1)

model = Wrapper(m).eval().cuda()

# 3) Dummy (fixed 224×224)
dummy = torch.randn(1, 3, 224, 224, device="cuda")

# 4) Export with the new Dynamo exporter
torch.onnx.export(
    model,
    dummy,
    "dinov2.onnx",
    input_names=["input"],
    output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=18,     # 18+ is good with dynamo
    dynamo=True,          # <- key: use new exporter
)
print("Wrote dinov2.onnx")
