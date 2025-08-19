# export_dinov2_onnx.py
import torch

# Grab the pretrained ViT-B/14 from the official repo
model = torch.hub.load(
    'facebookresearch/dinov2',          # repo
    'dinov2_vitb14',                    # entrypoint
    pretrained=True,
    source='github'
).eval().cuda()

class Wrapper(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        feats = self.m.forward_features(x)      # [N, D]
        return torch.nn.functional.normalize(feats, dim=-1)

dummy = torch.randn(1, 3, 224, 224, device="cuda")
wrapped = Wrapper(model).eval().cuda()

torch.onnx.export(
    wrapped, dummy, "dinov2.onnx",
    input_names=["input"], output_names=["features"],
    opset_version=17, do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}}
)
print("Wrote dinov2.onnx")
