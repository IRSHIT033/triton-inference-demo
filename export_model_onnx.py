# export_dinov2_onnx.py
import torch
from torchvision import transforms
# pip install dinov2 torch torchvision  (or install from facebookresearch/dinov2)
from dinov2.models.vision_transformer import dinov2_vitb14

model = dinov2_vitb14(pretrained=True)
model.eval().cuda()

dummy = torch.randn(1, 3, 224, 224, device="cuda")  # NCHW, already normalized later in client/preproc

# If you want features before classifier, make sure your forward returns them.
# The stock DINOv2 ViTs expose features via .forward_features(x)
class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        feats = self.m.forward_features(x)  # [N, D]
        # Return L2-normalized global features (common for DINOv2)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

wrapped = Wrapper(model).eval().cuda()

torch.onnx.export(
    wrapped,
    dummy,
    "dinov2.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},  # dynamic batch
)
print("Wrote dinov2.onnx")
