# export_dinov2_onnx.py
import torch

# 1) Load DINOv2 ViT-B/14 from GitHub via torch.hub
model = torch.hub.load(
    'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True, source='github'
).eval().cuda()

# 2) Wrap to return a single [N, D] tensor (CLS embedding)
class Wrapper(torch.nn.Module):
    def __init__(self, m): 
        super().__init__(); self.m = m
    def forward(self, x):
        feats = self.m.forward_features(x)
        if isinstance(feats, dict):
            # try the common keys first
            for k in ("x_norm_clstoken", "cls_token", "x_cls", "x"):
                if k in feats:
                    feats = feats[k]
                    break
            else:
                # last resort: take the first tensor value in the dict
                for v in feats.values():
                    if isinstance(v, torch.Tensor):
                        feats = v
                        break
                if isinstance(feats, dict):
                    raise RuntimeError(f"Unexpected forward_features() keys: {list(feats.keys())}")
        # L2-normalize feature vectors
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats

wrapped = Wrapper(model).eval().cuda()

# 3) Dummy input (NCHW, 224x224)
dummy = torch.randn(1, 3, 224, 224, device="cuda")

# 4) Export ONNX (dynamic batch)
torch.onnx.export(
    wrapped,
    dummy,
    "dinov2.onnx",
    input_names=["input"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
)
print("Wrote dinov2.onnx")
