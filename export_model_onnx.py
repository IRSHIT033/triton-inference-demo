import torch
from torch.export import Dim
from transformers import AutoModel

# Wrap to expose CLS embedding only
class DinoCLS(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()
    def forward(self, pixel_values):
        out = self.base(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]  # [B, H]

model_id = "facebook/dinov2-base"
wrapped = DinoCLS(AutoModel.from_pretrained(model_id)).eval()

dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

# Only describe inputs here (no outputs!)
dynamic_shapes = {
    "pixel_values": {
        0: Dim("batch",  min=1,  max=32),
        2: Dim("height", min=224, max=1024),
        3: Dim("width",  min=224, max=1024),
    }
}

torch.onnx.export(
    wrapped,
    (dummy,),
    "dinov2_base_cls.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    opset_version=18,          # 18+ recommended for shape ops
    do_constant_folding=True,
    dynamo=True,               # using the new exporter
    dynamic_shapes=dynamic_shapes
)
