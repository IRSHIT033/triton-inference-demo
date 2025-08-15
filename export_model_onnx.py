import torch
from torch.onnx import dynamo_export, ExportOptions, DynamicDim
from transformers import AutoModel

class DinoCLS(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()

    def forward(self, pixel_values: torch.Tensor):
        out = self.base(pixel_values=pixel_values)
        return out.last_hidden_state[:, 0, :]  # [B, H]

model_id = "facebook/dinov2-base"
base = AutoModel.from_pretrained(model_id)
wrapped = DinoCLS(base).eval()

dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

dynamic_shapes = {
    "pixel_values": {
        0: DynamicDim("batch"),
        2: DynamicDim("height"),
        3: DynamicDim("width"),
    }
}

exported = dynamo_export(
    wrapped,
    dummy,
    export_options=ExportOptions(
        opset_version=17,             # 18 also fine if your stack supports it
        dynamic_shapes=dynamic_shapes # <-- correct way with Dynamo
    ),
)

exported.save("dinov2_base_cls.onnx")

