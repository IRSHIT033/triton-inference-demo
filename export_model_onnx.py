# export_dinov2_onnx.py
import torch
from transformers import AutoModel

# Wrap to expose CLS embedding only
class DinoCLS(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()

    def forward(self, pixel_values):
        out = self.base(pixel_values=pixel_values)
        # out.last_hidden_state: [B, seq_len, H]; take CLS at position 0
        return out.last_hidden_state[:, 0, :]  # [B, H]

model_id = "facebook/dinov2-base"  # S/B/L/g -> 384/768/1024/1536 dims
base = AutoModel.from_pretrained(model_id)
wrapped = DinoCLS(base).eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    wrapped,
    (dummy,),
    "dinov2_base_cls.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeds"],
    dynamic_axes={
        "pixel_values": {0: "batch", 2: "height", 3: "width"},
        "image_embeds": {0: "batch"}
    },
    opset_version=17,  # or 18
    do_constant_folding=True,
    dynamo=True        # use TorchDynamo-based exporter
)
