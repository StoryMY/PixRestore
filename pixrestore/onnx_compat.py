"""Export-only equivalents for ONNX Runtime WebGPU compatibility."""
import torch
from torch import nn


class PatchConvMatMul(nn.Module):
    """Non-overlapping Conv2d expressed as patch packing and MatMul.

    The inner patch order is C, kernel row, kernel column, matching flattened
    Conv2d weights. The output remains NCHW for the following 1x1 convolution.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if (conv.groups != 1 or conv.padding != (0, 0)
                or conv.dilation != (1, 1) or conv.stride != conv.kernel_size
                or conv.padding_mode != 'zeros'):
            raise ValueError('Expected ungrouped, non-overlapping Conv2d without padding')
        self.kernel_size = conv.kernel_size
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.weight = nn.Parameter(conv.weight.detach().clone(), requires_grad=False)
        self.bias = (nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
                     if conv.bias is not None else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        kh, kw = self.kernel_size
        if not torch.jit.is_tracing() and (h % kh or w % kw):
            raise ValueError('Image dimensions must be multiples of the patch size')
        gh, gw = h // kh, w // kw
        patches = x.reshape(b, c, gh, kh, gw, kw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * kh * kw)
        result = torch.matmul(patches, self.weight.reshape(self.out_channels, -1).t())
        if self.bias is not None:
            result = result + self.bias
        return result.transpose(1, 2).reshape(b, self.out_channels, gh, gw)


def apply_webgpu_compat(model: nn.Module) -> None:
    """Call after loading weights, on the disposable model used for export."""
    # 本地验证发现 WebGPU 的 8x8、stride-8 卷积存在明显偏差，导致输出模糊。
    # 仅在导出时用等价的 patch 展开 + MatMul 绕过该路径，不改变训练和权重。
    conv = model.x_embedder.proj1
    if isinstance(conv, PatchConvMatMul):
        return
    if not isinstance(conv, nn.Conv2d):
        raise TypeError('Expected x_embedder.proj1 to be Conv2d')
    model.x_embedder.proj1 = PatchConvMatMul(conv)
