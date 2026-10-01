"""Faithful PyTorch ports of the five SRComp networks.

Parameter counts match the Keras summaries saved in the original notebooks
(and the counts quoted in the original README):

    A  24,953    full 3x3/5x5 residual stack
    B  20,327    spatially separable residual stack
    C  24,607    high-res + stride-2 path, dense convolutions
    D  17,367    high-res + stride-2 path, spatially separable
    E   8,277    high-res + stride-2 path, depthwise separable

Model B in the current notebook *source* is a later edit: the opening 5x5 is
written as a dense Conv2D, which would be 15,815 parameters. The executed
summary, the FLOP printout, and the README all describe the spatially
separable network (20,327 / 40,381). This module implements that executed
network: a (5, 1) convolution followed by a (1, 5) convolution, with tanh on
the second layer, matching every later separable pair in the same file.

Padding follows TensorFlow ``padding="same"`` (the extra pixel of an odd pad
goes to the bottom/right). Keras Conv2D defaults are reproduced: bias on,
Glorot uniform weights, zero bias. Keras ``SeparableConv2D`` applies bias and
the nonlinearity after the pointwise 1x1, not after the depthwise kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

LEARNING_RATES = {"A": 1e-3, "B": 1e-3, "C": 1e-3, "D": 2e-3, "E": 2e-3}

# Notebook-reported complexity. FLOPs use the keras-flops convention:
# 2 * MAC + bias add, residual add counted, activations and bilinear
# upsampling not counted. Values are per input pixel on an even map.
NOTEBOOK_COMPLEXITY = {
    "A": {"params": 24953, "flops_per_pixel": 49770},
    "B": {"params": 20327, "flops_per_pixel": 40381},
    "C": {"params": 24607, "flops_per_pixel": 23307},
    "D": {"params": 17367, "flops_per_pixel": 16455},
    "E": {"params": 8277, "flops_per_pixel": 9028},
}


def _glorot_uniform_(weight: torch.Tensor, fan_in: int, fan_out: int) -> None:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(weight, -limit, limit)


def tf_same_pad(x: torch.Tensor, kernel_h: int, kernel_w: int, stride_h: int, stride_w: int) -> torch.Tensor:
    """Pad NCHW activations the way tf.nn.conv2d padding='SAME' does."""
    height, width = x.shape[-2:]
    out_h = math.ceil(height / stride_h)
    out_w = math.ceil(width / stride_w)
    pad_h = max((out_h - 1) * stride_h + kernel_h - height, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - width, 0)
    top = pad_h // 2
    left = pad_w // 2
    bottom = pad_h - top
    right = pad_w - left
    if top == bottom == left == right == 0:
        return x
    return F.pad(x, (left, right, top, bottom))


class TFConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel, stride: int = 1, activation: str | None = None):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kh, self.kw = int(kernel[0]), int(kernel[1])
        self.stride = int(stride)
        self.activation = activation
        self.conv = nn.Conv2d(in_ch, out_ch, (self.kh, self.kw), stride=self.stride, padding=0, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_ch * self.kh * self.kw
        fan_out = self.out_ch * self.kh * self.kw
        _glorot_uniform_(self.conv.weight, fan_in, fan_out)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(tf_same_pad(x, self.kh, self.kw, self.stride, self.stride))
        if self.activation == "relu":
            return F.relu(y)
        if self.activation == "tanh":
            return torch.tanh(y)
        if self.activation is None:
            return y
        raise ValueError(f"unsupported activation {self.activation}")

    def flops(self, height: int, width: int) -> tuple[int, int, int]:
        out_h = math.ceil(height / self.stride)
        out_w = math.ceil(width / self.stride)
        macs = self.kh * self.kw * self.in_ch * self.out_ch * out_h * out_w
        bias = self.out_ch * out_h * out_w
        return 2 * macs + bias, out_h, out_w


class TFSeparableConv2d(nn.Module):
    """Keras SeparableConv2D with depth_multiplier=1."""

    def __init__(self, in_ch: int, out_ch: int, kernel, stride: int = 1, activation: str | None = "relu"):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kh, self.kw = int(kernel[0]), int(kernel[1])
        self.stride = int(stride)
        self.activation = activation
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, (self.kh, self.kw), stride=self.stride, padding=0, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keras depthwise kernel shape is (kh, kw, cin, depth_multiplier).
        _glorot_uniform_(self.depthwise.weight, self.kh * self.kw * self.in_ch, self.kh * self.kw * 1)
        _glorot_uniform_(self.pointwise.weight, self.in_ch, self.out_ch)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise(self.depthwise(tf_same_pad(x, self.kh, self.kw, self.stride, self.stride)))
        if self.activation == "relu":
            return F.relu(y)
        if self.activation == "tanh":
            return torch.tanh(y)
        if self.activation is None:
            return y
        raise ValueError(f"unsupported activation {self.activation}")

    def flops(self, height: int, width: int) -> tuple[int, int, int]:
        out_h = math.ceil(height / self.stride)
        out_w = math.ceil(width / self.stride)
        depthwise_macs = self.kh * self.kw * self.in_ch * out_h * out_w
        pointwise_macs = self.in_ch * self.out_ch * out_h * out_w
        bias = self.out_ch * out_h * out_w
        return 2 * (depthwise_macs + pointwise_macs) + bias, out_h, out_w


class BilinearUpsample2x(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def flops(self, height: int, width: int) -> tuple[int, int, int]:
        # keras-flops does not count UpSampling2D.
        return 0, height * 2, width * 2


class ResidualCNN(nn.Module):
    def __init__(self, layers: list[nn.Module], name: str):
        super().__init__()
        self.name = name
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y

    def flops_at(self, height: int, width: int) -> int:
        total = 0
        h, w = height, width
        for layer in self.layers:
            cost, h, w = layer.flops(h, w)
            total += cost
        total += height * width  # residual add, one channel
        return total


class DualPathCNN(nn.Module):
    def __init__(self, hr_layers: list[nn.Module], lr_layers: list[nn.Module], head_layers: list[nn.Module], name: str):
        super().__init__()
        self.name = name
        self.hr = nn.ModuleList(hr_layers)
        self.lr = nn.ModuleList(lr_layers)
        self.up = BilinearUpsample2x()
        self.head = nn.ModuleList(head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = x
        for layer in self.hr:
            high = layer(high)
        low = x
        for layer in self.lr:
            low = layer(low)
        low = self.up(low)
        if low.shape[-2:] != high.shape[-2:]:
            # Odd inputs: TF SAME + 2x upsample can be one pixel off.
            low = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([high, low], dim=1)
        for layer in self.head:
            y = layer(y)
        return x + y

    def flops_at(self, height: int, width: int) -> int:
        total = 0
        h, w = height, width
        for layer in self.hr:
            cost, h, w = layer.flops(h, w)
            total += cost
        lh, lw = height, width
        for layer in self.lr:
            cost, lh, lw = layer.flops(lh, lw)
            total += cost
        cost, lh, lw = self.up.flops(lh, lw)
        total += cost
        if (lh, lw) != (h, w):
            lh, lw = h, w
        for layer in self.head:
            cost, lh, lw = layer.flops(lh, lw)
            total += cost
        total += height * width
        return total


def model_a() -> ResidualCNN:
    return ResidualCNN(
        [
            TFConv2d(1, 32, 5, activation="tanh"),
            TFConv2d(32, 28, 3, activation="relu"),
            TFConv2d(28, 24, 3, activation="relu"),
            TFConv2d(24, 20, 3, activation="relu"),
            TFConv2d(20, 16, 3, activation="relu"),
            TFConv2d(16, 16, 3, activation="relu"),
            TFConv2d(16, 1, 5, activation="tanh"),
        ],
        name="A",
    )


def model_b() -> ResidualCNN:
    return ResidualCNN(
        [
            TFConv2d(1, 32, (5, 1), activation=None),
            TFConv2d(32, 32, (1, 5), activation="tanh"),
            TFConv2d(32, 28, (3, 1), activation=None),
            TFConv2d(28, 28, (1, 3), activation="relu"),
            TFConv2d(28, 24, (3, 1), activation=None),
            TFConv2d(24, 24, (1, 3), activation="relu"),
            TFConv2d(24, 20, (3, 1), activation=None),
            TFConv2d(20, 20, (1, 3), activation="relu"),
            TFConv2d(20, 16, (3, 1), activation=None),
            TFConv2d(16, 16, (1, 3), activation="relu"),
            TFConv2d(16, 16, (3, 1), activation=None),
            TFConv2d(16, 16, (1, 3), activation="relu"),
            TFConv2d(16, 1, (5, 1), activation=None),
            TFConv2d(1, 1, (1, 5), activation="tanh"),
        ],
        name="B",
    )


def model_c() -> DualPathCNN:
    return DualPathCNN(
        hr_layers=[
            TFConv2d(1, 20, 5, activation="tanh"),
            TFConv2d(20, 18, 3, activation="relu"),
            TFConv2d(18, 16, 3, activation="relu"),
        ],
        lr_layers=[
            TFConv2d(1, 40, 5, stride=2, activation="tanh"),
            TFConv2d(40, 28, 3, activation="relu"),
            TFConv2d(28, 24, 3, activation="relu"),
        ],
        head_layers=[TFConv2d(40, 1, 5, activation="tanh")],
        name="C",
    )


def model_d() -> DualPathCNN:
    return DualPathCNN(
        hr_layers=[
            TFConv2d(1, 24, 5, activation="tanh"),
            TFConv2d(24, 20, (3, 1), activation=None),
            TFConv2d(20, 20, (1, 3), activation="relu"),
            TFConv2d(20, 16, (3, 1), activation=None),
            TFConv2d(16, 16, (1, 3), activation="relu"),
        ],
        lr_layers=[
            TFConv2d(1, 40, 5, stride=2, activation="tanh"),
            TFConv2d(40, 32, (3, 1), activation=None),
            TFConv2d(32, 32, (1, 3), activation="relu"),
            TFConv2d(32, 24, (3, 1), activation=None),
            TFConv2d(24, 24, (1, 3), activation="relu"),
        ],
        head_layers=[
            TFConv2d(40, 1, (5, 1), activation=None),
            TFConv2d(1, 1, (1, 5), activation="tanh"),
        ],
        name="D",
    )


def model_e() -> DualPathCNN:
    return DualPathCNN(
        hr_layers=[
            TFConv2d(1, 24, 5, activation="tanh"),
            TFSeparableConv2d(24, 24, 5, activation="relu"),
            TFSeparableConv2d(24, 20, 5, activation="relu"),
        ],
        lr_layers=[
            TFConv2d(1, 36, 5, stride=2, activation="tanh"),
            TFSeparableConv2d(36, 36, 5, activation="relu"),
            TFSeparableConv2d(36, 24, 5, activation="relu"),
        ],
        head_layers=[TFConv2d(44, 1, 3, activation="tanh")],
        name="E",
    )


_BUILDERS = {"A": model_a, "B": model_b, "C": model_c, "D": model_d, "E": model_e}


def build_model(name: str) -> nn.Module:
    key = name.upper()
    if key not in _BUILDERS:
        raise KeyError(f"unknown model {name!r}; expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[key]()


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_per_pixel(model: nn.Module, size: int = 128) -> float:
    if size % 2:
        raise ValueError("flops_per_pixel reference size must be even")
    return model.flops_at(size, size) / (size * size)
