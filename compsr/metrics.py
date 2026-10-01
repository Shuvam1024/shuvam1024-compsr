"""Full-reference metrics on 8-bit luma.

MSE and PSNR use one dataset-wide mean, matching the notebook's Keras
``MeanSquaredError`` over every pixel of every patch. PSNR uses peak 1.0 on
the [0, 1] scale (equivalently peak 255 on code values). SSIM is Wang et al.
2004: 11-tap Gaussian, sigma 1.5, population covariance, averaged over patches.
Restored tensors are rounded and clamped to 8-bit before scoring, as in the
notebooks' ``round_postpredict``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

_SSIM_KERNEL: np.ndarray | None = None


def round_to_unit_interval(pred: torch.Tensor) -> torch.Tensor:
    """Scale a network output to 8-bit codes and back to [0, 1]."""
    codes = torch.round(pred.float() * 255.0).clamp(0.0, 255.0)
    return codes / 255.0


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred.double() - target.double()) ** 2).item())


def psnr_from_mse(mse_value: float, data_range: float = 1.0) -> float:
    if mse_value <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse_value)


def mse_reduction_percent(baseline: float, restored: float) -> float:
    if baseline <= 0.0:
        return float("nan")
    return (baseline - restored) / baseline * 100.0


def _ssim_kernel() -> np.ndarray:
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None:
        from scipy.ndimage import gaussian_filter1d

        sigma = 1.5
        truncate = 3.5
        radius = int(truncate * sigma + 0.5)
        impulse = np.zeros(2 * radius + 1, dtype=np.float64)
        impulse[radius] = 1.0
        kernel_1d = gaussian_filter1d(impulse, sigma=sigma, truncate=truncate, mode="constant", cval=0.0)
        kernel_1d = kernel_1d / kernel_1d.sum()
        _SSIM_KERNEL = np.outer(kernel_1d, kernel_1d)
    return _SSIM_KERNEL


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM of each image. ``pred`` and ``target`` are NCHW in [0, data_range]."""
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError("SSIM expects NCHW with one channel")
    kernel = torch.from_numpy(_ssim_kernel()).to(dtype=torch.float64, device=pred.device)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1])
    x = pred.double()
    y = target.double()
    mu_x = F.conv2d(x, kernel)
    mu_y = F.conv2d(y, kernel)
    mu_xx = F.conv2d(x * x, kernel)
    mu_yy = F.conv2d(y * y, kernel)
    mu_xy = F.conv2d(x * y, kernel)
    var_x = mu_xx - mu_x * mu_x
    var_y = mu_yy - mu_y * mu_y
    cov = mu_xy - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    score = numerator / denominator
    return score.mean(dim=(1, 2, 3))


def codes_to_unit(codes: torch.Tensor) -> torch.Tensor:
    return codes.double() / 255.0
