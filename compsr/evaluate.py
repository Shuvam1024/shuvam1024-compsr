"""Test-set MSE, PSNR, and SSIM against the Lanczos-upscaled baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from compsr.data import PatchDataset, collate_uint8
from compsr.metrics import (
    codes_to_unit,
    mse_reduction_percent,
    psnr_from_mse,
    round_to_unit_interval,
    ssim_per_image,
)


@torch.no_grad()
def evaluate_arrays(
    model: nn.Module,
    noisy_u8,
    clean_u8,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)
    loader = DataLoader(
        PatchDataset(noisy_u8, clean_u8),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uint8,
    )
    sse_base = 0.0
    sse_rest = 0.0
    ssim_base = 0.0
    ssim_rest = 0.0
    n_pix = 0
    n_img = 0
    for noisy, clean in loader:
        # collate already scaled uint8/255 in float32. Rebuild the exact codes
        # from the original batch via rounding so baseline pixels stay on the
        # 8-bit grid even if the division was not dyadic in float32.
        noisy = noisy.to(device)
        clean = clean.to(device)
        pred = model(noisy)
        restored = round_to_unit_interval(pred).double()
        clean_u = codes_to_unit(torch.round(clean.float() * 255.0).clamp(0, 255))
        noisy_u = codes_to_unit(torch.round(noisy.float() * 255.0).clamp(0, 255))
        sse_base += float(torch.sum((noisy_u - clean_u) ** 2).item())
        sse_rest += float(torch.sum((restored - clean_u) ** 2).item())
        ssim_base += float(ssim_per_image(noisy_u.float(), clean_u.float()).sum().item())
        ssim_rest += float(ssim_per_image(restored.float(), clean_u.float()).sum().item())
        n_pix += int(clean_u.numel())
        n_img += int(clean_u.shape[0])
    mse_base = sse_base / n_pix
    mse_rest = sse_rest / n_pix
    psnr_base = psnr_from_mse(mse_base)
    psnr_rest = psnr_from_mse(mse_rest)
    ssim_base_mean = ssim_base / n_img
    ssim_rest_mean = ssim_rest / n_img
    return {
        "mse_baseline": mse_base,
        "mse_restored": mse_rest,
        "mse_reduction_pct": mse_reduction_percent(mse_base, mse_rest),
        "psnr_baseline": psnr_base,
        "psnr_restored": psnr_rest,
        "psnr_gain_db": psnr_rest - psnr_base,
        "ssim_baseline": ssim_base_mean,
        "ssim_restored": ssim_rest_mean,
        "ssim_gain": ssim_rest_mean - ssim_base_mean,
        "n_images": n_img,
        "n_pixels": n_pix,
    }
