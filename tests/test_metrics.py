import numpy as np
import torch
from skimage.metrics import structural_similarity

from compsr.metrics import (
    mse,
    mse_reduction_percent,
    psnr_from_mse,
    round_to_unit_interval,
    ssim_per_image,
)


def test_mse_psnr_and_reduction_match_closed_form():
    pred = torch.tensor([0.0, 0.5, 1.0])
    target = torch.tensor([0.0, 0.0, 1.0])
    value = mse(pred, target)
    assert value == np.mean((pred.numpy() - target.numpy()) ** 2)
    assert psnr_from_mse(value) == 10 * np.log10(1.0 / value)
    assert abs(mse_reduction_percent(0.2, 0.15) - 25.0) < 1e-9


def test_round_to_unit_interval_clamps():
    raw = torch.tensor([-0.2, 0.5, 1.4])
    rounded = round_to_unit_interval(raw)
    assert torch.allclose(rounded, torch.tensor([0.0, 128 / 255, 1.0]))


def test_ssim_matches_skimage_wang():
    rng = np.random.default_rng(0)
    clean = rng.random((40, 40), dtype=np.float64)
    noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
    reference = structural_similarity(
        clean,
        noisy,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    pred = torch.from_numpy(noisy).view(1, 1, 40, 40)
    target = torch.from_numpy(clean).view(1, 1, 40, 40)
    got = float(ssim_per_image(pred, target).item())
    assert abs(got - reference) < 1e-6


def test_identical_images_have_unit_ssim():
    image = torch.rand(2, 1, 32, 32)
    scores = ssim_per_image(image, image)
    assert torch.allclose(scores, torch.ones(2, dtype=scores.dtype), atol=1e-6)
