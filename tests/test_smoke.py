import numpy as np
import torch

from compsr.models import build_model
from compsr.train import fit_arrays, set_seed


def _synthetic(n=16, size=16, seed=0):
    rng = np.random.default_rng(seed)
    clean = rng.integers(16, 236, size=(n, size, size), dtype=np.uint8)
    noisy = rng.integers(0, 256, size=(n, size, size), dtype=np.uint8)
    return noisy, clean


def test_smoke_train_is_finite_and_deterministic():
    noisy, clean = _synthetic()
    losses = []
    for _ in range(2):
        set_seed(4)
        model = build_model("E")
        history = fit_arrays(
            model,
            noisy,
            clean,
            epochs=1,
            batch_size=8,
            lr=2e-3,
            seed=4,
            device=torch.device("cpu"),
        )
        assert np.isfinite(history[0]["train_mse"])
        losses.append(history[0]["train_mse"])
        with torch.no_grad():
            out = model(torch.rand(2, 1, 16, 16))
        assert out.shape == (2, 1, 16, 16)
    assert losses[0] == losses[1]
