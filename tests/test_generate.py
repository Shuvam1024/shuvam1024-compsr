from pathlib import Path

import numpy as np

from compsr.generate import aligned_size, ffmpeg_commands, sample_patches, split_ids


def test_aligned_size_is_exact_and_even():
    assert aligned_size(1920, 1080, "2by1") == (1920, 1080, 960, 540)
    assert aligned_size(100, 50, "8by5") == (96, 48, 60, 30)
    assert aligned_size(90, 40, "4by3") == (88, 40, 66, 30)
    for ratio in ("2by1", "8by5", "4by3"):
        crop_w, crop_h, down_w, down_h = aligned_size(333, 257, ratio)
        num, den = {"2by1": (2, 1), "8by5": (8, 5), "4by3": (4, 3)}[ratio]
        assert crop_w % 2 == 0 and crop_h % 2 == 0
        assert down_w % 2 == 0 and down_h % 2 == 0
        assert crop_w * den == down_w * num
        assert crop_h * den == down_h * num


def test_patch_sampler_rejects_flat_patches():
    clean = np.zeros((32, 32), dtype=np.uint8)
    clean[8:24, 8:24] = np.arange(16 * 16, dtype=np.uint8).reshape(16, 16) + 20
    noisy = clean.copy()
    rng = np.random.RandomState(0)
    kept, noisy_kept = sample_patches(clean, noisy, patch=8, count=5, rng=rng, min_range=8)
    assert kept.shape == (5, 8, 8)
    assert noisy_kept.shape == kept.shape
    for patch in kept:
        assert int(patch.max()) - int(patch.min()) >= 8


def test_split_ids_are_disjoint_and_cover_train_pool():
    ids = list(range(1, 11)) + [801, 802]
    train = split_ids(ids, "train", seed=1)
    valid = split_ids(ids, "valid", seed=1)
    test = split_ids(ids, "test", seed=1)
    assert train.isdisjoint(valid)
    assert train | valid == set(range(1, 11))
    assert test == {801, 802}
    assert len(train) == 8


def test_ffmpeg_command_uses_libaom_lanczos_and_crf():
    commands = ffmpeg_commands(Path("in.png"), Path("work"), 64, 64, 32, 32, qp=30, cpu_used=6, matrix="bt709")
    joined = " ".join(" ".join(cmd) for cmd in commands)
    assert "libaom-av1" in joined
    assert "-crf 30" in joined
    assert "lanczos" in joined
    assert "param0=5" in joined
    assert "yuv420p" in joined
    assert "out_range=limited" in joined
