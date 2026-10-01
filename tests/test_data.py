import numpy as np
import pytest

from compsr.data import (
    find_entry,
    identical_groups,
    load_manifest,
    load_split,
    subset_pair,
)


def test_manifest_covers_twelve_settings_and_marks_4by3_unusable():
    manifest = load_manifest()
    assert len(manifest["files"]) == 36
    usable = [entry for entry in manifest["files"] if entry["usable"]]
    assert len(usable) == 24
    for entry in manifest["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["ratio"] in {"2by1", "8by5", "4by3"}
        assert int(entry["qp"]) in {20, 30, 40, 50}
    groups = identical_groups(manifest)
    assert any(len(group) >= 8 for group in groups)
    four = find_entry("4by3", 20, "train", manifest)
    assert four["usable"] is False
    with pytest.raises(RuntimeError, match="unusable"):
        load_split("data", "4by3", 20, "train")


def test_subset_is_deterministic_and_aligned():
    clean = np.arange(10, dtype=np.uint8).reshape(10, 1, 1) + np.zeros((10, 4, 4), dtype=np.uint8)
    noisy = clean + 1
    a_clean, a_noisy = subset_pair(clean, noisy, 4, seed=3)
    b_clean, b_noisy = subset_pair(clean, noisy, 4, seed=3)
    assert a_clean.shape == (4, 4, 4)
    assert np.array_equal(a_clean, b_clean)
    assert np.array_equal(a_noisy, b_noisy + 0)
    assert np.array_equal(a_noisy, a_clean + 1)
