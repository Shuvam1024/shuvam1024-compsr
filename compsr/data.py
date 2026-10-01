"""Hosted patch archives: URLs, checksums, loading, and the 4:3 release audit."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

RATIOS = ("2by1", "8by5", "4by3")
QPS = (20, 30, 40, 50)
SPLITS = ("train", "valid", "test")

# Patch geometry promised by the filenames. The 4:3 archives do not honor it.
SPLIT_PATCH = {"train": 48, "valid": 48, "test": 64}


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def checksum_manifest_path() -> Path:
    return package_dir() / "checksums.json"


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or checksum_manifest_path()
    return json.loads(manifest_path.read_text())


def iter_entries(manifest: dict | None = None):
    data = manifest if manifest is not None else load_manifest()
    yield from data["files"]


def find_entry(ratio: str, qp: int, split: str, manifest: dict | None = None) -> dict:
    for entry in iter_entries(manifest):
        if entry["ratio"] == ratio and int(entry["qp"]) == qp and entry["split"] == split:
            return entry
    raise KeyError(f"no manifest entry for {split} {ratio} QP {qp}")


def archive_path(data_dir: Path, entry: dict) -> Path:
    return Path(data_dir) / f"Ratio_{entry['ratio']}" / entry["filename"]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_entry(entry: dict, data_dir: Path, force: bool = False) -> Path:
    dest = archive_path(data_dir, entry)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        actual = sha256_file(dest)
        if actual != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for existing {dest}: {actual} != {entry['sha256']}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".partial")
    digest = hashlib.sha256()
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "compsr-dataset-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            digest.update(block)
    actual = digest.hexdigest()
    if actual != entry["sha256"]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {entry['filename']}: {actual} != {entry['sha256']}")
    tmp.replace(dest)
    return dest


def download_selection(
    data_dir: Path,
    ratio: str | None = None,
    qp: int | None = None,
    split: str | None = None,
    force: bool = False,
    usable_only: bool = False,
) -> list[Path]:
    saved = []
    for entry in iter_entries():
        if ratio is not None and entry["ratio"] != ratio:
            continue
        if qp is not None and int(entry["qp"]) != qp:
            continue
        if split is not None and entry["split"] != split:
            continue
        if usable_only and not entry["usable"]:
            continue
        print(f"fetch {entry['filename']}", flush=True)
        saved.append(download_entry(entry, data_dir, force=force))
    return saved


def _patchsize_of(value) -> int:
    return int(np.array(value).reshape(-1)[0])


def load_archive(path: Path) -> dict:
    with np.load(path) as archive:
        clean = np.ascontiguousarray(archive["data"])
        noisy = np.ascontiguousarray(archive["noisy_data"])
        patchsize = _patchsize_of(archive["patchsize"])
    if clean.shape != noisy.shape or clean.ndim != 3:
        raise ValueError(f"{path} expected matching (N, H, W) arrays, got {clean.shape} {noisy.shape}")
    if clean.shape[1] != clean.shape[2]:
        raise ValueError(f"{path} patches are not square: {clean.shape}")
    return {"clean": clean, "noisy": noisy, "patchsize": patchsize, "path": str(path)}


def load_split(data_dir: Path, ratio: str, qp: int, split: str, require_usable: bool = True) -> dict:
    entry = find_entry(ratio, qp, split)
    if require_usable and not entry["usable"]:
        raise RuntimeError(
            f"{entry['filename']} is marked unusable in the checksum manifest. {entry.get('note', '')}"
        )
    path = archive_path(data_dir, entry)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}. Run: compsr download --ratio {ratio} --qp {qp} --split {split}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"checksum mismatch for {path}: {actual} != {entry['sha256']}")
    loaded = load_archive(path)
    loaded["entry"] = entry
    spatial = int(loaded["clean"].shape[-1])
    if loaded["patchsize"] != spatial:
        loaded["patchsize_mismatch"] = True
    else:
        loaded["patchsize_mismatch"] = False
    return loaded


def subset_pair(clean: np.ndarray, noisy: np.ndarray, max_patches: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_patches is None or max_patches >= len(clean):
        return clean, noisy
    if max_patches < 1:
        raise ValueError("max_patches must be positive")
    rng = np.random.RandomState(seed)
    index = np.sort(rng.choice(len(clean), size=int(max_patches), replace=False))
    return clean[index], noisy[index]


class PatchDataset(Dataset):
    def __init__(self, noisy: np.ndarray, clean: np.ndarray):
        if len(noisy) != len(clean):
            raise ValueError("noisy/clean length mismatch")
        self.noisy = noisy
        self.clean = clean

    def __len__(self) -> int:
        return len(self.clean)

    def __getitem__(self, index: int):
        return self.noisy[index], self.clean[index]


def collate_uint8(batch):
    noisy = torch.from_numpy(np.stack([item[0] for item in batch])).unsqueeze(1)
    clean = torch.from_numpy(np.stack([item[1] for item in batch])).unsqueeze(1)
    return noisy.float().div_(255.0), clean.float().div_(255.0)


def identical_groups(manifest: dict | None = None) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for entry in iter_entries(manifest):
        groups.setdefault(entry["sha256"], []).append(entry["filename"])
    return [names for names in groups.values() if len(names) > 1]
