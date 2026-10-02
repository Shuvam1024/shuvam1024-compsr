"""Rebuild patch archives from full-resolution RGB images.

The original dataset script was never published. This follows the README
procedure and pins the choices the README leaves open:

* RGB is converted to limited-range YUV 4:2:0 (the released reference luma
  sits in 16..235). The matrix defaults to bt709 and is configurable.
* Both the full-resolution frame and the downscaled frame are cropped to even
  sizes so 4:2:0 is legal. The ratio's integer scale is exact.
* Downscale and upscale use Lanczos with ``param0=5`` (the ``a=5`` window).
* The downscaled frame is encoded with libaom-av1 at constant quality ``Q``
  (``ffmpeg -crf``), which is the 0-63 AV1 quantizer the README calls QP.
* A patch is kept when ``max(Y) - min(Y)`` on the reference patch is at least 8.
* Archives store ``data``, ``noisy_data`` (uint8, shape N,P,P) and ``patchsize``.

The 640/160 split of DIV2K images 1-800 is re-drawn from ``--split-seed``.
That seed is not the unpublished original split. Bit-exact reproduction of
the hosted ``.npz`` files is not claimed: libaom version, ``cpu-used``, color
matrix, crop origin, and patch sampler were not recorded.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

RATIO_FACTORS = {"2by1": (2, 1), "8by5": (8, 5), "4by3": (4, 3)}


def aligned_size(width: int, height: int, ratio: str) -> tuple[int, int, int, int]:
    """Return ``(crop_w, crop_h, down_w, down_h)`` with exact ratio and even sides."""
    num, den = RATIO_FACTORS[ratio]

    def shrink(dim: int) -> int:
        while dim >= num:
            if dim % 2 == 0 and dim % num == 0:
                down = dim * den // num
                if down % 2 == 0 and down >= 2:
                    return dim
            dim -= 1
        raise ValueError(f"image dimension too small to align for ratio {ratio}")

    crop_w = shrink(width)
    crop_h = shrink(height)
    return crop_w, crop_h, crop_w * den // num, crop_h * den // num


def sample_patches(
    clean: np.ndarray,
    noisy: np.ndarray,
    patch: int,
    count: int,
    rng: np.random.RandomState,
    min_range: int = 8,
    max_attempts_factor: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    if clean.shape != noisy.shape or clean.ndim != 2:
        raise ValueError("clean and noisy must be matching 2-D luma planes")
    height, width = clean.shape
    if patch > height or patch > width:
        raise ValueError(f"patch {patch} does not fit in {width}x{height}")
    kept_clean = []
    kept_noisy = []
    attempts = 0
    limit = max(count * max_attempts_factor, count)
    while len(kept_clean) < count and attempts < limit:
        attempts += 1
        y = int(rng.randint(0, height - patch + 1))
        x = int(rng.randint(0, width - patch + 1))
        reference = clean[y : y + patch, x : x + patch]
        if int(reference.max()) - int(reference.min()) < min_range:
            continue
        kept_clean.append(reference.copy())
        kept_noisy.append(noisy[y : y + patch, x : x + patch].copy())
    if not kept_clean:
        return (
            np.zeros((0, patch, patch), dtype=np.uint8),
            np.zeros((0, patch, patch), dtype=np.uint8),
        )
    return np.stack(kept_clean), np.stack(kept_noisy)


def save_archive(path: Path, clean: np.ndarray, noisy: np.ndarray, patch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        data=np.ascontiguousarray(clean, dtype=np.uint8),
        noisy_data=np.ascontiguousarray(noisy, dtype=np.uint8),
        patchsize=np.int64(patch),
    )


def div2k_id(path: Path) -> int | None:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) < 4:
        return None
    return int(digits[-4:])


def split_ids(ids: list[int], split: str, seed: int) -> set[int]:
    """80/20 redraw of ids in 1..800; ids in 801..900 are the test pool."""
    train_pool = sorted(i for i in ids if 1 <= i <= 800)
    test_pool = sorted(i for i in ids if 801 <= i <= 900)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(train_pool) if train_pool else np.array([], dtype=int)
    n_train = round(len(perm) * 0.8)
    if split == "train":
        return {int(i) for i in perm[:n_train]}
    if split == "valid":
        return {int(i) for i in perm[n_train:]}
    if split == "test":
        return set(test_pool)
    raise ValueError(f"unknown split {split}")


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")


def read_y_plane(path: Path, width: int, height: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    need = width * height * 3 // 2
    if raw.size < need:
        raise RuntimeError(f"{path} has {raw.size} bytes, expected at least {need} for yuv420p {width}x{height}")
    return raw[: width * height].reshape(height, width)


def ffmpeg_commands(
    png: Path,
    work: Path,
    crop_w: int,
    crop_h: int,
    down_w: int,
    down_h: int,
    qp: int,
    cpu_used: int,
    matrix: str,
) -> list[list[str]]:
    """Return the ffmpeg invocations for one image. ``work`` holds intermediates."""
    common_range = f"in_range=full:out_range=limited:in_color_matrix={matrix}:out_color_matrix={matrix}"
    orig = work / "orig.yuv"
    down = work / "down.yuv"
    coded = work / "coded.ivf"
    decoded = work / "decoded.yuv"
    up = work / "up.yuv"
    return [
        [
            "ffmpeg", "-y", "-i", str(png),
            "-vf", f"crop={crop_w}:{crop_h}:0:0,scale={crop_w}:{crop_h}:flags=lanczos:param0=5:{common_range},format=yuv420p",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", str(orig),
        ],
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{crop_w}x{crop_h}", "-i", str(orig),
            "-vf", f"scale={down_w}:{down_h}:flags=lanczos:param0=5:in_range=limited:out_range=limited",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", str(down),
        ],
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{down_w}x{down_h}", "-r", "1",
            "-i", str(down), "-frames:v", "1",
            "-c:v", "libaom-av1", "-crf", str(qp), "-b:v", "0", "-cpu-used", str(cpu_used),
            "-still-picture", "1", "-row-mt", "1", str(coded),
        ],
        [
            "ffmpeg", "-y", "-i", str(coded), "-frames:v", "1",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", str(decoded),
        ],
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{down_w}x{down_h}", "-i", str(decoded),
            "-vf", f"scale={crop_w}:{crop_h}:flags=lanczos:param0=5:in_range=limited:out_range=limited",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", str(up),
        ],
    ]


def process_image(
    png: Path,
    ratio: str,
    qp: int,
    patch: int,
    patches_per_image: int,
    rng: np.random.RandomState,
    cpu_used: int = 6,
    matrix: str = "bt709",
    min_range: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH")
    with Image.open(png) as image:
        width, height = image.size
    crop_w, crop_h, down_w, down_h = aligned_size(width, height, ratio)
    with tempfile.TemporaryDirectory(prefix="compsr-") as tmp:
        work = Path(tmp)
        for cmd in ffmpeg_commands(png, work, crop_w, crop_h, down_w, down_h, qp, cpu_used, matrix):
            _run(cmd)
        clean = read_y_plane(work / "orig.yuv", crop_w, crop_h)
        noisy = read_y_plane(work / "up.yuv", crop_w, crop_h)
    return sample_patches(clean, noisy, patch, patches_per_image, rng, min_range=min_range)


def generate_archive(
    images: list[Path],
    output: Path,
    ratio: str,
    qp: int,
    patch: int,
    patches_per_image: int,
    seed: int,
    cpu_used: int = 6,
    matrix: str = "bt709",
    min_range: int = 8,
) -> dict:
    rng = np.random.RandomState(seed)
    clean_parts = []
    noisy_parts = []
    for image_path in images:
        clean, noisy = process_image(
            image_path, ratio, qp, patch, patches_per_image, rng, cpu_used=cpu_used, matrix=matrix, min_range=min_range
        )
        if len(clean):
            clean_parts.append(clean)
            noisy_parts.append(noisy)
        print(f"{image_path.name}: kept {len(clean)} patches", flush=True)
    if not clean_parts:
        raise RuntimeError("no patches passed the contrast check")
    clean_all = np.concatenate(clean_parts, axis=0)
    noisy_all = np.concatenate(noisy_parts, axis=0)
    save_archive(output, clean_all, noisy_all, patch)
    return {"path": str(output), "n_patches": len(clean_all), "patch": patch, "ratio": ratio, "qp": qp}
