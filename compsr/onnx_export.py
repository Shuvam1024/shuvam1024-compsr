"""Export a restored-luma network to ONNX and check it against PyTorch."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from compsr.models import build_model


def export_model(name: str, path: Path, height: int = 64, width: int = 64, opset: int = 17, seed: int = 0) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = build_model(name).eval()
    dummy = torch.rand(1, 1, height, width)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["luma"],
        output_names=["restored"],
        dynamic_axes={"luma": {0: "batch", 2: "height", 3: "width"}, "restored": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
        dynamo=False,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # The legacy tracer freezes SAME padding and bilinear resize to the example
    # resolution, so the graph is valid at that HxW (any batch), not at other sizes.
    checks = []
    for batch in (1, 2):
        sample = torch.rand(batch, 1, height, width)
        with torch.no_grad():
            reference = model(sample).numpy()
        produced = session.run(["restored"], {"luma": sample.numpy()})[0]
        max_abs = float(np.max(np.abs(reference - produced)))
        checks.append({"batch": batch, "height": height, "width": width, "max_abs_diff": max_abs})
    # Timing on the export resolution, batch 8.
    timed = torch.rand(8, 1, height, width).numpy()
    for _ in range(3):
        session.run(["restored"], {"luma": timed})
    started = time.perf_counter()
    loops = 10
    for _ in range(loops):
        session.run(["restored"], {"luma": timed})
    ort_s = time.perf_counter() - started
    with torch.no_grad():
        blob = torch.from_numpy(timed)
        for _ in range(3):
            model(blob)
        started = time.perf_counter()
        for _ in range(loops):
            model(blob)
        torch_s = time.perf_counter() - started
    images = loops * timed.shape[0]
    return {
        "model": name,
        "path": str(path),
        "opset": opset,
        "checks": checks,
        "max_abs_diff": max(row["max_abs_diff"] for row in checks),
        "onnxruntime_images_per_s": images / ort_s,
        "pytorch_images_per_s": images / torch_s,
        "timing_batch": int(timed.shape[0]),
        "timing_size": height,
    }
