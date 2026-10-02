"""CPU inference throughput. GPU is used when ``device`` is cuda and present."""

from __future__ import annotations

import time

import torch
from torch import nn


def measure_throughput(
    model: nn.Module,
    size: int = 64,
    batch: int = 16,
    iters: int = 20,
    warmup: int = 5,
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    sample = torch.rand(batch, 1, size, size, device=dev)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            model(sample)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    images = iters * batch
    pixels = images * size * size
    return {
        "device": device,
        "batch": batch,
        "height": size,
        "width": size,
        "iters": iters,
        "images_per_s": images / elapsed,
        "mpix_per_s": pixels / elapsed / 1e6,
        "ms_per_image": elapsed / images * 1000.0,
        "elapsed_s": elapsed,
    }


def measure_frame_latency(model: nn.Module, height: int = 720, width: int = 1280, iters: int = 10, device: str = "cpu") -> dict:
    dev = torch.device(device)
    model = model.to(dev).eval()
    frame = torch.rand(1, 1, height, width, device=dev)
    with torch.no_grad():
        for _ in range(2):
            model(frame)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            model(frame)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    return {
        "device": device,
        "height": height,
        "width": width,
        "ms_per_frame": elapsed / iters * 1000.0,
        "frames_per_s": iters / elapsed,
    }
