"""Adam / MSE training with a fixed seed and an equal sample budget."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from compsr.data import PatchDataset, collate_uint8, load_split, subset_pair
from compsr.evaluate import evaluate_arrays
from compsr.models import LEARNING_RATES, build_model, flops_per_pixel, parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_name(model: str, ratio: str, qp: int, seed: int) -> str:
    return f"{model}_{ratio}_qp{qp}_seed{seed}"


def fit_arrays(
    model: nn.Module,
    noisy: np.ndarray,
    clean: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    valid: tuple[np.ndarray, np.ndarray] | None = None,
    on_epoch=None,
) -> list[dict]:
    """Train ``model`` in place. Returns one history row per epoch.

    The objective is unrounded MSE, matching Keras ``fit``. Validation MSE in
    the history is the same objective on the provided validation arrays.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        PatchDataset(noisy, clean),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_uint8,
        drop_last=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-7)
    history = []
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        sse = 0.0
        count = 0
        started = time.perf_counter()
        for batch_noisy, batch_clean in loader:
            batch_noisy = batch_noisy.to(device)
            batch_clean = batch_clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_noisy)
            loss = torch.nn.functional.mse_loss(pred, batch_clean, reduction="sum")
            loss.backward()
            optimizer.step()
            sse += float(loss.item())
            count += batch_clean.numel()
        row = {
            "epoch": epoch,
            "train_mse": sse / count,
            "seconds": time.perf_counter() - started,
        }
        if valid is not None:
            row["val_mse"] = _objective_mse(model, valid[0], valid[1], batch_size, device)
        history.append(row)
        if on_epoch is not None:
            on_epoch(epoch, model, row)
        val_txt = f" val_mse={row['val_mse']:.6e}" if "val_mse" in row else ""
        print(
            f"epoch {epoch}/{epochs} train_mse={row['train_mse']:.6e}{val_txt} ({row['seconds']:.1f}s)",
            flush=True,
        )
    return history


@torch.no_grad()
def _objective_mse(model: nn.Module, noisy: np.ndarray, clean: np.ndarray, batch_size: int, device: torch.device) -> float:
    model.eval()
    sse = 0.0
    count = 0
    loader = DataLoader(
        PatchDataset(noisy, clean),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_uint8,
    )
    for batch_noisy, batch_clean in loader:
        pred = model(batch_noisy.to(device))
        err = torch.nn.functional.mse_loss(pred, batch_clean.to(device), reduction="sum")
        sse += float(err.item())
        count += batch_clean.numel()
    return sse / count


def train_setting(
    model_name: str,
    ratio: str,
    qp: int,
    seed: int,
    epochs: int,
    batch_size: int,
    data_dir: Path,
    results_dir: Path,
    max_train_patches: int | None = None,
    max_valid_patches: int | None = None,
    subset_seed: int = 0,
    lr: float | None = None,
    device: str = "cpu",
    eval_test: bool = True,
) -> dict:
    learning_rate = float(LEARNING_RATES[model_name] if lr is None else lr)
    dev = torch.device(device)
    print(f"loading {ratio} QP {qp}", flush=True)
    train = load_split(data_dir, ratio, qp, "train", require_usable=True)
    valid = load_split(data_dir, ratio, qp, "valid", require_usable=True)
    train_clean, train_noisy = subset_pair(train["clean"], train["noisy"], max_train_patches, subset_seed)
    valid_clean, valid_noisy = subset_pair(valid["clean"], valid["noisy"], max_valid_patches, subset_seed)
    set_seed(seed)
    model = build_model(model_name)
    started = time.perf_counter()
    history = fit_arrays(
        model,
        train_noisy,
        train_clean,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        seed=seed,
        device=dev,
        valid=(valid_noisy, valid_clean),
    )
    train_seconds = time.perf_counter() - started
    name = run_name(model_name, ratio, qp, seed)
    results_dir = Path(results_dir)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    torch.save(
        {
            "model": model_name,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "ratio": ratio,
            "qp": int(qp),
            "seed": int(seed),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": learning_rate,
            "max_train_patches": max_train_patches,
            "max_valid_patches": max_valid_patches,
            "subset_seed": int(subset_seed),
        },
        ckpt_path,
    )
    record = {
        "model": model_name,
        "ratio": ratio,
        "qp": int(qp),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": learning_rate,
        "subset_seed": int(subset_seed),
        "max_train_patches": max_train_patches,
        "max_valid_patches": max_valid_patches,
        "n_train_available": len(train["clean"]),
        "n_valid_available": len(valid["clean"]),
        "n_train": len(train_clean),
        "n_valid": len(valid_clean),
        "params": parameter_count(model),
        "flops_per_pixel": flops_per_pixel(model),
        "history": history,
        "train_seconds": train_seconds,
        "checkpoint": str(ckpt_path),
        "device": device,
        "torch": torch.__version__,
    }
    if eval_test:
        test = load_split(data_dir, ratio, qp, "test", require_usable=True)
        print(f"evaluating full test set n={len(test['clean'])}", flush=True)
        eval_started = time.perf_counter()
        metrics = evaluate_arrays(model, test["noisy"], test["clean"], batch_size=max(batch_size, 64), device=dev)
        metrics["seconds"] = time.perf_counter() - eval_started
        metrics["n_test"] = len(test["clean"])
        record["test"] = metrics
        print(
            f"test PSNR {metrics['psnr_baseline']:.4f} -> {metrics['psnr_restored']:.4f} "
            f"(gain {metrics['psnr_gain_db']:+.4f} dB)  "
            f"SSIM gain {metrics['ssim_gain']:+.5f}  "
            f"MSE reduction {metrics['mse_reduction_pct']:.3f}%",
            flush=True,
        )
    run_dir = results_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{name}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    record["result_path"] = str(out_path)
    return record
