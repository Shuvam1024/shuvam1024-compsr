"""Turn per-run JSON into CSV, a markdown table, and plain matplotlib plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = [
    "model",
    "ratio",
    "qp",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "subset_seed",
    "max_train_patches",
    "n_train",
    "n_train_available",
    "n_valid",
    "n_test",
    "params",
    "flops_per_pixel",
    "mse_baseline",
    "mse_restored",
    "mse_reduction_pct",
    "psnr_baseline",
    "psnr_restored",
    "psnr_gain_db",
    "ssim_baseline",
    "ssim_restored",
    "ssim_gain",
    "final_train_mse",
    "final_val_mse",
    "train_seconds",
]


def load_runs(results_dir: Path) -> list[dict]:
    rows = []
    run_dir = Path(results_dir) / "runs"
    if not run_dir.exists():
        return rows
    for path in sorted(run_dir.glob("*.json")):
        record = json.loads(path.read_text())
        test = record.get("test") or {}
        history = record.get("history") or []
        final = history[-1] if history else {}
        rows.append(
            {
                "model": record["model"],
                "ratio": record["ratio"],
                "qp": int(record["qp"]),
                "seed": int(record["seed"]),
                "epochs": int(record["epochs"]),
                "batch_size": int(record["batch_size"]),
                "lr": record["lr"],
                "subset_seed": record.get("subset_seed"),
                "max_train_patches": record.get("max_train_patches"),
                "n_train": record.get("n_train"),
                "n_train_available": record.get("n_train_available"),
                "n_valid": record.get("n_valid"),
                "n_test": test.get("n_test", test.get("n_images")),
                "params": record.get("params"),
                "flops_per_pixel": record.get("flops_per_pixel"),
                "mse_baseline": test.get("mse_baseline"),
                "mse_restored": test.get("mse_restored"),
                "mse_reduction_pct": test.get("mse_reduction_pct"),
                "psnr_baseline": test.get("psnr_baseline"),
                "psnr_restored": test.get("psnr_restored"),
                "psnr_gain_db": test.get("psnr_gain_db"),
                "ssim_baseline": test.get("ssim_baseline"),
                "ssim_restored": test.get("ssim_restored"),
                "ssim_gain": test.get("ssim_gain"),
                "final_train_mse": final.get("train_mse"),
                "final_val_mse": final.get("val_mse"),
                "train_seconds": record.get("train_seconds"),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in COLUMNS})


def _group(rows: list[dict]):
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["ratio"], row["qp"], row["model"]), []).append(row)
    return groups


def _fmt(values: list[float], digits: int) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return f"{arr[0]:.{digits}f}"
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No runs recorded yet.\n"
    lines = [
        "Test metrics are measured on the full hosted test split against the original luma.",
        "Gains are restored minus the Lanczos-upscaled AV1 reconstruction.",
        "Where several seeds were trained, the cell is mean ± sample standard deviation (ddof=1).",
        "",
    ]
    ratios = []
    for row in rows:
        if row["ratio"] not in ratios:
            ratios.append(row["ratio"])
    for ratio in ratios:
        lines.append(f"### Ratio {ratio}")
        lines.append("")
        lines.append("| QP | Model | Params | PSNR gain (dB) | SSIM gain | MSE reduction (%) |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        subset = [row for row in rows if row["ratio"] == ratio]
        qps = sorted({row["qp"] for row in subset})
        models = []
        for row in subset:
            if row["model"] not in models:
                models.append(row["model"])
        groups = _group(subset)
        for qp in qps:
            gains = []
            for model in models:
                group = groups.get((ratio, qp, model), [])
                if not group:
                    continue
                gains.append((float(np.mean([g["psnr_gain_db"] for g in group])), model, group))
            best_model = max(gains, key=lambda item: item[0])[1] if gains else None
            for model in models:
                group = groups.get((ratio, qp, model), [])
                if not group:
                    continue
                mark = model + (" *" if model == best_model else "")
                lines.append(
                    "| {qp} | {model} | {params} | {psnr} | {ssim} | {mse} |".format(
                        qp=qp,
                        model=mark,
                        params=int(group[0]["params"]),
                        psnr=_fmt([g["psnr_gain_db"] for g in group], 3),
                        ssim=_fmt([g["ssim_gain"] for g in group], 4),
                        mse=_fmt([g["mse_reduction_pct"] for g in group], 2),
                    )
                )
        lines.append("")
        lines.append("\\* Highest mean PSNR gain at that QP.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_gain_vs_params(rows: list[dict], path: Path, ratio: str = "2by1", qp: int = 50) -> None:
    plt = _style()
    chosen = [row for row in rows if row["ratio"] == ratio and int(row["qp"]) == qp]
    if not chosen:
        return
    models = []
    for row in chosen:
        if row["model"] not in models:
            models.append(row["model"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model in models:
        group = [row for row in chosen if row["model"] == model]
        xs = float(group[0]["params"])
        ys = [float(row["psnr_gain_db"]) for row in group]
        y = float(np.mean(ys))
        yerr = float(np.std(ys, ddof=1)) if len(ys) > 1 else None
        ax.errorbar(xs, y, yerr=yerr, fmt="o", capsize=3, label=model)
        ax.annotate(model, (xs, y), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Test PSNR gain (dB)")
    ax.set_title(f"PSNR gain vs size, ratio {ratio}, QP {qp}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gain_vs_qp(rows: list[dict], path: Path) -> None:
    plt = _style()
    ratios = []
    for row in rows:
        if row["ratio"] not in ratios:
            ratios.append(row["ratio"])
    if not ratios:
        return
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.0), sharey=True)
    if len(ratios) == 1:
        axes = [axes]
    for ax, ratio in zip(axes, ratios):
        subset = [row for row in rows if row["ratio"] == ratio]
        models = []
        for row in subset:
            if row["model"] not in models:
                models.append(row["model"])
        for model in models:
            group = [row for row in subset if row["model"] == model]
            qps = sorted({int(row["qp"]) for row in group})
            means = []
            for qp in qps:
                vals = [float(row["psnr_gain_db"]) for row in group if int(row["qp"]) == qp]
                means.append(float(np.mean(vals)))
            ax.plot(qps, means, marker="o", label=model)
        ax.set_title(f"Ratio {ratio}")
        ax.set_xlabel("AV1 QP")
        ax.set_ylabel("Test PSNR gain (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    rows = load_runs(results_dir)
    csv_path = results_dir / "metrics.csv"
    table_path = results_dir / "table.md"
    write_csv(rows, csv_path)
    table = render_table(rows)
    table_path.write_text(table)
    plot_gain_vs_params(rows, results_dir / "psnr_gain_vs_params.png")
    plot_gain_vs_qp(rows, results_dir / "psnr_gain_vs_qp.png")
    return {"rows": len(rows), "csv": str(csv_path), "table": str(table_path)}
