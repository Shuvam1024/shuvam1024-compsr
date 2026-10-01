"""Command line for download, training, evaluation, generation, and reports."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml

from compsr.bench import measure_frame_latency, measure_throughput
from compsr.data import (
    QPS,
    RATIOS,
    SPLITS,
    download_selection,
    identical_groups,
    load_manifest,
    load_split,
)
from compsr.models import (
    LEARNING_RATES,
    NOTEBOOK_COMPLEXITY,
    build_model,
    flops_per_pixel,
    parameter_count,
)
from compsr.report import write_report
from compsr.train import run_name, train_setting


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def cmd_download(args) -> int:
    ratio = None if args.ratio == "all" else args.ratio
    qp = None if args.qp == "all" else int(args.qp)
    split = None if args.split == "all" else args.split
    saved = download_selection(
        Path(args.data_dir),
        ratio=ratio,
        qp=qp,
        split=split,
        force=args.force,
        usable_only=args.usable_only,
    )
    print(f"downloaded {len(saved)} archives")
    return 0


def cmd_audit(args) -> int:
    manifest = load_manifest()
    groups = identical_groups(manifest)
    usable = [entry for entry in manifest["files"] if entry["usable"]]
    unusable = [entry for entry in manifest["files"] if not entry["usable"]]
    report = {
        "n_files": len(manifest["files"]),
        "n_usable": len(usable),
        "n_unusable": len(unusable),
        "identical_sha256_groups": groups,
        "unusable": [
            {"filename": entry["filename"], "sha256": entry["sha256"], "note": entry.get("note", "")}
            for entry in unusable
        ],
    }
    if args.data_dir and Path(args.data_dir).exists() and args.check_arrays:
        reference = load_split(Path(args.data_dir), "2by1", 50, "test", require_usable=True)
        sample = load_split(Path(args.data_dir), "4by3", 50, "test", require_usable=False)
        report["four_by_three_test_matches_2by1_qp50_test_multiset"] = _same_multiset(
            reference["clean"], reference["noisy"], sample["clean"], sample["noisy"]
        )
        report["four_by_three_test_patchsize_field"] = sample["patchsize"]
        report["four_by_three_test_spatial"] = int(sample["clean"].shape[-1])
        train = load_split(Path(args.data_dir), "4by3", 20, "train", require_usable=False)
        report["four_by_three_train_patchsize_field"] = train["patchsize"]
        report["four_by_three_train_spatial"] = list(train["clean"].shape)
        report["four_by_three_train_arrays_equal_its_test"] = bool(
            (train["clean"] == sample["clean"]).all() and (train["noisy"] == sample["noisy"]).all()
        )
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")
    print(text)
    return 0


def _same_multiset(clean_a, noisy_a, clean_b, noisy_b) -> bool:
    import hashlib

    def bags(clean, noisy):
        clean = clean.reshape(len(clean), -1)
        noisy = noisy.reshape(len(noisy), -1)
        found = {}
        for index in range(len(clean)):
            key = hashlib.blake2b(clean[index].tobytes() + noisy[index].tobytes(), digest_size=8).digest()
            found[key] = found.get(key, 0) + 1
        return found

    return bags(clean_a, noisy_a) == bags(clean_b, noisy_b)


def cmd_train(args) -> int:
    cfg = _load_config(args.config)
    train_setting(
        model_name=args.model,
        ratio=args.ratio,
        qp=int(args.qp),
        seed=int(args.seed),
        epochs=int(args.epochs if args.epochs is not None else cfg.get("epochs", 1)),
        batch_size=int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 256)),
        data_dir=Path(args.data_dir or cfg.get("data_dir", "data")),
        results_dir=Path(args.results_dir or cfg.get("results_dir", "results")),
        max_train_patches=args.max_train_patches if args.max_train_patches is not None else cfg.get("max_train_patches"),
        max_valid_patches=args.max_valid_patches if args.max_valid_patches is not None else cfg.get("max_valid_patches"),
        subset_seed=int(cfg.get("subset_seed", 0) if args.subset_seed is None else args.subset_seed),
        lr=args.lr,
        device=args.device or cfg.get("device", "cpu"),
        eval_test=not args.no_eval,
    )
    return 0


def cmd_train_grid(args) -> int:
    cfg = _load_config(args.config)
    data_dir = Path(args.data_dir or cfg.get("data_dir", "data"))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    ratios = list(cfg["ratios"])
    qps = [int(q) for q in cfg["qps"]]
    models = list(cfg["models"])
    seeds = [int(s) for s in cfg.get("seeds", [0])]
    headline = cfg.get("headline") or {}
    headline_seeds = [int(s) for s in headline.get("seeds", seeds)]
    for ratio in ratios:
        for qp in qps:
            run_seeds = seeds
            if ratio == headline.get("ratio") and qp == int(headline.get("qp", -1)):
                run_seeds = headline_seeds
            for model in models:
                for seed in run_seeds:
                    out = results_dir / "runs" / f"{run_name(model, ratio, qp, seed)}.json"
                    if out.exists() and not args.force:
                        print(f"skip existing {out.name}", flush=True)
                        continue
                    train_setting(
                        model_name=model,
                        ratio=ratio,
                        qp=qp,
                        seed=seed,
                        epochs=epochs,
                        batch_size=batch_size,
                        data_dir=data_dir,
                        results_dir=results_dir,
                        max_train_patches=cfg.get("max_train_patches"),
                        max_valid_patches=cfg.get("max_valid_patches"),
                        subset_seed=int(cfg.get("subset_seed", 0)),
                        device=cfg.get("device", "cpu"),
                        eval_test=True,
                    )
    protocol = {
        "epochs": epochs,
        "batch_size": batch_size,
        "ratios": ratios,
        "qps": qps,
        "models": models,
        "seeds": seeds,
        "headline": headline,
        "max_train_patches": cfg.get("max_train_patches"),
        "max_valid_patches": cfg.get("max_valid_patches"),
        "subset_seed": cfg.get("subset_seed", 0),
        "learning_rates": {name: LEARNING_RATES[name] for name in models},
        "optimizer": "Adam",
        "adam_eps": 1e-7,
        "loss": "mse on unrounded network output in [0, 1]",
        "eval": "8-bit rounded output; dataset-wide MSE/PSNR; mean per-patch SSIM",
        "device": cfg.get("device", "cpu"),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": platform.processor(),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    write_report(results_dir)
    return 0


def cmd_evaluate(args) -> int:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    loaded = load_split(Path(args.data_dir), args.ratio or ckpt["ratio"], int(args.qp or ckpt["qp"]), "test")
    from compsr.evaluate import evaluate_arrays

    metrics = evaluate_arrays(model, loaded["noisy"], loaded["clean"], batch_size=args.batch_size, device=torch.device(args.device))
    metrics["n_test"] = len(loaded["clean"])
    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


def cmd_bench(args) -> int:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    rows = []
    for name in args.models.split(","):
        model = build_model(name.strip())
        patch = measure_throughput(model, size=args.size, batch=args.batch, device=device)
        frame = measure_frame_latency(model, height=args.frame_height, width=args.frame_width, device=device)
        rows.append(
            {
                "model": name.strip(),
                "params": parameter_count(model),
                "flops_per_pixel": flops_per_pixel(model),
                "notebook_params": NOTEBOOK_COMPLEXITY[name.strip()]["params"],
                "notebook_flops_per_pixel": NOTEBOOK_COMPLEXITY[name.strip()]["flops_per_pixel"],
                "patch_throughput": patch,
                "frame_latency": frame,
            }
        )
        print(
            f"{name.strip()}: {patch['images_per_s']:.2f} patches/s at {args.size}x{args.size} "
            f"batch {args.batch}; {frame['ms_per_frame']:.1f} ms/frame at {args.frame_width}x{args.frame_height}",
            flush=True,
        )
    payload = {"device": device, "torch": torch.__version__, "models": rows}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def cmd_export_onnx(args) -> int:
    from compsr.onnx_export import export_model

    results = []
    for name in args.models.split(","):
        path = Path(args.output_dir) / f"model_{name.strip()}.onnx"
        results.append(export_model(name.strip(), path, height=args.size, width=args.size))
        print(f"{name.strip()}: max abs diff {results[-1]['max_abs_diff']:.3e}", flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(results, indent=2) + "\n")
    worst = max(row["max_abs_diff"] for row in results)
    if worst > args.atol:
        print(f"ONNX mismatch {worst} exceeds atol {args.atol}")
        return 1
    return 0


def cmd_generate(args) -> int:
    from compsr.generate import div2k_id, generate_archive, split_ids

    images = sorted(Path(args.images_dir).glob(args.glob))
    if args.split:
        ids = []
        chosen = []
        id_to_path = {}
        for image in images:
            image_id = div2k_id(image)
            if image_id is None:
                continue
            id_to_path[image_id] = image
            ids.append(image_id)
        keep = split_ids(ids, args.split, args.split_seed)
        chosen = [id_to_path[i] for i in sorted(keep)]
        images = chosen
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit("no images matched")
    info = generate_archive(
        images,
        Path(args.output),
        ratio=args.ratio,
        qp=int(args.qp),
        patch=args.patch_size,
        patches_per_image=args.patches_per_image,
        seed=args.seed,
        cpu_used=args.cpu_used,
        matrix=args.matrix,
    )
    print(json.dumps(info, indent=2))
    return 0


def cmd_report(args) -> int:
    info = write_report(Path(args.results_dir))
    print(json.dumps(info))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compsr", description="AV1 compression restoration CNNs")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download hosted npz archives and verify sha256")
    download.add_argument("--data-dir", default="data")
    download.add_argument("--ratio", default="all", choices=["all", *RATIOS])
    download.add_argument("--qp", default="all", choices=["all", *[str(q) for q in QPS]])
    download.add_argument("--split", default="all", choices=["all", *SPLITS])
    download.add_argument("--usable-only", action="store_true")
    download.add_argument("--force", action="store_true")
    download.set_defaults(func=cmd_download)

    audit = sub.add_parser("audit", help="report duplicate and unusable hosted archives")
    audit.add_argument("--data-dir", default="data")
    audit.add_argument("--output", default="")
    audit.add_argument("--check-arrays", action="store_true")
    audit.set_defaults(func=cmd_audit)

    train = sub.add_parser("train", help="train one model on one ratio/QP")
    train.add_argument("--config", default="configs/train.yaml")
    train.add_argument("--model", required=True, choices=sorted(LEARNING_RATES))
    train.add_argument("--ratio", required=True, choices=list(RATIOS))
    train.add_argument("--qp", required=True, type=int)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--max-train-patches", type=int, default=None)
    train.add_argument("--max-valid-patches", type=int, default=None)
    train.add_argument("--subset-seed", type=int, default=None)
    train.add_argument("--data-dir", default=None)
    train.add_argument("--results-dir", default=None)
    train.add_argument("--device", default=None)
    train.add_argument("--no-eval", action="store_true")
    train.set_defaults(func=cmd_train)

    grid = sub.add_parser("train-grid", help="train every model/setting listed in the config")
    grid.add_argument("--config", default="configs/train.yaml")
    grid.add_argument("--data-dir", default=None)
    grid.add_argument("--results-dir", default=None)
    grid.add_argument("--force", action="store_true")
    grid.set_defaults(func=cmd_train_grid)

    evaluate = sub.add_parser("evaluate", help="score a checkpoint on the full test split")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--ratio", default=None)
    evaluate.add_argument("--qp", type=int, default=None)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", default="")
    evaluate.set_defaults(func=cmd_evaluate)

    bench = sub.add_parser("bench", help="measure inference throughput")
    bench.add_argument("--models", default="A,B,C,D,E")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--size", type=int, default=64)
    bench.add_argument("--batch", type=int, default=16)
    bench.add_argument("--frame-height", type=int, default=720)
    bench.add_argument("--frame-width", type=int, default=1280)
    bench.add_argument("--output", default="results/efficiency.json")
    bench.set_defaults(func=cmd_bench)

    export = sub.add_parser("export-onnx", help="export networks and compare ONNX Runtime outputs")
    export.add_argument("--models", default="A,B,C,D,E")
    export.add_argument("--output-dir", default="results/onnx")
    export.add_argument("--size", type=int, default=64)
    export.add_argument("--atol", type=float, default=1e-4)
    export.add_argument("--report", default="results/onnx_report.json")
    export.set_defaults(func=cmd_export_onnx)

    generate = sub.add_parser("generate", help="build an npz from RGB images via Lanczos + AV1")
    generate.add_argument("--images-dir", required=True)
    generate.add_argument("--glob", default="*.png")
    generate.add_argument("--output", required=True)
    generate.add_argument("--ratio", required=True, choices=list(RATIOS))
    generate.add_argument("--qp", required=True, type=int)
    generate.add_argument("--patch-size", type=int, default=48)
    generate.add_argument("--patches-per-image", type=int, default=16)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--split", choices=["train", "valid", "test"], default=None)
    generate.add_argument("--split-seed", type=int, default=0)
    generate.add_argument("--limit", type=int, default=0)
    generate.add_argument("--cpu-used", type=int, default=6)
    generate.add_argument("--matrix", default="bt709", choices=["bt709", "bt601"])
    generate.set_defaults(func=cmd_generate)

    report = sub.add_parser("report", help="write metrics.csv, table.md, and plots from run JSON")
    report.add_argument("--results-dir", default="results")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
