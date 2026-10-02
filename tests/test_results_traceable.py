"""Committed metrics must be recomputable from the run files.

This test is copied into tests/ only after the grid, bench, and ONNX report exist.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from compsr.metrics import mse_reduction_percent, psnr_from_mse
from compsr.models import LEARNING_RATES, NOTEBOOK_COMPLEXITY, build_model
from compsr.report import load_runs, render_efficiency, render_table

ROOT = Path("results")
NOTEBOOK_BASELINE_MSE = 0.0022601327  # printed by the 2:1 QP 50 notebook test cells


def test_grid_covers_the_protocol_and_only_usable_settings():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    expected = []
    for ratio in protocol["ratios"]:
        assert ratio != "4by3"
        for qp in protocol["qps"]:
            seeds = protocol["seeds"]
            headline = protocol["headline"]
            if ratio == headline["ratio"] and int(qp) == int(headline["qp"]):
                seeds = headline["seeds"]
            for model in protocol["models"]:
                for seed in seeds:
                    expected.append(f"{model}_{ratio}_qp{qp}_seed{seed}.json")
    found = sorted(path.name for path in (ROOT / "runs").glob("*.json"))
    assert found == sorted(expected)
    assert len(found) == 50
    assert protocol["epochs"] == 6
    assert protocol["batch_size"] == 256
    assert protocol["max_train_patches"] == 16384
    assert protocol["subset_seed"] == 0


def test_each_run_matches_closed_form_metrics_and_notebook_complexity():
    baselines = defaultdict(list)
    for path in sorted((ROOT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        test = record["test"]
        assert record["epochs"] == 6
        assert len(record["history"]) == 6
        assert record["n_train"] == 16384
        assert record["n_valid"] == 4096
        assert test["n_images"] == 34635
        assert test["n_test"] == 34635
        assert test["n_pixels"] == 34635 * 64 * 64
        assert record["params"] == NOTEBOOK_COMPLEXITY[record["model"]]["params"]
        assert record["flops_per_pixel"] == NOTEBOOK_COMPLEXITY[record["model"]]["flops_per_pixel"]
        assert record["lr"] == LEARNING_RATES[record["model"]]
        mse_b = test["mse_baseline"]
        mse_r = test["mse_restored"]
        assert abs(test["psnr_baseline"] - psnr_from_mse(mse_b)) < 1e-9
        assert abs(test["psnr_restored"] - psnr_from_mse(mse_r)) < 1e-9
        assert abs(test["psnr_gain_db"] - (test["psnr_restored"] - test["psnr_baseline"])) < 1e-9
        assert abs(test["mse_reduction_pct"] - mse_reduction_percent(mse_b, mse_r)) < 1e-6
        assert abs(test["ssim_gain"] - (test["ssim_restored"] - test["ssim_baseline"])) < 1e-9
        baselines[(record["ratio"], record["qp"])].append(mse_b)
        ckpt = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        model = build_model(record["model"])
        model.load_state_dict(ckpt["state_dict"])
        assert ckpt["epochs"] == 6
        assert ckpt["seed"] == record["seed"]
    for key, values in baselines.items():
        assert max(values) - min(values) == 0.0, key
    qp50 = baselines[("2by1", 50)][0]
    assert abs(qp50 - NOTEBOOK_BASELINE_MSE) < 1e-8


def test_csv_table_and_readme_are_rendered_from_the_run_files():
    rows = load_runs(ROOT)
    table = render_table(rows)
    assert (ROOT / "table.md").read_text() == table
    readme = Path("README.md").read_text()
    assert table.strip() in readme
    with (ROOT / "metrics.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows) == 50
    by_key = {(row["model"], row["ratio"], int(row["qp"]), int(row["seed"])): row for row in rows}
    for csv_row in csv_rows:
        key = (csv_row["model"], csv_row["ratio"], int(csv_row["qp"]), int(csv_row["seed"]))
        src = by_key[key]
        for field in ("psnr_gain_db", "ssim_gain", "mse_reduction_pct", "mse_baseline", "n_test"):
            assert float(csv_row[field]) == float(src[field])
    assert (ROOT / "psnr_gain_vs_params.png").stat().st_size > 1000
    assert (ROOT / "psnr_gain_vs_qp.png").stat().st_size > 1000


def test_efficiency_readme_matches_measurement_files():
    efficiency = json.loads((ROOT / "efficiency.json").read_text())
    onnx_rows = json.loads((ROOT / "onnx_report.json").read_text())
    text = render_efficiency(efficiency, onnx_rows)
    assert (ROOT / "efficiency.md").read_text() == text
    assert text.strip() in Path("README.md").read_text()
    assert efficiency["device"] == "cpu"
    assert efficiency["cuda_available"] is False
    assert [row["model"] for row in efficiency["models"]] == ["A", "B", "C", "D", "E"]
    assert [row["model"] for row in onnx_rows] == ["A", "B", "C", "D", "E"]
    for row in efficiency["models"]:
        name = row["model"]
        assert row["params"] == NOTEBOOK_COMPLEXITY[name]["params"]
        assert row["flops_per_pixel"] == NOTEBOOK_COMPLEXITY[name]["flops_per_pixel"]
        assert row["notebook_params"] == row["params"]
    for row in onnx_rows:
        assert row["max_abs_diff"] < 1e-4
