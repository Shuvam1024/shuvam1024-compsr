import json
from pathlib import Path

from compsr.report import render_efficiency, render_table, write_report


def _row(model, qp, seed, gain, ssim, mse, params=100):
    return {
        "model": model,
        "ratio": "2by1",
        "qp": qp,
        "seed": seed,
        "params": params,
        "psnr_gain_db": gain,
        "ssim_gain": ssim,
        "mse_reduction_pct": mse,
    }


def test_table_marks_best_mean_and_reports_sample_std():
    rows = [
        _row("A", 50, 0, 0.20, 0.004, 4.0, params=24953),
        _row("A", 50, 1, 0.30, 0.006, 6.0, params=24953),
        _row("E", 50, 0, 0.10, 0.002, 2.0, params=8277),
    ]
    table = render_table(rows)
    assert "A *" in table
    assert "0.250 ± 0.071" in table
    assert "8277" in table
    assert "E *" not in table
    assert "Model A has the highest mean PSNR gain at every QP in this table." in table
    assert "Model E has the lowest mean PSNR gain at every QP in this table." in table


def test_efficiency_table_uses_only_recorded_fields():
    efficiency = {
        "device": "cpu",
        "torch": "2.14.1+cpu",
        "cpu": "x86_64",
        "cpu_count": 4,
        "torch_threads": 4,
        "cuda_available": False,
        "models": [
            {
                "model": "E",
                "params": 8277,
                "flops_per_pixel": 9028,
                "patch_throughput": {
                    "images_per_s": 12.5,
                    "height": 64,
                    "width": 64,
                    "batch": 16,
                },
                "frame_latency": {"ms_per_frame": 40.0, "width": 1280, "height": 720},
            }
        ],
    }
    onnx_rows = [
        {
            "model": "E",
            "max_abs_diff": 1.2e-6,
            "onnxruntime_images_per_s": 20.0,
            "pytorch_images_per_s": 15.0,
            "timing_batch": 8,
            "timing_size": 64,
        }
    ]
    text = render_efficiency(efficiency, onnx_rows)
    assert "Device `cpu`" in text
    assert "CUDA available at measurement time: False." in text
    assert "| E | 8277 | 9028 | 12.50 (64×64, batch 16) | 40.00 (1280×720) |" in text
    efficiency["models"][0]["patch_throughput"]["images_per_s_samples"] = [10.0, 12.0]
    efficiency["models"][0]["frame_latency"]["ms_per_frame_samples"] = [40.0, 42.0]
    repeated = render_efficiency(efficiency, onnx_rows)
    assert "11.00 ± 1.41 (64×64, batch 16)" in repeated
    assert "41.00 ± 1.41 (1280×720)" in repeated
    assert "| E | 1.200e-06 | 20.00 | 15.00 | batch 8, 64×64 |" in text


def test_write_report_splices_results_and_efficiency(tmp_path: Path):
    run = {
        "model": "A",
        "ratio": "2by1",
        "qp": 50,
        "seed": 0,
        "epochs": 6,
        "batch_size": 256,
        "lr": 0.001,
        "subset_seed": 0,
        "max_train_patches": 16384,
        "n_train": 16384,
        "n_train_available": 271788,
        "n_valid": 4096,
        "params": 24953,
        "flops_per_pixel": 49770,
        "history": [{"epoch": 6, "train_mse": 0.001, "val_mse": 0.001}],
        "train_seconds": 10.0,
        "test": {
            "n_test": 34635,
            "mse_baseline": 0.002,
            "mse_restored": 0.0019,
            "mse_reduction_pct": 5.0,
            "psnr_baseline": 26.0,
            "psnr_restored": 26.2,
            "psnr_gain_db": 0.2,
            "ssim_baseline": 0.8,
            "ssim_restored": 0.81,
            "ssim_gain": 0.01,
        },
    }
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "A_2by1_qp50_seed0.json").write_text(json.dumps(run))
    (tmp_path / "efficiency.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "torch": "2.14.1+cpu",
                "cpu": "x86_64",
                "cpu_count": 4,
                "torch_threads": 4,
                "cuda_available": False,
                "models": [
                    {
                        "model": "A",
                        "params": 24953,
                        "flops_per_pixel": 49770,
                        "patch_throughput": {
                            "images_per_s": 3.0,
                            "height": 64,
                            "width": 64,
                            "batch": 16,
                        },
                        "frame_latency": {"ms_per_frame": 90.0, "width": 1280, "height": 720},
                    }
                ],
            }
        )
    )
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- RESULTS -->\n<!-- /RESULTS -->\nmid\n<!-- EFFICIENCY -->\n<!-- /EFFICIENCY -->\nafter\n")
    info = write_report(tmp_path, readme=readme)
    assert info["rows"] == 1
    text = readme.read_text()
    assert "0.200" in text
    assert "| A | 24953 | 49770 | 3.00 (64×64, batch 16) | 90.00 (1280×720) |" in text
    assert text.startswith("before\n<!-- RESULTS -->")
    assert text.endswith("<!-- /EFFICIENCY -->\nafter\n")
    csv = (tmp_path / "metrics.csv").read_text()
    assert "34635" in csv
    assert (tmp_path / "psnr_gain_vs_params.png").exists()
    assert (tmp_path / "psnr_gain_vs_qp.png").exists()
