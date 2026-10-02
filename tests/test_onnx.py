from pathlib import Path

import pytest

from compsr.onnx_export import export_model


@pytest.mark.slow
def test_onnx_matches_pytorch_on_even_sizes(tmp_path: Path):
    report = export_model("E", tmp_path / "e.onnx", height=32, width=32)
    assert report["max_abs_diff"] < 1e-4
    assert {check["batch"] for check in report["checks"]} == {1, 2}
