import json
from pathlib import Path


def test_audit_file_records_unusable_4by3_release():
    audit = json.loads(Path("results/dataset_audit.json").read_text())
    assert audit["n_files"] == 36
    assert audit["n_usable"] == 24
    assert audit["n_unusable"] == 12
    assert audit["four_by_three_test_matches_2by1_qp50_test_multiset"] is True
    assert audit["four_by_three_train_arrays_equal_its_test"] is True
    assert audit["four_by_three_train_patchsize_field"] == 48
    assert audit["four_by_three_train_spatial"] == [34635, 64, 64]


def test_pilot_file_is_labeled_as_a_probe():
    pilot = json.loads(Path("results/pilot_2by1_qp50.json").read_text())
    assert pilot["epochs"] == 8
    assert set(pilot["models"]) == {"A", "E"}
    for name in ("A", "E"):
        assert len(pilot["models"][name]["curve"]) == 8
        assert pilot["models"][name]["full"]["n_images"] == 34635
