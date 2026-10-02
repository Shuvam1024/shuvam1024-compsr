import torch

from compsr.models import (
    NOTEBOOK_COMPLEXITY,
    build_model,
    flops_per_pixel,
    parameter_count,
)


def test_parameter_counts_and_flops_match_notebooks():
    for name, expected in NOTEBOOK_COMPLEXITY.items():
        model = build_model(name)
        assert parameter_count(model) == expected["params"]
        assert flops_per_pixel(model, 128) == expected["flops_per_pixel"]
        assert flops_per_pixel(model, 64) == expected["flops_per_pixel"]


def test_output_shape_matches_input_even_and_odd():
    for name in NOTEBOOK_COMPLEXITY:
        model = build_model(name).eval()
        for size in (48, 64, 47):
            sample = torch.rand(2, 1, size, size)
            with torch.no_grad():
                out = model(sample)
            assert out.shape == sample.shape


def test_same_seed_same_init():
    torch.manual_seed(7)
    first = build_model("E")
    torch.manual_seed(7)
    second = build_model("E")
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
