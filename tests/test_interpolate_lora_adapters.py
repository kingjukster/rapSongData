import torch

from scripts.interpolate_lora_adapters import interpolate_states


def test_rank_concatenation_exactly_interpolates_lora_deltas() -> None:
    torch.manual_seed(7)
    a_key = "layer.lora_A.weight"
    b_key = "layer.lora_B.weight"
    left = {a_key: torch.randn(2, 5), b_key: torch.randn(4, 2)}
    right = {a_key: torch.randn(2, 5), b_key: torch.randn(4, 2)}
    output = interpolate_states(
        left,
        right,
        left_weight=0.5,
        right_weight=0.5,
        left_scale=2.0,
        right_scale=2.0,
        output_scale=1.0,
    )
    expected = 0.5 * 2.0 * (left[b_key] @ left[a_key]) + 0.5 * 2.0 * (right[b_key] @ right[a_key])
    observed = output[b_key] @ output[a_key]
    torch.testing.assert_close(observed, expected)
    assert output[a_key].shape == (4, 5)
    assert output[b_key].shape == (4, 4)
