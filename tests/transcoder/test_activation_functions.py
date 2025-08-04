import torch
import pytest
from circuit_tracer.transcoder.activation_functions import JumpReLU, TopK


def test_JumpReLU_filters_activations_below_threshold():
    threshold = torch.tensor([1.0, 0.5, 1.5, 2.0, 0.8])
    act_fn = JumpReLU(threshold=threshold, bandwidth=1.0)

    # Test input with values both above and below threshold
    x = torch.tensor([[-1.0, 0.5, 1.0, 1.5, 2.0], [2.0, 1.0, 2.0, 3.0, 0.5]])
    result = act_fn(x)

    # Values below threshold should be 0, values > threshold should be preserved
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0], [2.0, 1.0, 2.0, 3.0, 0.0]])

    assert torch.allclose(result, expected)


def test_JumpReLU_preserves_activations_above_threshold():
    threshold = torch.tensor([0.5, 0.3, 1.0, 2.0])
    act_fn = JumpReLU(threshold=threshold, bandwidth=1.0)

    # Test input with all values above threshold
    x = torch.tensor([[1.0, 2.0, 3.5, 10.0], [0.8, 1.5, 2.0, 5.0]])
    result = act_fn(x)

    # All values should be preserved since they're above threshold
    assert torch.allclose(result, x)


def test_JumpReLU_zeros_activations_below_threshold():
    threshold = torch.tensor([2.0, 3.0, 2.5, 2.1])
    act_fn = JumpReLU(threshold=threshold, bandwidth=1.0)

    # Test input with all values below threshold
    x = torch.tensor([[-1.0, 0.0, 1.0, 1.9], [1.5, 2.5, 2.0, 1.8]])
    result = act_fn(x)

    # All values should be zero since they're below threshold
    expected = torch.zeros_like(x)
    assert torch.allclose(result, expected)


def test_TopK_keeps_top_k_values():
    act_fn = TopK(k=2)

    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 2.0]])
    result = act_fn(x)

    # Should keep the 2 largest values in each row and zero the rest
    expected = torch.tensor([[0.0, 5.0, 3.0, 0.0], [4.0, 0.0, 6.0, 0.0]])
    assert torch.allclose(result, expected)


def test_TopK_keeps_single_maximum():
    act_fn = TopK(k=1)

    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 2.0]])
    result = act_fn(x)

    # Should keep only the maximum value in each row
    expected = torch.tensor([[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
    assert torch.allclose(result, expected)


def test_TopK_handles_negative_values():
    act_fn = TopK(k=2)

    x = torch.tensor([[-1.0, -5.0, -3.0, -2.0], [-4.0, -1.0, -6.0, -3.0]])
    result = act_fn(x)

    # Should keep the 2 largest (least negative) values in each row
    expected = torch.tensor([[-1.0, 0.0, 0.0, -2.0], [0.0, -1.0, 0.0, -3.0]])
    assert torch.allclose(result, expected)


def test_TopK_handles_mixed_positive_negative():
    act_fn = TopK(k=3)

    x = torch.tensor([[-1.0, 2.0, -3.0, 4.0, 0.0], [1.0, -2.0, 5.0, -1.0, 3.0]])
    result = act_fn(x)

    # Should keep the 3 largest values in each row
    expected = torch.tensor([[0.0, 2.0, 0.0, 4.0, 0.0], [1.0, 0.0, 5.0, 0.0, 3.0]])
    assert torch.allclose(result, expected)


def test_TopK_k_equals_tensor_size():
    act_fn = TopK(k=4)

    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 2.0]])
    result = act_fn(x)

    # Should keep all values since k equals tensor size
    assert torch.allclose(result, x)


def test_TopK_k_larger_than_tensor_size():
    act_fn = TopK(k=10)

    x = torch.tensor([[1.0, 5.0, 3.0], [2.0, 4.0, 1.0]])

    # Should raise RuntimeError when k > tensor size
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        act_fn(x)


def test_TopK_handles_tied_values():
    act_fn = TopK(k=2)

    x = torch.tensor([[1.0, 3.0, 3.0, 2.0], [4.0, 2.0, 2.0, 1.0]])
    result = act_fn(x)

    # Should keep 2 values in each row, including at least one of the tied values
    # The exact behavior depends on torch.topk tie-breaking
    assert torch.sum(result != 0, dim=-1).eq(2).all()
    assert torch.max(result, dim=-1).values.allclose(torch.tensor([3.0, 4.0]))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (5,)])
def test_TopK_works_with_different_shapes(shape: tuple[int, ...]):
    act_fn = TopK(k=2)

    x = torch.randn(shape)
    result = act_fn(x)

    # Should have same shape as input
    assert result.shape == x.shape

    # Should have exactly k non-zero values along last dimension
    non_zero_count = torch.sum(result != 0, dim=-1)
    expected_count = min(2, shape[-1])
    assert torch.all(non_zero_count == expected_count)


def test_TopK_preserves_dtype():
    act_fn = TopK(k=2)

    x = torch.tensor([[1.0, 5.0, 3.0, 2.0], [4.0, 1.0, 6.0, 2.0]], dtype=torch.float32)
    result = act_fn(x)

    assert result.dtype == torch.float32


def test_TopK_2d_tensor():
    act_fn = TopK(k=2)

    x = torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    result = act_fn(x)

    # Each row should have 2 non-zero values (the top 2 in each row)
    expected = torch.tensor([[0.0, 5.0, 3.0], [4.0, 0.0, 6.0]])
    assert torch.allclose(result, expected)


def test_JumpReLU_3d_tensor():
    # Test JumpReLU with 3D input and 2D threshold
    # Input shape: (2, 3, 4), Threshold shape: (3, 4)
    threshold = torch.tensor([[0.5, 1.0, 0.8, 1.2], [0.3, 1.5, 0.9, 0.7], [1.1, 0.4, 1.3, 0.6]])
    act_fn = JumpReLU(threshold=threshold, bandwidth=1.0)

    x = torch.tensor(
        [
            [[0.6, 1.2, 0.7, 1.5], [0.2, 1.8, 1.0, 0.8], [1.2, 0.5, 1.4, 0.5]],
            [[0.4, 0.9, 0.8, 1.3], [0.5, 1.6, 0.8, 0.9], [1.0, 0.3, 1.2, 0.7]],
        ]
    )
    result = act_fn(x)

    # Values above threshold should be preserved, values below should be 0
    expected = torch.tensor(
        [
            [[0.6, 1.2, 0.0, 1.5], [0.0, 1.8, 1.0, 0.8], [1.2, 0.5, 1.4, 0.0]],
            [[0.0, 0.0, 0.0, 1.3], [0.5, 1.6, 0.0, 0.9], [0.0, 0.0, 0.0, 0.7]],
        ]
    )
    assert torch.allclose(result, expected)


def test_TopK_3d_tensor():
    act_fn = TopK(k=2)

    # Input shape: (2, 3, 4) - should keep top 2 values along last dimension
    x = torch.tensor(
        [
            [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 6.0, 2.0], [3.0, 2.0, 1.0, 4.0]],
            [[2.0, 1.0, 4.0, 3.0], [1.0, 5.0, 2.0, 6.0], [4.0, 3.0, 2.0, 1.0]],
        ]
    )
    result = act_fn(x)

    # Should keep the 2 largest values along the last dimension for each position
    expected = torch.tensor(
        [
            [[0.0, 4.0, 0.0, 3.0], [5.0, 0.0, 6.0, 0.0], [3.0, 0.0, 0.0, 4.0]],
            [[0.0, 0.0, 4.0, 3.0], [0.0, 5.0, 0.0, 6.0], [4.0, 3.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(result, expected)
