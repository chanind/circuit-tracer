import torch
import pytest
from torch import nn

from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.transcoder.activation_functions import JumpReLU


def test_CrossLayerTranscoder_forward_with_relu_no_skip():
    """Test CrossLayerTranscoder forward pass with ReLU activation and no skip connection."""
    d_model = 3
    d_transcoder = 4
    layers = [0, 1]

    transcoder = CrossLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=nn.ReLU(),
        layers=layers,
        skip_connection=False,
    )

    # Set hardcoded parameters
    with torch.no_grad():
        # W_enc: n_layers x d_model x d_transcoder (2 x 3 x 4)
        transcoder.W_enc[0] = torch.tensor(
            [[0.5, -0.3, 0.2, 0.1], [0.1, 0.4, -0.2, 0.3], [0.2, -0.1, 0.5, -0.2]]
        )
        transcoder.W_enc[1] = torch.tensor(
            [[-0.2, 0.6, 0.3, -0.1], [0.3, -0.1, 0.5, 0.2], [-0.1, 0.2, -0.3, 0.4]]
        )

        # W_dec: n_layers x d_transcoder x n_layers x d_model (2 x 4 x 2 x 3)
        transcoder.W_dec[0] = torch.tensor(
            [
                [[0.4, 0.2, -0.1], [0.3, 0.1, -0.2]],
                [[-0.1, 0.3, 0.2], [0.2, -0.2, 0.3]],
                [[0.2, -0.4, 0.1], [0.1, 0.3, -0.1]],
                [[0.3, 0.1, -0.2], [-0.2, 0.4, 0.2]],
            ]
        )
        transcoder.W_dec[1] = torch.tensor(
            [
                [[-0.3, 0.1, 0.2], [0.4, -0.1, 0.1]],
                [[0.2, -0.2, 0.3], [-0.1, 0.2, -0.3]],
                [[0.1, 0.3, -0.1], [0.3, -0.2, 0.2]],
                [[-0.2, 0.4, 0.2], [0.1, -0.1, 0.4]],
            ]
        )

        # b_enc: n_layers x d_transcoder (2 x 4)
        transcoder.b_enc[0] = torch.tensor([0.1, -0.2, 0.05, 0.15])
        transcoder.b_enc[1] = torch.tensor([-0.1, 0.15, -0.05, 0.2])

        # b_dec: n_layers x d_model (2 x 3)
        transcoder.b_dec[0] = torch.tensor([0.02, -0.03, 0.01])
        transcoder.b_dec[1] = torch.tensor([-0.01, 0.04, -0.02])

        # b_in: n_layers x d_model (2 x 3)
        transcoder.b_in[0] = torch.tensor([0.1, 0.05, -0.02])
        transcoder.b_in[1] = torch.tensor([-0.05, 0.1, 0.03])

    # Test input: n_layers x d_model (2 x 3)
    input_acts = torch.tensor([[1.0, 0.5, -0.2], [-0.3, 0.8, 0.4]])

    result = transcoder.forward(input_acts)

    expected = torch.tensor([[0.2496, 0.3538, -0.0061], [0.2148, 0.1897, 0.1845]])
    assert torch.allclose(result, expected, atol=1e-4)


def test_CrossLayerTranscoder_forward_with_relu_with_skip():
    """Test CrossLayerTranscoder forward pass with ReLU activation and skip connection."""
    d_model = 3
    d_transcoder = 4
    layers = [0, 1]

    transcoder = CrossLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=nn.ReLU(),
        layers=layers,
        skip_connection=True,
    )

    # Set hardcoded parameters
    with torch.no_grad():
        # W_enc: 2 x 3 x 4
        transcoder.W_enc[0] = torch.tensor(
            [[0.3, 0.4, -0.2, 0.1], [0.2, -0.1, 0.3, 0.5], [-0.1, 0.2, 0.4, -0.3]]
        )
        transcoder.W_enc[1] = torch.tensor(
            [[0.4, -0.2, 0.3, 0.2], [-0.1, 0.3, -0.2, 0.4], [0.2, 0.1, -0.3, 0.3]]
        )

        # W_dec: 2 x 4 x 2 x 3
        transcoder.W_dec[0] = torch.tensor(
            [
                [[0.5, 0.2, -0.1], [0.4, -0.1, 0.2]],
                [[0.1, 0.6, 0.3], [-0.2, 0.3, 0.1]],
                [[-0.2, 0.3, 0.4], [0.1, -0.3, 0.5]],
                [[0.4, -0.1, 0.2], [0.3, -0.1, 0.2]],
            ]
        )
        transcoder.W_dec[1] = torch.tensor(
            [
                [[0.3, -0.1, 0.2], [0.2, 0.4, -0.2]],
                [[0.2, 0.4, -0.2], [0.1, -0.3, 0.5]],
                [[0.1, -0.3, 0.5], [-0.2, 0.3, 0.1]],
                [[-0.2, 0.3, 0.1], [0.5, 0.2, -0.1]],
            ]
        )

        # W_skip: 2 x 3 x 3
        transcoder.W_skip[0] = torch.tensor(
            [[0.8, 0.1, -0.05], [0.05, 0.9, 0.1], [-0.02, 0.03, 0.85]]
        )
        transcoder.W_skip[1] = torch.tensor(
            [[0.7, 0.2, 0.03], [0.1, 0.8, -0.05], [0.04, -0.01, 0.9]]
        )

        # b_enc: 2 x 4
        transcoder.b_enc[0] = torch.tensor([0.1, -0.05, 0.08, -0.03])
        transcoder.b_enc[1] = torch.tensor([-0.02, 0.04, -0.06, 0.05])

        # b_dec: 2 x 3
        transcoder.b_dec[0] = torch.tensor([0.02, 0.03, -0.01])
        transcoder.b_dec[1] = torch.tensor([-0.01, 0.02, 0.03])

        # b_in: 2 x 3
        transcoder.b_in[0] = torch.tensor([0.1, 0.2, -0.05])
        transcoder.b_in[1] = torch.tensor([0.05, -0.1, 0.08])

    # Test input: 2 x 3
    input_acts = torch.tensor([[0.8, 0.5, -0.3], [0.4, 0.7, 0.2]])

    result = transcoder.forward(input_acts)

    expected = torch.tensor([[0.8869, 0.7302, -0.1530], [0.8003, 0.7620, 0.3061]])
    assert torch.allclose(result, expected, atol=1e-4)


def test_CrossLayerTranscoder_forward_with_jumprelu_no_skip():
    """Test CrossLayerTranscoder forward pass with JumpReLU activation and no skip connection."""
    d_model = 3
    d_transcoder = 4
    layers = [0, 1]

    # threshold shape: n_layers x d_transcoder = 2 x 4
    threshold = torch.tensor([[0.2, 0.15, 0.25, 0.1], [0.18, 0.12, 0.22, 0.14]])
    jump_relu = JumpReLU(threshold=threshold, bandwidth=1.0)
    transcoder = CrossLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=jump_relu,
        layers=layers,
        skip_connection=False,
    )

    # Set hardcoded parameters
    with torch.no_grad():
        # W_enc: 2 x 3 x 4
        transcoder.W_enc[0] = torch.tensor(
            [[0.4, 0.3, -0.2, 0.1], [-0.2, 0.5, 0.3, -0.1], [0.1, -0.3, 0.4, 0.2]]
        )
        transcoder.W_enc[1] = torch.tensor(
            [[-0.3, 0.2, 0.4, -0.1], [0.3, -0.2, 0.1, 0.4], [-0.1, 0.4, -0.2, 0.3]]
        )

        # W_dec: 2 x 4 x 2 x 3
        transcoder.W_dec[0] = torch.tensor(
            [
                [[0.6, -0.1, 0.2], [0.3, -0.2, 0.1]],
                [[0.2, 0.4, -0.3], [0.2, 0.4, -0.3]],
                [[-0.1, 0.3, 0.4], [0.1, -0.1, 0.3]],
                [[0.3, -0.2, 0.1], [0.4, 0.2, -0.1]],
            ]
        )
        transcoder.W_dec[1] = torch.tensor(
            [
                [[0.4, 0.2, -0.1], [0.6, -0.1, 0.2]],
                [[-0.2, 0.3, 0.2], [0.2, 0.4, -0.3]],
                [[0.1, -0.1, 0.3], [-0.1, 0.3, 0.4]],
                [[0.2, 0.4, -0.3], [0.3, -0.2, 0.1]],
            ]
        )

        # b_enc: 2 x 4
        transcoder.b_enc[0] = torch.tensor([0.05, 0.1, -0.03, 0.08])
        transcoder.b_enc[1] = torch.tensor([-0.04, 0.06, 0.02, -0.05])

        # b_dec: 2 x 3
        transcoder.b_dec[0] = torch.tensor([-0.02, 0.01, 0.03])
        transcoder.b_dec[1] = torch.tensor([0.01, -0.03, 0.02])

        # b_in: 2 x 3
        transcoder.b_in[0] = torch.tensor([0.0, 0.1, -0.05])
        transcoder.b_in[1] = torch.tensor([0.02, -0.08, 0.04])

    # Test input: 2 x 3
    input_acts = torch.tensor([[0.6, 0.4, -0.2], [-0.1, 0.8, 0.3]])

    result = transcoder.forward(input_acts)

    expected = torch.tensor([[0.3760, 0.3821, -0.2105], [0.4275, 0.0152, -0.0150]])
    assert torch.allclose(result, expected, atol=1e-4)


def test_CrossLayerTranscoder_forward_with_jumprelu_with_skip():
    """Test CrossLayerTranscoder forward pass with JumpReLU activation and skip connection."""
    d_model = 3
    d_transcoder = 4
    layers = [0, 1]

    # threshold shape: n_layers x d_transcoder = 2 x 4
    threshold = torch.tensor([[0.18, 0.12, 0.15, 0.20], [0.16, 0.14, 0.13, 0.17]])
    jump_relu = JumpReLU(threshold=threshold, bandwidth=1.0)
    transcoder = CrossLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=jump_relu,
        layers=layers,
        skip_connection=True,
    )

    # Set hardcoded parameters
    with torch.no_grad():
        # W_enc: 2 x 3 x 4
        transcoder.W_enc[0] = torch.tensor(
            [[0.2, 0.3, -0.1, 0.4], [0.4, -0.1, 0.2, 0.3], [-0.2, 0.1, 0.3, -0.1]]
        )
        transcoder.W_enc[1] = torch.tensor(
            [[-0.1, 0.5, 0.2, -0.3], [0.3, 0.2, -0.1, 0.4], [0.1, -0.2, 0.4, 0.2]]
        )

        # W_dec: 2 x 4 x 2 x 3
        transcoder.W_dec[0] = torch.tensor(
            [
                [[0.5, 0.1, -0.2], [0.3, -0.1, 0.4]],
                [[0.2, 0.4, 0.1], [-0.2, 0.3, 0.2]],
                [[-0.1, 0.3, 0.2], [0.2, -0.1, 0.3]],
                [[0.3, -0.1, 0.4], [0.1, 0.6, -0.1]],
            ]
        )
        transcoder.W_dec[1] = torch.tensor(
            [
                [[0.3, -0.2, 0.1], [0.5, 0.1, -0.2]],
                [[0.1, 0.6, -0.1], [0.2, 0.4, 0.1]],
                [[0.2, -0.1, 0.3], [-0.1, 0.3, 0.2]],
                [[-0.2, 0.3, 0.2], [0.3, -0.1, 0.4]],
            ]
        )

        # W_skip: 2 x 3 x 3
        transcoder.W_skip[0] = torch.tensor(
            [[0.9, 0.05, -0.02], [0.02, 0.85, 0.1], [-0.01, 0.03, 0.8]]
        )
        transcoder.W_skip[1] = torch.tensor(
            [[0.8, 0.1, 0.02], [0.15, 0.7, -0.05], [0.05, -0.02, 0.75]]
        )

        # b_enc: 2 x 4
        transcoder.b_enc[0] = torch.tensor([0.02, 0.03, -0.01, 0.05])
        transcoder.b_enc[1] = torch.tensor([-0.01, 0.05, 0.02, -0.03])

        # b_dec: 2 x 3
        transcoder.b_dec[0] = torch.tensor([0.01, -0.02, 0.03])
        transcoder.b_dec[1] = torch.tensor([0.02, 0.01, -0.01])

        # b_in: 2 x 3
        transcoder.b_in[0] = torch.tensor([0.05, 0.1, -0.02])
        transcoder.b_in[1] = torch.tensor([0.1, 0.05, 0.03])

    # Test input: 2 x 3
    input_acts = torch.tensor([[0.5, 0.3, -0.1], [0.2, 0.6, 0.4]])

    result = transcoder.forward(input_acts)

    expected = torch.tensor([[0.6740, 0.2362, 0.1736], [0.4131, 0.6316, 0.4058]])
    assert torch.allclose(result, expected, atol=1e-4)


def test_CrossLayerTranscoder_encode_without_activation():
    """Test CrossLayerTranscoder encode method without applying activation function."""
    d_model = 3
    d_transcoder = 4
    layers = [0, 1]

    transcoder = CrossLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=nn.ReLU(),
        layers=layers,
        skip_connection=False,
    )

    # Set hardcoded parameters
    with torch.no_grad():
        transcoder.W_enc[0] = torch.tensor(
            [[0.5, 0.3, -0.2, 0.1], [0.2, 0.4, 0.3, -0.1], [-0.1, 0.2, 0.5, 0.3]]
        )
        transcoder.W_enc[1] = torch.tensor(
            [[0.3, -0.4, 0.2, 0.2], [-0.2, 0.3, -0.1, 0.4], [0.1, -0.2, 0.4, -0.3]]
        )
        transcoder.b_enc[0] = torch.tensor([0.1, 0.05, -0.03, 0.08])
        transcoder.b_enc[1] = torch.tensor([-0.05, 0.07, 0.02, -0.04])
        transcoder.b_in[0] = torch.tensor([0.1, 0.2, -0.05])
        transcoder.b_in[1] = torch.tensor([0.03, -0.06, 0.02])

    input_acts = torch.tensor([[0.8, 0.6, -0.3], [0.5, 0.3, 0.1]])

    # Test without activation function
    result = transcoder.encode(input_acts, apply_activation_function=False)

    expected = torch.tensor([[0.5550, 0.3700, -0.1750, 0.0350], [0.0270, -0.0260, 0.1100, 0.1740]])
    assert torch.allclose(result, expected, atol=1e-4)

    # Test with activation function (should apply ReLU)
    result_with_activation = transcoder.encode(input_acts, apply_activation_function=True)
    expected_with_relu = torch.tensor(
        [[0.5550, 0.3700, 0.0000, 0.0350], [0.0270, 0.0000, 0.1100, 0.1740]]
    )  # Negative values become 0 after ReLU
    assert torch.allclose(result_with_activation, expected_with_relu, atol=1e-4)


def test_CrossLayerTranscoder_compute_skip_raises_error_when_no_skip_connection():
    """Test that compute_skip raises ValueError when transcoder has no skip connection."""
    transcoder = CrossLayerTranscoder(
        d_model=3, d_transcoder=4, activation_function=nn.ReLU(), layers=[0], skip_connection=False
    )

    input_acts = torch.tensor([[1.0, 0.5, -0.2]])

    with pytest.raises(ValueError, match="Transcoder has no skip connection"):
        transcoder.compute_skip(input_acts)


def test_CrossLayerTranscoder_n_layers_property():
    """Test that n_layers property returns correct number of layers."""
    layers = [0, 1, 2]
    transcoder = CrossLayerTranscoder(
        d_model=3,
        d_transcoder=4,
        activation_function=nn.ReLU(),
        layers=layers,
        skip_connection=False,
    )

    assert transcoder.n_layers == 3
    assert transcoder.n_layers == len(layers)
