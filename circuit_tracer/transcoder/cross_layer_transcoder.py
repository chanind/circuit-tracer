from typing import Callable, Optional

import torch
from torch import nn


class CrossLayerTranscoder(nn.Module):
    d_model: int
    d_transcoder: int
    layers: list[int]
    W_enc: nn.Parameter  # shape n_layers x d_model x d_transcoder
    W_dec: nn.Parameter  # shape n_layers x d_transcoder x n_layers x d_model
    b_in: nn.Parameter  # shape n_layers x d_model
    b_enc: nn.Parameter  # shape n_layers x d_transcoder
    b_dec: nn.Parameter  # shape n_layers x d_model
    W_skip: Optional[nn.Parameter]  # shape n_layers x d_model x d_model
    activation_function: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function: Callable[[torch.Tensor], torch.Tensor],
        layers: list[int],
        skip_connection: bool = False,
    ):
        """Cross layer transcoder implementation

        Args:
            d_model (int): The dimension of the model.
            d_transcoder (int): The dimension of the transcoder.
            activation_function (nn.Module): The activation function.
            layers (int): list of layers in order where the
            skip_connection (bool): Whether there is a skip connection
        """
        super().__init__()

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layers = layers
        n_layers = len(layers)

        self.W_enc = nn.Parameter(torch.zeros(n_layers, d_model, d_transcoder))
        self.W_dec = nn.Parameter(torch.zeros(n_layers, d_transcoder, n_layers, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_layers, d_transcoder))
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_model))
        self.b_in = nn.Parameter(torch.zeros(n_layers, d_model))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(n_layers, d_model, d_model))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    @property
    def n_layers(self):
        return len(self.layers)

    def encode(self, input_acts: torch.Tensor, apply_activation_function: bool = True):
        # Use einsum for element-wise processing of each layer
        input_centered = input_acts.to(self.W_enc.dtype) - self.b_in
        pre_acts = torch.einsum("ij,ijk->ik", input_centered, self.W_enc) + self.b_enc
        if not apply_activation_function:
            return pre_acts
        acts = self.activation_function(pre_acts)
        return acts

    def decode(self, acts):
        # Use einsum for element-wise processing of each layer
        return torch.einsum("ij,ijkl->kl", acts, self.W_dec) + self.b_dec

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            # Use einsum for element-wise processing of each layer
            input_centered = input_acts - self.b_in
            return torch.einsum("ij,ijk->ik", input_centered, self.W_skip)
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts):
        transcoder_acts = self.encode(input_acts)
        decoded = self.decode(transcoder_acts)
        decoded = decoded.detach()
        decoded.requires_grad = True

        if self.W_skip is not None:
            skip = self.compute_skip(input_acts)
            decoded = decoded + skip

        return decoded
