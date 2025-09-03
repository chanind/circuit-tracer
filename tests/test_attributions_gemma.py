from functools import partial

import pytest
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, Gemma2Config, Gemma2ForCausalLM

from circuit_tracer import Graph, ReplacementModel, attribute
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils import get_default_device


def verify_token_and_error_edges(
    model: ReplacementModel,
    graph: Graph,
    act_atol=1e-3,
    act_rtol=1e-3,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(get_default_device())
    active_features = graph.active_features.to(get_default_device())
    logit_tokens = graph.logit_tokens.to(get_default_device())
    total_active_features = active_features.size(0)
    pos_start = 1  # ignore first token (BOS)

    ctx = model.setup_attribution(s)

    error_vectors = ctx.error_vectors
    token_vectors = ctx.token_vectors

    logits, activation_cache = model.get_activations(s, apply_activation_function=False)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    _, freeze_hooks = model.setup_intervention_with_freeze(
        s, constrained_layers=range(model.cfg.n_layers)
    )

    def verify_intervention(expected_effects, intervention):
        new_activation_cache, activation_hooks = model._get_activation_caching_hooks(
            apply_activation_function=False
        )

        fwd_hooks = [*freeze_hooks, intervention, *activation_hooks]
        new_logits = model.run_with_hooks(s, fwd_hooks=fwd_hooks)
        new_logits = new_logits.squeeze(0)

        new_activation_cache = torch.stack(new_activation_cache)
        new_relevant_activations = new_activation_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    def hook_error_intervention(activations, hook, layer: int, pos: int):
        steering_vector = torch.zeros_like(activations)
        steering_vector[:, pos] += error_vectors[layer, pos]
        return activations + steering_vector

    for error_node_layer in range(error_vectors.size(0)):
        for error_node_pos in range(pos_start, error_vectors.size(1)):
            error_node_index = error_node_layer * error_vectors.size(1) + error_node_pos
            expected_effects = adjacency_matrix[:, total_active_features + error_node_index]
            intervention = (
                f"blocks.{error_node_layer}.{model.feature_output_hook}",
                partial(hook_error_intervention, layer=error_node_layer, pos=error_node_pos),
            )
            verify_intervention(expected_effects, intervention)

    def hook_token_intervention(activations, hook, pos: int):
        steering_vector = torch.zeros_like(activations)
        steering_vector[:, pos] += token_vectors[pos]
        return activations + steering_vector

    total_error_nodes = error_vectors.size(0) * error_vectors.size(1)
    for token_pos in range(pos_start, token_vectors.size(0)):
        expected_effects = adjacency_matrix[
            :, total_active_features + total_error_nodes + token_pos
        ]
        intervention = ("hook_embed", partial(hook_token_intervention, pos=token_pos))
        verify_intervention(expected_effects, intervention)


def verify_feature_edges(
    model: ReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=5e-4,
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    s = graph.input_tokens
    adjacency_matrix = graph.adjacency_matrix.to(get_default_device())
    active_features = graph.active_features.to(get_default_device())
    logit_tokens = graph.logit_tokens.to(get_default_device())
    total_active_features = active_features.size(0)

    logits, activation_cache = model.get_activations(s, apply_activation_function=False)
    logits = logits.squeeze(0)

    relevant_activations = activation_cache[
        active_features[:, 0], active_features[:, 1], active_features[:, 2]
    ]
    relevant_logits = logits[-1, logit_tokens]
    demeaned_relevant_logits = relevant_logits - logits[-1].mean()

    def verify_intervention(
        expected_effects, layer: int, pos: int, feature_idx: int, new_activation
    ):
        new_logits, new_activation_cache = model.feature_intervention(
            s,
            [(layer, pos, feature_idx, new_activation)],
            constrained_layers=range(model.cfg.n_layers),
            apply_activation_function=False,
        )
        new_logits = new_logits.squeeze(0)

        new_relevant_activations = new_activation_cache[
            active_features[:, 0], active_features[:, 1], active_features[:, 2]
        ]
        new_relevant_logits = new_logits[-1, logit_tokens]
        new_demeaned_relevant_logits = new_relevant_logits - new_logits[-1].mean()

        expected_activation_difference = expected_effects[:total_active_features]
        expected_logit_difference = expected_effects[-len(logit_tokens) :]

        assert torch.allclose(
            new_relevant_activations,
            relevant_activations + expected_activation_difference,
            atol=act_atol,
            rtol=act_rtol,
        )
        assert torch.allclose(
            new_demeaned_relevant_logits,
            demeaned_relevant_logits + expected_logit_difference,
            atol=logit_atol,
            rtol=logit_rtol,
        )

    random_order = torch.randperm(active_features.size(0))
    chosen_nodes = random_order[:n_samples]
    for chosen_node in tqdm(chosen_nodes):
        layer, pos, feature_idx = active_features[chosen_node].tolist()
        old_activation = activation_cache[layer, pos, feature_idx]
        new_activation = old_activation * 2
        expected_effects = adjacency_matrix[:, chosen_node]
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)


def load_dummy_gemma_model(cfg: Gemma2Config):
    transcoders = {
        layer_idx: SingleLayerTranscoder(
            cfg.hidden_size, cfg.hidden_size * 4, JumpReLU(torch.tensor(0.0), 0.1), layer_idx
        )
        for layer_idx in range(cfg.num_hidden_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)

    transcoder_set = TranscoderSet(
        transcoders, feature_input_hook="mlp.hook_in", feature_output_hook="mlp.hook_out"
    )

    hf_model = Gemma2ForCausalLM(cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # to avoid gated repos
    model = ReplacementModel.from_hf_model(hf_model, tokenizer, transcoder_set)

    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    for _, param in model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)

    assert isinstance(model.transcoders, TranscoderSet)
    for transcoder in model.transcoders:
        assert isinstance(transcoder.activation_function, JumpReLU)
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)

    return model


def verify_small_gemma_model(s: torch.Tensor):
    # Create a small Gemma2 config for testing
    gemma_small_cfg = Gemma2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=0.0,
        sliding_window=4096,
    )
    model = load_dummy_gemma_model(gemma_small_cfg)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


def verify_large_gemma_model(s: torch.Tensor):
    # Create a larger Gemma2 config for testing
    gemma_large_cfg = Gemma2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=32,
        hidden_act="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        attn_logit_softcapping=50.0,
        final_logit_softcapping=0.0,
        sliding_window=4096,
    )
    model = load_dummy_gemma_model(gemma_large_cfg)
    graph = attribute(s, model)

    verify_token_and_error_edges(model, graph)
    verify_feature_edges(model, graph)


def verify_gemma_2_2b(s: str):
    model = ReplacementModel.boot_transformers("google/gemma-2-2b", "gemma")
    graph = attribute(s, model)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


def verify_gemma_2_2b_clt(s: str):
    model = ReplacementModel.boot_transformers("google/gemma-2-2b", "mntss/clt-gemma-2-2b-426k")
    graph = attribute(s, model)

    print("Changing logit softcap to 0, as the logits will otherwise be off.")
    with model.zero_softcap():
        verify_token_and_error_edges(model, graph)
        verify_feature_edges(model, graph)


def test_small_gemma_model():
    s = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
    verify_small_gemma_model(s)


def test_large_gemma_model():
    s = torch.tensor([0, 113, 24, 53, 27])
    verify_large_gemma_model(s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemma_2_2b():
    s = "The National Digital Analytics Group (ND"
    verify_gemma_2_2b(s)


# def test_gemma_2_2b_clt():
#     s = "The National Digital Analytics Group (ND"
#     verify_gemma_2_2b_clt(s)


if __name__ == "__main__":
    torch.manual_seed(42)
    test_small_gemma_model()
    test_large_gemma_model()
    test_gemma_2_2b()
    # test_gemma_2_2b_clt()  # This will pass, but is slow to run
