import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from circuit_tracer import Graph, ReplacementModel, attribute
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder
from circuit_tracer.utils import get_default_device


def create_clt_model(cfg: GPT2Config):
    """Create a CLT and ReplacementModel with random weights."""
    # Create CLT with 4x expansion
    clt = CrossLayerTranscoder(
        n_layers=cfg.n_layer,
        d_transcoder=cfg.n_embd * 4,
        d_model=cfg.n_embd,
        dtype=torch.float32,
        lazy_decoder=False,
    )

    # Initialize CLT weights
    with torch.no_grad():
        for param in clt.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)

    # Create model
    hf_model = GPT2LMHeadModel(cfg)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = ReplacementModel.from_hf_model(hf_model, tokenizer, clt)

    # Monkey patch all_special_ids if necessary
    type(model.tokenizer).all_special_ids = property(lambda self: [0])  # type: ignore

    # Initialize model weights
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                nn.init.uniform_(param, a=-0.1, b=0.1)

    return model


def verify_feature_edges(
    model: ReplacementModel,
    graph: Graph,
    n_samples: int = 100,
    act_atol=5e-4,
    act_rtol=1e-5,
    logit_atol=1e-5,
    logit_rtol=1e-3,
):
    """Verify that feature interventions produce the expected effects using feature_intervention
    method."""
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
        expected_effects,
        layer: int | torch.Tensor,
        pos: int | torch.Tensor,
        feature_idx: int | torch.Tensor,
        new_activation,
    ):
        new_logits, new_activation_cache = model.feature_intervention(
            s,
            [(layer, pos, feature_idx, new_activation)],
            constrained_layers=range(model.cfg.n_layer),
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
        layer, pos, feature_idx = active_features[chosen_node]
        old_activation = activation_cache[layer, pos, feature_idx]
        new_activation = old_activation * 2
        expected_effects = adjacency_matrix[:, chosen_node]
        verify_intervention(expected_effects, layer, pos, feature_idx, new_activation)


def test_clt_attribution():
    """Test CLT attribution and intervention mechanism."""
    # Minimal config
    cfg = GPT2Config(
        n_layer=4,
        n_embd=8,
        n_ctx=32,
        n_head=2,
        intermediate_size=32,
        activation_function="gelu",
        vocab_size=50,
        model_type="gpt2",
    )

    # Create model
    model = create_clt_model(cfg)

    # Run attribution
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    graph = attribute(prompt, model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32)

    # Test feature interventions
    n_active = len(graph.active_features)
    n_samples = min(100, n_active)

    verify_feature_edges(model, graph, n_samples=n_samples)


if __name__ == "__main__":
    test_clt_attribution()
