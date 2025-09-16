import pytest
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer, GPT2LMHeadModel

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from tests._comparison.attribution.attribute import attribute as legacy_attribute
from tests._comparison.replacement_model import ReplacementModel as LegacyReplacementModel


@pytest.fixture
def gpt2_transcoder_set_pair() -> tuple[TranscoderSet, TranscoderSet]:
    # return separate objects to avoid sharing hooks and stuff
    transcoder_set1 = get_gpt2_transcoder_set()
    transcoder_set2 = get_gpt2_transcoder_set()
    with torch.no_grad():
        for param1, param2 in zip(transcoder_set1.parameters(), transcoder_set2.parameters()):
            assert param1.shape == param2.shape
            data = torch.randn_like(param1)
            param1.data = data
            param2.data = data
    return transcoder_set1, transcoder_set2


def get_gpt2_transcoder_set() -> TranscoderSet:
    return TranscoderSet(
        transcoders={
            i: SingleLayerTranscoder(
                d_model=768,
                d_transcoder=128,
                layer_idx=i,
                activation_function=F.relu,
            )
            for i in range(12)
        },
        feature_input_hook="hook_resid_mid",
        feature_output_hook="hook_mlp_out",
    )


def test_bridge_gpt2_replacement_model_behaves_like_legacy_replacement_model(
    gpt2_transcoder_set_pair: tuple[TranscoderSet, TranscoderSet],
):
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", gpt2_transcoder_set_pair[0], device="cpu"
    )
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, gpt2_transcoder_set_pair[1], device=torch.device("cpu")
    )

    test_inputs = hf_tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_model.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_model.run_with_cache(test_inputs)

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore

    for layer in range(12):
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.hook_resid_mid"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.ln2.hook_in"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_mlp_out"],
            bridge_cache[f"blocks.{layer}.hook_mlp_out"],
            atol=1e-4,
            rtol=1e-4,
        )
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_mlp_out"],
            bridge_cache[f"blocks.{layer}.mlp.hook_out"],
            atol=1e-4,
            rtol=1e-4,
        )

    bridge_ctx = bridge_model.setup_attribution(test_inputs)
    legacy_ctx = legacy_model.setup_attribution(test_inputs)
    assert torch.allclose(
        bridge_ctx.activation_matrix.to_dense(),
        legacy_ctx.activation_matrix.to_dense(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.allclose(bridge_ctx.error_vectors, legacy_ctx.error_vectors, atol=1e-2, rtol=1e-2)
    assert torch.allclose(bridge_ctx.token_vectors, legacy_ctx.token_vectors, atol=1e-3, rtol=1e-3)
    assert torch.allclose(bridge_ctx.decoder_vecs, legacy_ctx.decoder_vecs, atol=1e-3, rtol=1e-3)
    assert torch.allclose(bridge_ctx.encoder_vecs, legacy_ctx.encoder_vecs, atol=1e-3, rtol=1e-3)
    assert torch.allclose(
        bridge_ctx.encoder_to_decoder_map, legacy_ctx.encoder_to_decoder_map, atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        bridge_ctx.decoder_locations, legacy_ctx.decoder_locations, atol=1e-3, rtol=1e-3
    )


def test_bridge_attribute_behaves_like_legacy_attribute(
    gpt2_transcoder_set_pair: tuple[TranscoderSet, TranscoderSet],
):
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, gpt2_transcoder_set_pair[0], device=torch.device("cpu")
    )
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", gpt2_transcoder_set_pair[1], device="cpu"
    )

    prompt = torch.tensor([[0, 1, 2, 3, 4, 5]])
    bridge_graph = attribute(
        prompt, bridge_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )
    legacy_graph = legacy_attribute(
        prompt, legacy_model, max_n_logits=5, desired_logit_prob=0.8, batch_size=32
    )

    assert bridge_graph.input_string == legacy_graph.input_string
    assert torch.allclose(bridge_graph.input_tokens, legacy_graph.input_tokens)
    assert torch.allclose(bridge_graph.logit_tokens, legacy_graph.logit_tokens)
    assert torch.allclose(
        bridge_graph.logit_probabilities, legacy_graph.logit_probabilities, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(bridge_graph.active_features, legacy_graph.active_features)
    assert torch.allclose(
        bridge_graph.activation_values, legacy_graph.activation_values, atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(bridge_graph.selected_features, legacy_graph.selected_features)
    assert bridge_graph.scan == legacy_graph.scan
    assert torch.allclose(
        bridge_graph.adjacency_matrix,
        legacy_graph.adjacency_matrix,
        atol=1e-2,
        rtol=1e-2,
    )


def test_TransformerBridge_gpt2_behaves_like_HookedTransformer_gpt2():
    """
    This isn't actually a test of the ReplacementModel, but if this fails, we have no hope.
    """
    legacy_gpt2 = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_gpt2: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_gpt2.enable_compatibility_mode(no_processing=True)

    test_inputs = bridge_gpt2.tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_gpt2.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_gpt2.run_with_cache(test_inputs)

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore
    for layer in range(12):
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.hook_resid_mid"],
            atol=1e-2,
            rtol=1e-2,
        )
