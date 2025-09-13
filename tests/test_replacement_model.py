import pytest
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoTokenizer, GPT2LMHeadModel

from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from tests._comparison.replacement_model import ReplacementModel as LegacyReplacementModel


@pytest.fixture
def gpt2_transcoder_set() -> TranscoderSet:
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
    gpt2_transcoder_set: TranscoderSet,
):
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", gpt2_transcoder_set, device="cpu"
    )
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, gpt2_transcoder_set, device=torch.device("cpu")
    )

    test_inputs = hf_tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_model.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_model.run_with_cache(test_inputs)

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore

    for layer in range(12):
        assert torch.allclose(
            legacy_cache[f"blocks.{layer}.hook_resid_mid"],
            bridge_cache[f"blocks.{layer}.hook_resid_mid"],
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

    def assert_hook_close(legacy_cache, bridge_cache, hook):
        assert torch.allclose(
            legacy_cache[hook],
            bridge_cache[hook],
            atol=1e-2,
            rtol=1e-2,
        ), f"{hook} is not close"

    assert torch.allclose(legacy_logits, bridge_logits)  # type: ignore
    for layer in range(12):
        assert_hook_close(legacy_cache, bridge_cache, f"blocks.{layer}.hook_resid_mid")
