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


@pytest.fixture
def replacement_model_pair(
    gpt2_transcoder_set_pair: tuple[TranscoderSet, TranscoderSet],
) -> tuple[ReplacementModel, LegacyReplacementModel]:
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="cpu")
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    legacy_model = LegacyReplacementModel.from_pretrained_and_transcoders(
        "gpt2", gpt2_transcoder_set_pair[0], device="cpu"
    )
    bridge_model = ReplacementModel.from_hf_model(
        hf_model, hf_tokenizer, gpt2_transcoder_set_pair[1], device=torch.device("cpu")
    )
    return (bridge_model, legacy_model)


def test_bridge_gpt2_replacement_model_behaves_like_legacy_replacement_model(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    test_inputs = bridge_model.tokenizer.encode("Hello, world!", return_tensors="pt")

    legacy_logits, legacy_cache = legacy_model.run_with_cache(test_inputs)
    bridge_logits, bridge_cache = bridge_model.run_with_cache(test_inputs)  # type: ignore

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

    bridge_ctx = bridge_model.setup_attribution(test_inputs)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(test_inputs)  # type: ignore
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


def test_TransformerBridge_backward_gradients_differ_from_HookedTransformer():
    """
    This is a bug in TransformerLens where backward hooks see different gradient values
    in TransformerBridge vs HookedTransformer, even though forward passes are identical.
    Until this test passes, we will struggle to get the bridge code to work.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Collect gradient sums from backward hooks
    hooked_grad_sum = torch.zeros(1)
    bridge_grad_sum = torch.zeros(1)

    def sum_hooked_grads(grad, hook=None):
        nonlocal hooked_grad_sum
        hooked_grad_sum = grad.sum()
        return None

    def sum_bridge_grads(grad, hook=None):
        nonlocal bridge_grad_sum
        bridge_grad_sum = grad.sum()
        return None

    # Run with HookedTransformer
    hooked_model.zero_grad()
    with hooked_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_hooked_grads)]):
        hooked_out = hooked_model(test_input)
        hooked_out.sum().backward()

    # Run with TransformerBridge
    bridge_model.zero_grad()
    with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", sum_bridge_grads)]):
        bridge_out = bridge_model(test_input)
        bridge_out.sum().backward()

    assert torch.allclose(hooked_out, bridge_out, atol=1e-2, rtol=1e-2), (
        f"Output differs by {abs(hooked_out - bridge_out).item():.6f}"
    )
    # This assertion demonstrates the bug - gradient values differ
    assert torch.allclose(hooked_grad_sum, bridge_grad_sum, atol=1e-2, rtol=1e-2), (
        f"Gradient sums differ by {abs(hooked_grad_sum - bridge_grad_sum).item():.6f}"
    )


def test_TransformerBridge_run_with_cache_vs_forward():
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])
    bridge_logits_cache, _ = bridge_model.run_with_cache(test_input)
    bridge_logits_manual = bridge_model(test_input)

    assert torch.allclose(bridge_logits_cache, bridge_logits_manual, atol=1e-2)


def test_TransformerBridge_run_with_cache():
    """
    This is a bug in TransformerLens where TransformerBridge.run_with_cache() returns
    incorrect cached activation values, even though manual hooks work correctly.
    The issue only occurs when caching all hooks - using names_filter works correctly.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    test_input = torch.tensor([[1, 2, 3]])

    # Method 1: run_with_cache (buggy for TransformerBridge)
    _, hooked_cache = hooked_model.run_with_cache(test_input)
    _, bridge_cache = bridge_model.run_with_cache(test_input)

    # Method 2: Manual hooks (correct for both models)
    manual_cache = {}

    def make_cache_hook(name):
        def hook_fn(acts, hook):
            manual_cache[name] = acts.clone()
            return acts

        return hook_fn

    hooked_model.reset_hooks()
    with hooked_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("hooked"))]):
        hooked_model(test_input)

    bridge_model.reset_hooks()
    with bridge_model.hooks(fwd_hooks=[("blocks.0.hook_mlp_out", make_cache_hook("bridge"))]):
        bridge_model(test_input)

    # Demonstrate that using names_filter DOES work correctly
    _, bridge_cache_filtered = bridge_model.run_with_cache(
        test_input, names_filter=lambda name: name == "blocks.0.hook_mlp_out"
    )

    # the manual values match the hooked values
    assert torch.allclose(manual_cache["hooked"], manual_cache["bridge"], atol=1e-4)

    assert torch.allclose(
        bridge_cache_filtered["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-4
    )

    # Verify cache values match manual hooks for HookedTransformer
    assert torch.allclose(hooked_cache["blocks.0.hook_mlp_out"], manual_cache["hooked"], atol=1e-5)

    # This assertion demonstrates the bug - TransformerBridge run_with_cache gives wrong values
    assert torch.allclose(
        bridge_cache["blocks.0.hook_mlp_out"], manual_cache["bridge"], atol=1e-2, rtol=1e-2
    ), (
        f"TransformerBridge run_with_cache gives incorrect cached values! "
        f"Cache sum: {bridge_cache['blocks.0.hook_mlp_out'].sum():.6f}, "
        f"Manual hooks sum: {manual_cache['bridge'].sum():.6f}, "
        f"Diff: {(bridge_cache['blocks.0.hook_mlp_out'] - manual_cache['bridge']).abs().max():.6f}"
    )


def test_TransformerBridge_hooks_ignores_backward_hooks():
    """Minimal test demonstrating that TransformerBridge.hooks() doesn't register backward hooks.

    This is a bug in TransformerLens where TransformerBridge.hooks() accepts bwd_hooks
    but doesn't actually register them, while HookedTransformer.hooks() does.
    """
    # Create both models with the same configuration
    hooked_model = HookedTransformer.from_pretrained_no_processing("gpt2", device_map="cpu")
    bridge_model: TransformerBridge = TransformerBridge.boot_transformers("gpt2", device="cpu")  # type: ignore
    bridge_model.enable_compatibility_mode(no_processing=True)

    # Create a simple backward hook that tracks if it was called
    hook_called = {"hooked": False, "bridge": False}

    def make_test_hook(model_type):
        def hook_fn(grad, hook=None):
            hook_called[model_type] = True
            # For HookedTransformer, the hook doesn't modify the gradient
            return None

        return hook_fn

    # Test input
    test_input = torch.tensor([[1, 2, 3]])

    # Test HookedTransformer - backward hooks should work
    with hooked_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", make_test_hook("hooked"))]):
        output = hooked_model(test_input)
        # Check that the backward hook was registered
        assert len(hooked_model.blocks[0].hook_mlp_out.bwd_hooks) > 0  # type: ignore

        # Trigger backward pass
        output.sum().backward()

    # Test TransformerBridge - backward hooks are ignored (BUG)
    # With compatibility mode, TransformerBridge should have the same hook names as HookedTransformer
    with bridge_model.hooks(bwd_hooks=[("blocks.0.hook_mlp_out", make_test_hook("bridge"))]):
        output = bridge_model(test_input)
        # This assertion demonstrates the bug - no backward hooks are registered
        assert len(bridge_model.blocks[0].hook_mlp_out.bwd_hooks) > 0

        # Backward pass won't trigger the hook
        output.sum().backward()

    # Verify the hooks were called appropriately
    assert hook_called["hooked"], "HookedTransformer backward hook should have been called"
    assert hook_called["bridge"], "TransformerBridge backward hook was not called (BUG)"


def test_bridge_context_compute_batch_behaves_like_legacy_context_compute_batch(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

    test_inputs = bridge_model.tokenizer.encode("Hello, world!", return_tensors="pt")

    # Setup attribution contexts
    bridge_ctx = bridge_model.setup_attribution(test_inputs)  # type: ignore
    legacy_ctx = legacy_model.setup_attribution(test_inputs)  # type: ignore

    # Run forward passes to populate residual activations
    with bridge_ctx.install_hooks(bridge_model):
        cache = {}

        def _cache_ln_final_in_hook(acts, hook):
            cache["ln_final.hook_in"] = acts

        bridge_model.run_with_hooks(
            test_inputs.expand(32, -1),  # type: ignore
            fwd_hooks=[("ln_final.hook_in", _cache_ln_final_in_hook)],
        )
        residual = cache["ln_final.hook_in"]
        bridge_ctx._resid_activations[-1] = bridge_model.ln_final._original_component(residual)

    with legacy_ctx.install_hooks(legacy_model):
        legacy_model.run_with_cache(test_inputs.expand(32, -1), names_filter="ln_final.hook_in")  # type: ignore
        residual = legacy_model.ln_final(legacy_ctx._resid_activations[-1])
        legacy_ctx._resid_activations[-1] = residual

    # Test compute_batch with logit vectors (similar to how it's used in attribution)
    n_layers, n_pos, _ = bridge_ctx.activation_matrix.shape
    batch_size = 3

    # Create test inject_values (similar to logit_vecs in attribution)
    inject_values = torch.randn(batch_size, bridge_model.cfg.d_model)
    layers = torch.full((batch_size,), n_layers)
    positions = torch.full((batch_size,), n_pos - 1)

    legacy_rows = legacy_ctx.compute_batch(
        layers=layers,
        positions=positions,
        inject_values=inject_values,
        retain_graph=True,
    )

    bridge_rows = bridge_ctx.compute_batch(
        layers=layers,
        positions=positions,
        inject_values=inject_values,
        retain_graph=True,
    )

    assert torch.allclose(bridge_rows, legacy_rows, atol=1e-2, rtol=1e-2)

    # Test compute_batch with feature vectors (similar to encoder_vecs usage)
    feat_layers, feat_pos, _ = bridge_ctx.activation_matrix.indices()
    if len(feat_layers) > 0:
        # Take first few features for testing
        n_test_features = min(5, len(feat_layers))
        test_indices = torch.arange(n_test_features)

        bridge_feature_rows = bridge_ctx.compute_batch(
            layers=feat_layers[test_indices],
            positions=feat_pos[test_indices],
            inject_values=bridge_ctx.encoder_vecs[test_indices],
            retain_graph=False,
        )

        legacy_feature_rows = legacy_ctx.compute_batch(
            layers=feat_layers[test_indices],
            positions=feat_pos[test_indices],
            inject_values=legacy_ctx.encoder_vecs[test_indices],
            retain_graph=False,
        )

        assert torch.allclose(bridge_feature_rows, legacy_feature_rows, atol=1e-2, rtol=1e-2)


def test_bridge_attribute_behaves_like_legacy_attribute(
    replacement_model_pair: tuple[ReplacementModel, LegacyReplacementModel],
):
    bridge_model, legacy_model = replacement_model_pair

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
