"""Native TE expert projections versus an independent Torch reference.

Run on one GPU. These tests retain Flash's per-expert 2560/768 dimensions,
with two local experts and short sequences. They do not validate EP/FSDP.
"""

import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.bailing.bailing_moe3_activation import BailingMoe3ClampedSwiGLU, _build_clamped_mlp
from megatron.bridge.models.bailing.bailing_moe3_spec import _missing_mcore_features


logger = logging.getLogger(__name__)
pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(bool(_missing_mcore_features()), reason="Requires native Ling-capable MCore dev"),
]


@pytest.mark.parametrize("is_mtp", [False, True])
def test_flash_native_mla_has_kv_norm_without_extra_direct_q_norm(native_groups, is_mtp):
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.bailing.bailing_moe3_bridge import BailingMoeV3Bridge
    from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec
    from tests.unit_tests.models.bailing.test_bailing_moe3_bridge import _flash_config

    config = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=_flash_config()))
    config.init_model_with_meta_device = True
    config.finalize()
    spec = bailing_moe3_hybrid_stack_spec(config).submodules.mla_layer
    with torch.device("meta"):
        layer = build_module(
            spec,
            config=config,
            layer_number=1 if is_mtp else 11,
            is_mtp_layer=is_mtp,
            add_layer_offset=False,
            pg_collection=native_groups,
        )
    assert set(dict(layer.named_parameters())) == {
        "input_layernorm.weight",
        "self_attention.linear_q_proj.weight",
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.kv_layernorm.weight",
        "self_attention.linear_kv_up_proj.weight",
        "self_attention.linear_gate.weight",
        "self_attention.linear_proj.weight",
    }


@pytest.mark.parametrize(
    "layer_number,is_mtp,routed,shared",
    [(70, False, None, 5), (72, False, 4, 5), (84, False, 4, 7), (2, True, None, None)],
)
def test_native_layer_binds_limits_before_building_experts(native_groups, layer_number, is_mtp, routed, shared):
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.bailing.bailing_moe3_provider import BailingMoe3HybridProvider
    from megatron.bridge.models.bailing.bailing_moe3_spec import bailing_moe3_hybrid_stack_spec

    config = BailingMoe3HybridProvider(
        num_layers=84,
        hidden_size=2560,
        num_attention_heads=32,
        ffn_hidden_size=6144,
        moe_ffn_hidden_size=768,
        moe_shared_expert_intermediate_size=768,
        num_moe_experts=512,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        init_model_with_meta_device=True,
        expert_swiglu_limit_list=[0] * 35 + [4] * 7,
        share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
    )
    config.finalize()
    spec = bailing_moe3_hybrid_stack_spec(config).submodules.moe_layer
    with torch.device("meta"):
        layer = build_module(
            spec,
            config=config,
            layer_number=layer_number,
            is_mtp_layer=is_mtp,
            add_layer_offset=False,
            pg_collection=native_groups,
        )
    assert all(parameter.is_meta for parameter in layer.parameters())
    for module, expected in ((layer.mlp.experts, routed), (layer.mlp.shared_experts, shared)):
        if expected is None:
            assert not isinstance(module.activation_func, BailingMoe3ClampedSwiGLU)
        else:
            assert isinstance(module.activation_func, BailingMoe3ClampedSwiGLU)
            assert module.activation_func.limit == expected


@pytest.fixture(scope="module")
def native_groups(tmp_path_factory):
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # TE calls cuBLAS directly; torch.backends.matmul.allow_tf32 alone does not
    # control its FP32 grouped weight-gradient GEMM.
    previous_tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    rendezvous = tmp_path_factory.mktemp("ling_dist") / "store"
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield ProcessGroupCollection.use_mpu_process_groups()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    if previous_tf32_override is None:
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    else:
        os.environ["NVIDIA_TF32_OVERRIDE"] = previous_tf32_override


def _compare(actual, reference, name):
    actual, reference = actual.detach().double().flatten(), reference.detach().double().flatten()
    if actual.norm() == 0 and reference.norm() == 0:
        return
    cosine = F.cosine_similarity(actual, reference, dim=0).item()
    similarity = (1 - (actual - reference).norm() / (actual.norm() + reference.norm()).clamp_min(1e-30)).item()
    logger.info("%s cosine=%.9f symmetric_similarity=%.9f", name, cosine, similarity)
    assert cosine >= 0.999, (name, cosine)
    assert similarity >= 0.999, (name, similarity)


@pytest.mark.parametrize("kind,limit", [("experts", 4.0), ("shared_experts", 5.0), ("shared_experts", 7.0)])
def test_native_te_clipped_expert_output_and_gradients(native_groups, kind, limit):
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=2,
        hidden_size=2560,
        num_attention_heads=32,
        ffn_hidden_size=6144,
        moe_ffn_hidden_size=768,
        moe_shared_expert_intermediate_size=768,
        num_moe_experts=512,
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        params_dtype=torch.float32,
        use_cpu_initialization=False,
    )
    submodules = get_moe_module_spec(use_te=True, num_experts=512, moe_grouped_gemm=True).keywords["submodules"]
    extra = {"num_local_experts": 2} if kind == "experts" else {"gate": False}
    module = _build_clamped_mlp(
        builder=getattr(submodules, kind), limit=limit, config=config, pg_collection=native_groups, **extra
    ).cuda()
    torch.manual_seed(456)
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(std=param.shape[-1] ** -0.5)
    x = (4 * torch.randn(16, 2560, device="cuda")).requires_grad_()
    rx = x.detach().double().requires_grad_()
    weights = {name: p.detach().double().requires_grad_() for name, p in module.named_parameters()}
    probs = torch.linspace(0.1, 0.9, 16, device="cuda").requires_grad_()
    rprobs = probs.detach().double().requires_grad_()

    def reference(inputs, w1, w2, scales=None):
        gate, up = (inputs @ w1.T).chunk(2, dim=-1)
        activated = gate / (1 + torch.exp(-gate))
        activated = torch.minimum(activated, torch.full_like(activated, limit))
        linear = torch.maximum(torch.minimum(up, torch.full_like(up, limit)), torch.full_like(up, -limit))
        hidden = activated * linear
        if scales is not None:
            hidden = hidden * scales.unsqueeze(-1)
        return hidden @ w2.T

    if kind == "experts":
        output, bias = module(x, torch.tensor([9, 7], device="cpu"), probs)
        expected = torch.cat(
            [
                reference(rx[:9], weights["linear_fc1.weight0"], weights["linear_fc2.weight0"], rprobs[:9]),
                reference(rx[9:], weights["linear_fc1.weight1"], weights["linear_fc2.weight1"], rprobs[9:]),
            ]
        )
    else:
        output, bias = module(x), None
        expected = reference(rx, weights["linear_fc1.weight"], weights["linear_fc2.weight"])
    assert bias is None
    _compare(output, expected, "output")
    upstream = torch.randn_like(output)
    (output * upstream).sum().backward()
    (expected * upstream.double()).sum().backward()
    _compare(x.grad, rx.grad, "input gradient")
    if kind == "experts":
        _compare(probs.grad, rprobs.grad, "routing probability gradient")
    for name, param in module.named_parameters():
        assert param.grad is not None, name
        _compare(param.grad, weights[name].grad, name)
