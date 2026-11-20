from functools import partial
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.bailing.bailing_moe3_activation import (
    BailingMoe3ClampedSwiGLU,
    build_bailing_moe3_moe,
    validate_swiglu_limits,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("limit", [4.0, 5.0, 7.0])
def test_post_silu_clamp_forward_and_analytic_backward(limit):
    # Include a gate above L whose SiLU is below L: pre-SiLU clipping fails here.
    gate = torch.tensor([-12.0, -1.0, 0.0, limit + 0.01, limit + 1.0, 12.0], dtype=torch.float64)
    up = torch.tensor([-12.0, 0.5, 2.0, -1.0, 3.0, 12.0], dtype=torch.float64)
    inputs = torch.cat([gate, up]).requires_grad_()
    actual = BailingMoe3ClampedSwiGLU(config=None, limit=limit)(inputs)
    sigmoid = 1 / (1 + torch.exp(-gate))
    silu = gate * sigmoid
    clipped_gate = torch.minimum(silu, torch.full_like(silu, limit))
    clipped_up = torch.maximum(torch.minimum(up, torch.full_like(up, limit)), torch.full_like(up, -limit))
    torch.testing.assert_close(actual, clipped_gate * clipped_up, rtol=1e-12, atol=1e-12)
    actual.sum().backward()
    gate_grad = clipped_up * sigmoid * (1 + gate * (1 - sigmoid)) * (silu <= limit)
    up_grad = clipped_gate * (up.abs() <= limit)
    torch.testing.assert_close(inputs.grad, torch.cat([gate_grad, up_grad]), rtol=1e-12, atol=1e-12)


def _capture(**kwargs):
    return kwargs


@pytest.mark.parametrize(
    "physical_layer,mtp,expert_limit,shared_limit",
    [
        (68, False, None, None),
        (70, False, None, 5),
        (72, False, 4, 5),
        (82, False, 4, 7),
        (84, False, 4, 7),
        (2, True, None, None),
    ],
)
def test_clipping_uses_logical_layer_and_excludes_mtp(physical_layer, mtp, expert_limit, shared_limit):
    native_experts = partial(_capture, submodules=SimpleNamespace(activation_func=None))
    native_shared = partial(_capture, submodules=SimpleNamespace(activation_func=None))
    native = partial(_capture, submodules=SimpleNamespace(experts=native_experts, shared_experts=native_shared))
    result = build_bailing_moe3_moe(
        builder=native,
        expert_limits=[0] * 35 + [4] * 7,
        shared_limits=[0] * 34 + [5] * 6 + [7] * 2,
        layer_number=physical_layer,
        is_mtp_layer=mtp,
        config=SimpleNamespace(),
    )
    assert result["layer_number"] == physical_layer
    for name, expected in (("experts", expert_limit), ("shared_experts", shared_limit)):
        assert getattr(result["submodules"], name).keywords.get("limit") == expected
    # A later layer must never mutate the reusable native spec.
    assert native.keywords["submodules"].experts is native_experts
    assert native_experts.keywords["submodules"].activation_func is None


@pytest.mark.parametrize("limits", [[0] * 41, [0] * 41 + [-1], [0] * 41 + [float("nan")]])
def test_reject_invalid_clipping_schedules(limits):
    with pytest.raises(ValueError, match="expert_swiglu_limit_list"):
        validate_swiglu_limits(limits, num_layers=42, field="expert_swiglu_limit_list")
