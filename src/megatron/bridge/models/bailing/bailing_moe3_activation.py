# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ling's post-SiLU clipping, preserving the native TE projection modules."""

import copy
import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn.functional as F


def validate_swiglu_limits(limits: Sequence[float] | None, *, num_layers: int, field: str) -> list[float] | None:
    """Validate HF logical-layer limits; zero denotes an unclipped layer."""
    if limits is None:
        return None
    if len(limits) != num_layers:
        raise ValueError(f"{field} must have {num_layers} logical-layer entries, got {len(limits)}.")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in limits):
        raise ValueError(f"{field} must contain finite, nonnegative clipping limits.")
    return list(limits)


class BailingMoe3ClampedSwiGLU(torch.nn.Module):
    """Apply ``clamp(silu(gate), max=L) * clamp(up, -L, L)``.

    This follows inclusionAI's SGLang BailingMLP implementation at revision
    822e3e2dd1bb236f5223d205aa4dcae91c8b3d40. MCore's generic activation
    clamp instead clips the gate *before* SiLU and is not equivalent.
    """

    def __init__(self, *, config: object, limit: float) -> None:
        super().__init__()
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("Ling SwiGLU requires a finite positive clipping limit.")
        self.limit = limit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, up = inputs.chunk(2, dim=-1)
        return F.silu(gate).clamp(max=self.limit) * up.clamp(min=-self.limit, max=self.limit)


def _build_clamped_mlp(*, builder, limit: float, config, **kwargs):
    """Change only the activation slot in an existing native MCore builder."""
    if config.bias_activation_fusion or config.use_transformer_engine_op_fuser:
        raise ValueError("Ling post-SiLU clipping requires unfused activations (disable activation/op fusion).")
    if config.activation_func_clamp_value is not None:
        raise ValueError("Ling post-SiLU clipping cannot be combined with MCore's pre-SiLU clamp.")
    config = copy.copy(config)
    config.use_te_activation_func = True
    submodules = copy.deepcopy(builder.keywords["submodules"])
    submodules.activation_func = partial(BailingMoe3ClampedSwiGLU, limit=limit)
    return builder(config=config, submodules=submodules, **kwargs)


def _build_clamped_experts(num_local_experts, config, *, builder, limit, **kwargs):
    """Preserve the native ExpertsBuilder's two positional arguments."""
    return _build_clamped_mlp(
        builder=builder, limit=limit, config=config, num_local_experts=num_local_experts, **kwargs
    )


def build_bailing_moe3_moe(
    *,
    builder: partial,
    expert_limits: Sequence[float] | None,
    shared_limits: Sequence[float] | None,
    layer_number: int,
    is_mtp_layer: bool = False,
    **kwargs: object,
) -> torch.nn.Module:
    """Attach independent routed/shared limits using global Hybrid positions.

    Each logical block occupies two positions. MTP positions restart at one
    and have no entries in the published clipping lists, so remain unclipped.
    """
    if is_mtp_layer:
        return builder(layer_number=layer_number, is_mtp_layer=True, **kwargs)
    logical_layer = (layer_number - 1) // 2
    submodules = copy.deepcopy(builder.keywords["submodules"])
    for field, limits in (("experts", expert_limits), ("shared_experts", shared_limits)):
        limit = limits[logical_layer] if limits is not None else 0
        if limit:
            wrapper = _build_clamped_experts if field == "experts" else _build_clamped_mlp
            setattr(submodules, field, partial(wrapper, builder=getattr(submodules, field), limit=limit))
    return builder(submodules=submodules, layer_number=layer_number, is_mtp_layer=False, **kwargs)
