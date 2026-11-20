# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""HybridModel provider for Ling 3.0."""

from dataclasses import dataclass

from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class BailingMoe3HybridProvider(HybridModelProvider, MLATransformerConfig):
    """Combine HybridModel construction with MLA configuration for Ling 3.0."""

    multi_latent_attention: bool = True
    # Hybrid's Flash-only default excludes cuDNN; FlashAttention 2 cannot
    # handle Ling MLA's unequal Q/K and V head dimensions on H100.
    attention_backend: AttnBackend = AttnBackend.auto
    # HF logical block indices, not HybridModel's attention/MLP positions.
    expert_swiglu_limit_list: list[float] | None = None
    share_expert_swiglu_limit_list: list[float] | None = None
