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
"""Ling 3.0 Flash SFT and reduced-depth training smoke recipes for H100 GPUs."""

import os

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_squad_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig


LING_V3_FLASH_HF_MODEL = "inclusionAI/Ling-3.0-flash"
LING_V3_FLASH_HF_REVISION = "fcc60a52b34c69c6271d0ac8a8cb0815020b6a43"  # pragma: allowlist secret


def ling_v3_flash_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Full-parameter SFT with EP8, m-FSDP v1, and TP=PP=CP=ETP=1.

    Dense parameters shard across 64 ranks and expert parameters across eight
    expert-DP ranks. Set LING_V3_FLASH_HF_PATH for an offline snapshot, and pass
    --pretrained_checkpoint to run_recipe.py to load the validated weights.
    """
    cfg = _sft_common()
    hf_path = os.environ.get("LING_V3_FLASH_HF_PATH", LING_V3_FLASH_HF_MODEL)
    revision = {"revision": LING_V3_FLASH_HF_REVISION} if hf_path == LING_V3_FLASH_HF_MODEL else {}
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True, **revision).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.tokenizer.hf_tokenizer_kwargs = revision
    cfg.checkpoint.hf_trust_remote_code = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.init_model_with_meta_device = True
    cfg.model.seq_length = 4096
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.mtp_loss_scaling_factor = 0.0
    cfg.model.bias_activation_fusion = False
    cfg.model.gradient_accumulation_fusion = False
    cfg.model.use_transformer_engine_op_fuser = False
    cfg.model.fp8 = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.megatron_fsdp_version = 1
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.average_in_collective = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.megatron_fsdp_main_params_dtype = torch.float32
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = True

    cfg.dataset = default_squad_config(seq_length=cfg.model.seq_length, enable_offline_packing=False)
    cfg.dataset.dataset_kwargs["pad_to_max_length"] = True
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 128
    cfg.train.train_iters = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.fully_parallel_save = False
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 4
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "NVIDIA_TF32_OVERRIDE": "0",
    }
    return cfg


def ling_v3_flash_smoke_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Train Flash's first six main blocks plus MTP at length 4096 on eight H100s.

    This early, unoptimized functional smoke keeps the first group of five
    KDA blocks and one MLA block, including its original dense/MoE schedule
    and clipping limits. Production hidden/head/expert dimensions are retained.
    Inherit the 64-GPU SFT baseline's optimizer, precision, m-FSDP v1, and EP8
    settings (expert-DP1 on eight GPUs), including one MTP layer with zero
    auxiliary-loss weight. Override only the reduced workload and
    random-initialization smoke controls. Checkpoint I/O is disabled.
    """
    cfg = ling_v3_flash_sft_64gpu_h100_bf16_config()

    # Each HF block occupies an attention position and an MLP position in MCore.
    # Keep the first six-block group from the original Flash schedule.
    num_logical_layers = 6
    cfg.model.hybrid_layer_pattern = cfg.model.hybrid_layer_pattern[: 2 * num_logical_layers]
    cfg.model.num_layers = len(cfg.model.hybrid_layer_pattern)
    cfg.model.moe_layer_freq = [int(layer == "E") for layer in cfg.model.hybrid_layer_pattern]
    if cfg.model.expert_swiglu_limit_list is not None:
        cfg.model.expert_swiglu_limit_list = cfg.model.expert_swiglu_limit_list[:num_logical_layers]
    if cfg.model.share_expert_swiglu_limit_list is not None:
        cfg.model.share_expert_swiglu_limit_list = cfg.model.share_expert_swiglu_limit_list[:num_logical_layers]
    cfg.model.perform_initialization = True

    cfg.dataset = MockGPTDatasetConfig(
        seq_length=cfg.model.seq_length,
        random_seed=cfg.rng.seed,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        split="1,0,0",
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )
    cfg.train.global_batch_size = 8
    cfg.train.train_iters = 2
    cfg.scheduler.lr_warmup_iters = 0
    cfg.scheduler.lr_decay_iters = None  # Resolve from train_iters, including CLI overrides.
    cfg.validation.eval_iters = 0
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.load = None
    cfg.checkpoint.save = None
    return cfg


ling_v3_flash_sft_config = ling_v3_flash_sft_64gpu_h100_bf16_config

__all__ = [
    "LING_V3_FLASH_HF_MODEL",
    "LING_V3_FLASH_HF_REVISION",
    "ling_v3_flash_sft_64gpu_h100_bf16_config",
    "ling_v3_flash_sft_config",
    "ling_v3_flash_smoke_pretrain_8gpu_h100_bf16_config",
]
