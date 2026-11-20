import importlib
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.bailing.bailing_moe3_bridge import BailingMoeV3Bridge
from megatron.bridge.training.config import MockGPTDatasetConfig
from tests.unit_tests.models.bailing.test_bailing_moe3_bridge import _flash_config, requires_ling_mcore


pytestmark = [pytest.mark.unit, requires_ling_mcore]


@pytest.mark.parametrize("local", [False, True])
def test_flash_sft_recipe_preserves_architecture_and_fsdp_contract(monkeypatch, local):
    module = importlib.import_module("megatron.bridge.recipes.bailing.h100.ling_v3_flash")
    source = _flash_config(
        expert_swiglu_limit_list=[0] * 35 + [4] * 7,
        share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
    )
    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=source))
    calls = []

    class FakeBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return SimpleNamespace(to_megatron_provider=lambda **_: provider)

    monkeypatch.setattr(module, "AutoBridge", FakeBridge)
    if local:
        monkeypatch.setenv("LING_V3_FLASH_HF_PATH", "/offline/ling-flash")
    else:
        monkeypatch.delenv("LING_V3_FLASH_HF_PATH", raising=False)
    cfg = module.ling_v3_flash_sft_64gpu_h100_bf16_config()
    expected_path = "/offline/ling-flash" if local else module.LING_V3_FLASH_HF_MODEL
    revision = {} if local else {"revision": module.LING_V3_FLASH_HF_REVISION}
    assert calls == [(expected_path, {"trust_remote_code": True, **revision})]
    assert cfg.tokenizer.tokenizer_model == expected_path
    assert cfg.tokenizer.hf_tokenizer_kwargs == revision
    for config in (cfg.model, cfg.ddp, cfg.scheduler, cfg.train, cfg.checkpoint):
        assert set(vars(config)) <= set(config.__dataclass_fields__)
    assert cfg.model.expert_swiglu_limit_list == source.expert_swiglu_limit_list
    assert cfg.model.share_expert_swiglu_limit_list == source.share_expert_swiglu_limit_list
    assert cfg.model.pipeline_model_parallel_size == cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.context_parallel_size == cfg.model.expert_tensor_parallel_size == 1
    assert cfg.model.attention_backend == AttnBackend.auto
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.ddp.use_megatron_fsdp and cfg.ddp.megatron_fsdp_version == 1
    assert (
        cfg.ddp.data_parallel_sharding_strategy
        == cfg.ddp.expert_data_parallel_sharding_strategy
        == "optim_grads_params"
    )
    assert cfg.ddp.megatron_fsdp_main_params_dtype == cfg.ddp.megatron_fsdp_main_grads_dtype == torch.float32
    assert cfg.checkpoint.ckpt_format == "fsdp_dtensor"
    assert cfg.checkpoint.save_interval == 50
    assert cfg.train.global_batch_size == 128 and cfg.train.micro_batch_size == 1
    assert cfg.train.train_iters == 100
    assert cfg.model.seq_length == cfg.dataset.seq_length == 2048
    assert cfg.dataset.enable_offline_packing is False
    assert cfg.dataset.dataset_kwargs["pad_to_max_length"] is True
    assert cfg.model.mtp_loss_scaling_factor == 0
    assert cfg.model.bias_activation_fusion is False
    assert cfg.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] != "1"


@pytest.mark.parametrize("steps", [2, 10])
def test_flash_smoke_recipe_is_exported_and_valid_without_checkpoint(monkeypatch, steps):
    module = importlib.import_module("megatron.bridge.recipes.bailing.h100.ling_v3_flash")
    source = _flash_config(
        expert_swiglu_limit_list=[0] * 35 + [4] * 7,
        share_expert_swiglu_limit_list=[0] * 34 + [5] * 6 + [7] * 2,
    )
    provider = BailingMoeV3Bridge().provider_bridge(SimpleNamespace(config=source))
    original_pattern = provider.hybrid_layer_pattern
    weight_requests = []

    class FakeBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            return FakeBridge()

        def to_megatron_provider(self, *, load_weights):
            weight_requests.append(load_weights)
            return provider

    monkeypatch.setattr(module, "AutoBridge", FakeBridge)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    baseline_factory = module.ling_v3_flash_sft_64gpu_h100_bf16_config
    baseline_calls = []

    def build_baseline():
        baseline_calls.append(True)
        return baseline_factory()

    monkeypatch.setattr(module, "ling_v3_flash_sft_64gpu_h100_bf16_config", build_baseline)
    factory = module.ling_v3_flash_smoke_pretrain_8gpu_h100_bf16_config
    for package in (
        "megatron.bridge.recipes",
        "megatron.bridge.recipes.bailing",
        "megatron.bridge.recipes.bailing.h100",
    ):
        assert getattr(importlib.import_module(package), factory.__name__) is factory
    cfg = factory()
    cfg.train.train_iters = steps
    # Check recipe assignments before MCore adds its derived private state.
    for config in (cfg.model, cfg.ddp, cfg.scheduler, cfg.train, cfg.checkpoint):
        assert set(vars(config)) <= set(config.__dataclass_fields__)
    cfg.validate()
    cfg.model.finalize()

    assert weight_requests == [False]
    assert baseline_calls == [True]
    assert cfg.checkpoint.load is cfg.checkpoint.pretrained_checkpoint is cfg.checkpoint.save is None
    assert isinstance(cfg.dataset, MockGPTDatasetConfig)
    assert cfg.scheduler.lr_decay_iters == steps
    assert cfg.scheduler.lr_warmup_iters == 0
    assert cfg.model.num_layers == len(cfg.model.hybrid_layer_pattern) == len(cfg.model.moe_layer_freq) == 12
    assert cfg.model.hybrid_layer_pattern == original_pattern[:12] == "K-K-KEKEKE+E"
    assert cfg.model.hybrid_layer_pattern.count("K") == 5
    assert cfg.model.hybrid_layer_pattern.count("+") == 1
    assert cfg.model.hybrid_layer_pattern.count("-") == 2
    assert sum(cfg.model.moe_layer_freq) == 4
    assert cfg.model.expert_swiglu_limit_list == source.expert_swiglu_limit_list[:6] == [0] * 6
    assert cfg.model.share_expert_swiglu_limit_list == source.share_expert_swiglu_limit_list[:6] == [0] * 6
    assert cfg.model.hidden_size == source.hidden_size
    assert cfg.model.num_moe_experts == source.num_experts
    assert cfg.model.seq_length == cfg.dataset.seq_length == 4096
    assert cfg.model.perform_initialization
    assert cfg.model.mtp_num_layers == 0 and cfg.model.mtp_hybrid_override_pattern is None
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.optimizer.lr == 5e-6 and cfg.optimizer.adam_beta2 == 0.98
    assert cfg.dataset.random_seed == cfg.rng.seed == 5678
    assert cfg.ddp.use_megatron_fsdp and cfg.ddp.megatron_fsdp_version == 1
