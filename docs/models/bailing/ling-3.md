# Ling 3.0

[Ling 3.0](https://huggingface.co/collections/inclusionAI/ling-30) is a hybrid Mixture-of-Experts language model family from inclusionAI. Megatron Bridge supports the public Tiny, Tiny Base, and Flash checkpoints through the Bailing bridge with automatic Hugging Face configuration and bidirectional weight conversion.

Tiny and Tiny Base have 24 logical KDA/MLA blocks, 128 routed experts, and low-rank-Q MLA; Tiny Base additionally includes one MTP layer. Flash has 42 logical blocks, 512 routed experts, direct-Q MLA, and one MTP layer. Each logical block maps to two MCore HybridModel positions. All variants require Hugging Face remote code.

<a id="megatron-core-requirements"></a>

**Megatron-Core requirements.** Conversion and SFT require native HybridModel KDA and head-wise MLA output gating, which the default main pin does not yet provide. Selecting `bash scripts/switch_mcore.sh dev` alone does not establish compatibility. Historical validation with Bridge `faf7687e` and MCore dev `4a4de865` was blocked by shared Bridge imports (`set_default_log_ranks` and `FullyShardedDataParallelV1`) unavailable in that dev pin. Unsupported native capabilities are rejected before Ling provider construction.

The Tiny Base verification card (`examples/model_verification_cards/ling-3.0-tiny-base/card.yaml`) records the exact historical Bridge/MCore combination used for successful conversion. These results do not validate the current Bridge with its dev pin or with the historical MCore revision alone. SFT remains unverified; Megatron KDA text generation is unsupported. Ling uses native MCore layers rather than Kimi K3 because the direct KDA forget-gate projection, RoPE-enabled MLA, and head-wise output gate have different layouts and semantics.

See the [Bailing examples](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/bailing) for conversion commands and `src/megatron/bridge/recipes/bailing/h100/ling_v3_tiny_base.py` for the eight-GPU BF16 full-parameter SFT recipe.

## Flash SFT baseline on 64 H100 GPUs

`ling_v3_flash_sft_64gpu_h100_bf16_config` targets 64 H100 80 GB GPUs with
TP=PP=CP=ETP=1, EP=8, and Megatron-FSDP **v1** full parameter/gradient/optimizer
sharding. Dense data parallelism is 64; expert data parallelism is 8. The recipe
uses BF16 parameters, FP32 main parameters and gradients, sequence length 2048,
micro batch 1, global batch 128, and 100 SQuAD SFT steps, padding samples to 2048.
Packing, FP8, CUDA graphs,
activation fusion, and shared-expert overlap are disabled. Checkpoints use
`fsdp_dtensor` at steps 50 and 100. The functional baseline below completed;
default-mode numerical replay and bitwise reproducibility remain limitations.

The Flash checkpoint's `expert_swiglu_limit_list` and
`share_expert_swiglu_limit_list` are preserved on import and export. Entries
address the 42 logical blocks; zero means no clipping. Routed experts in blocks
35–41 use limit 4. Shared experts in blocks 34–39 use 5 and blocks 40–41 use 7
(zero-based indices). MTP is outside these lists and remains unclipped.
The activation is `min(silu(gate), L) * clamp(up, -L, L)`, following
[SGLang's Bailing implementation](https://github.com/sgl-project/sglang/blob/822e3e2dd1bb236f5223d205aa4dcae91c8b3d40/python/sglang/srt/models/bailing_moe_v3.py).
MCore's pre-SiLU clamp is not equivalent. Bridge binds the logical index before
native MoE construction and replaces only the activation builder; TE projections,
native MoE dispatch, backward, and checkpoint parameter names remain unchanged.
The Ling provider selects attention backends automatically: Hybrid's Flash-only
default is incompatible with MLA's unequal Q/K and V head dimensions when only
FlashAttention 2 is installed. Automatic selection also enables cuDNN attention.
Flash also requires KV-only latent normalization: the native QK-norm resolver
otherwise adds a fused norm to the direct-Q projection. A Bridge MLA subclass
retains native validation and KV norm selection while selecting the declared
plain Q projection. Its forward/backward are inherited, and both main and MTP
layers must have exactly the checkpoint's parameter names.

The focused Bridge suite passed 117 tests on H100 with MCore dev
`43a2246f4e9ea02834c1df29ab8cf6c07b9e2dd6`. Native TE expert projections at
hidden size 2560 / FFN size 768 were compared in FP32 against an independent
FP64 Torch reference for outputs, inputs, routing probabilities, and all weight
gradients. The minimum symmetric similarity was 0.9999997. Strict FP32 testing
requires `NVIDIA_TF32_OVERRIDE=0`: PyTorch's TF32 switch alone does not control
TE's cuBLAS grouped weight-gradient GEMM. These module results do not establish
full-model BF16, distributed conversion, or SFT parity.

The full 64-H100 HF → m-FSDP → HF round trip compared all 63,783 tensors
exactly, with zero mismatches, missing tensors, or unexpected tensors. The
63,672 BF16 tensors were promoted losslessly to FP32 main parameters; the
111 source FP32 tensors retained their values. Native `fsdp_dtensor` save,
parameter zeroing, and reload restored identical local-state hashes on all
64 ranks. With the causal HF reference described below, all three prompts
(English, Chinese, and code) passed the cosine threshold of 0.99 for every
logical layer and the final logits, with identical next-token predictions.
The minimum layer cosine was 0.99771; logit cosines were 0.99944–0.99973.
These BF16 outputs are numerically close, not bitwise identical.

The 64-H100 SQuAD baseline completed 100 optimizer steps with EP8, TP=PP=CP=1,
m-FSDP v1, BF16 compute, FP32 optimizer state, sequence length 2048, MBS1,
and GBS128. All losses and gradient norms were finite, with no skipped steps.
Training loss was 1.7707 initially and 0.1542 at step 100; validation loss was
0.1245 at step 50 and 0.1291 at step 100 (four validation iterations each).
The additional post-training validation on the next batches was 0.1522.
The step-50 and step-100 optimizer checkpoints saved successfully;
resuming step 50 also completed five updates and saved step 55. However, the
default-mode replay failed a relative tolerance of 1e-3: maximum relative
differences were 0.00794 for loss and 0.01930 for gradient norm. This recipe
does not enable deterministic mode. Native unfused MoE uses `scatter_add`,
and kernel tuning can select different accumulation orders; the responsible
operation has not been isolated.

A separate controlled audit loaded step 50, enabled the existing determinism
overrides, saved step 51, and compared continuous step 52 with a fresh process
resuming step 51. On all 64 ranks, the pre-step model state (6,255 tensors per
rank), both microbatch inputs, and per-token loss outputs had identical SHA-256
hashes. The scheduler and consumed-sample counters also matched. The reported
loss was exactly 0.1791224033 in both runs; gradient norms were 2.7026512623 and
2.7026770115 (relative difference 9.53e-6). This one-step comparison passes the
original 1e-3 numerical tolerance, but the stricter bitwise check fails. The
audit compared optimizer metadata only: `DistributedOptimizer.state_dict()`
does not expose moment tensors. Neither elementwise optimizer-state equality
nor elementwise gradient equality was established by this audit.

For this controlled configuration, add `--deterministic` and
`model.gdn_pre_gated_delta_rule_fusion=false` to the launcher below. Export
`NCCL_ALGO=Ring`, `NCCL_EP_HT_EM_AG_SCAN_MODE=1`,
`NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`MAMBA_DETERMINISTIC=1`, and `CAUSAL_CONV1D_DETERMINISTIC=1` before starting
Python, in addition to the recipe's `NVIDIA_TF32_OVERRIDE=0` and
`CUDA_DEVICE_MAX_CONNECTIONS=32`. These settings do not establish bitwise
backward reproducibility or a multi-step deterministic SFT baseline.

Default-mode step time across steps 11–100, excluding the first step after
checkpoint 50, had median 6.276 s and mean 7.601 s. Throughput computed from
the mean was 34,490 padded tokens/s across 64 GPUs; it includes padding.
A late-training NVML sample reached 58.96 GiB on the busiest GPU; this is a
sample, not a whole-run peak. This is a short functional baseline, not a
long-training quality or optimized-throughput result.

Validation used Bridge base `b25a454abcaff82b0fa58434a3ade50141b42634` plus
this change and MCore dev `43a2246f4e9ea02834c1df29ab8cf6c07b9e2dd6` on
64 H100 GPUs, with the `nemo_26.06.sqsh` container and the pinned HF revision
below. SLURM jobs were 22349123 (117 tests), 22347401 (weight round trip and
checkpoint reload), 22348446 (causal forward), 22348676 (100-step baseline
and five-step default resume), and 22350420 (controlled replay). Run artifacts
are under `runtime/ckpt_tools/ling3_flash_20261119` in the local toolkit
checkout, including source/script hashes, resolved config, original logs,
TensorBoard scalars, per-rank replay JSON, and the retained failed checks.

An eight-H100 integration smoke also completed two native Adam steps with
m-FSDP v1, EP2, expert-DP4, TP=PP=CP=1, and activation recomputation. It retained
Flash's production hidden/head/expert dimensions and used six logical blocks,
sequence length 256, random weights, and no MTP. All ranks had finite loss and
gradient norm; peak allocated memory was 48.13 GiB. No MCore or m-FSDP source
changes were required. This reduced-depth smoke is not the 64-GPU SFT baseline.

The launch configuration is separate from the recipe. In an existing 64-rank
distributed environment, after numerical validation:

```bash
export LING_V3_FLASH_HF_PATH=/path/to/pinned/ling-flash-snapshot
uv run python scripts/training/run_recipe.py \
  --recipe ling_v3_flash_sft_64gpu_h100_bf16_config --mode sft \
  --pretrained_checkpoint /path/to/validated/checkpoint \
  --save_dir /path/to/baseline/checkpoints
```

The public HF revision is pinned to
`fcc60a52b34c69c6271d0ac8a8cb0815020b6a43`. A local snapshot must correspond to
that revision. Before baseline training, validate module outputs and gradients,
pure-layout weight round trips, full-model logits, and checkpoint save/load.
The published HF Python forward omits the clipping lists, so it is insufficient
as the sole numerical reference for Flash. Cross-check against the cited
SGLang implementation and record the reference revision and clipping convention.
Use `attn_implementation="eager"` for the HF reference and verify causal masks:
the published MLA forward always uses eager attention, while its SDPA mask path
can omit the mask that this eager implementation needs. A default HF run is
therefore not a sufficient causal-attention reference.
Pipeline parallelism is outside this baseline's validation scope.

## Flash smoke recipe on one eight-GPU H100 node

`ling_v3_flash_smoke_pretrain_8gpu_h100_bf16_config` is an **early, unoptimized**
reduced-depth functional smoke that starts from random weights and uses mock
tokens. It keeps Flash's first group, containing five KDA blocks followed by
one MLA block. These are **6 complete HF blocks** and **12 MCore HybridModel
positions**, because MCore counts each attention and MLP position separately.
The pattern is `K-K-KEKEKE+E`.

Hidden/head/expert dimensions and the original first-six-block dense/MoE
schedule are preserved: two dense MLPs followed by four MoE MLPs. Clipping lists
are sliced from the original HF configuration; all first-six-block limits are
zero. MTP is disabled. Sequence length is **4096**, micro batch 1, global batch
8, and training runs for two optimizer steps. Parallelism is TP=PP=CP=ETP=1,
EP=8, with expert-DP1 and m-FSDP v1 full sharding. This smoke does not establish
pretrained Flash fine-tuning quality or steady-state throughput.

The smoke calls `ling_v3_flash_sft_64gpu_h100_bf16_config()` directly and keeps
its optimizer, BF16/FP32 precision, seed, parallelism, recomputation, and
environment settings. It only overrides the layer prefix, sequence length,
MTP, random initialization, mock data, global batch/step horizon, schedule
duration, validation, and checkpoint I/O. The eight-GPU allocation changes
dense DP to 8 and expert DP to 1 while keeping the baseline's EP8 and PP1.

From the repository root inside a compatible environment on an allocated
eight-GPU node:

```bash
export LING_V3_FLASH_HF_PATH=/path/to/pinned/ling-flash-snapshot
uv run python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/training/run_recipe.py \
  --recipe ling_v3_flash_smoke_pretrain_8gpu_h100_bf16_config \
  --mode pretrain --max_steps 2
```

The HF path supplies model configuration and tokenizer files, not pretrained
weights. Omit `--load_dir`, `--pretrained_checkpoint`, and `--save_dir`: the
recipe explicitly disables checkpoint loading and saving. `--max_steps`
also determines the learning-rate decay duration. The SFT entry requires a
checkpoint, so this random-initialization smoke uses the pretraining entry.

The current six-block, sequence-4096 recipe passed four focused recipe tests,
the runner dry-run, and two optimizer steps on eight H100 80 GB GPUs (SLURM
allocation 22421638, using the MCore/container revisions above). Losses were
12.48887 and 12.37632, with finite gradient norms, no skipped/NaN iterations,
and exit code zero. Rank 0 reported peak allocated memory of 37.721 GB and peak
reserved memory of 42.855 GB; these are not all-rank maxima. No CPU offload was
enabled. The second step took 0.603 s and reported 44.3 model TFLOP/s/GPU;
one post-compilation step is not a steady-state performance measurement.

The earlier six-block, sequence-256 version passed all 146 recipe/runner tests
and completed two optimizer steps on eight H100 GPUs (SLURM job 22414077).
Those historical results apply to the previous smaller workload.

<!-- BEGIN GENERATED VERIFIED CONFIGURATIONS -->

## Verified configurations

Choose an exact recorded configuration to see its command and expected result. These selectors are generated from the authoritative verification cards and never synthesize combinations.

<a id="verified-ling-3.0-tiny-base"></a>
### Run a configuration

Choose a workflow, precision, and exact recorded combination. The command and expected result update below.

<div class="verification-model-explorer" data-model-explorer>
  <div class="verification-model-controls" hidden>
    <div class="verification-capability-tabs" role="tablist" aria-label="Workflow">
      <button type="button" role="tab" aria-selected="true" data-capability-tab="import-export">Import & Export</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="pretrain">Pretrain</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="benchmark" disabled>Benchmark</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="sft">SFT</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="lora">LoRA</button>
      <button type="button" role="tab" aria-selected="false" data-capability-tab="long-context">Long Context</button>
    </div>
    <div class="verification-filter-row">
      <div class="verification-precision-controls" aria-label="Precision filter">
        <span>Precision</span>
        <button type="button" class="is-active" data-precision="">All</button>
        <button type="button" data-precision="bf16">BF16</button>
        <button type="button" data-precision="fp8_mx">FP8 MX</button>
        <button type="button" data-precision="nvfp4">NVFP4</button>
      </div>
      <div class="verification-hardware-controls" aria-label="GPU filter">
        <span>GPU</span>
        <button type="button" class="is-active" data-hardware="">All</button>
        <button type="button" data-hardware="H200">H200</button>
      </div>
      <span class="verification-combination-count" aria-live="polite"></span>
    </div>
  </div>
  <div class="verification-combination-list" hidden>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-hf-to-megatron-cpu" aria-controls="ling-3-0-tiny-base-hf-to-megatron-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-hf-to-megatron-gpu" aria-controls="ling-3-0-tiny-base-hf-to-megatron-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Import · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-megatron-to-hf-cpu" aria-controls="ling-3-0-tiny-base-megatron-to-hf-cpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · CPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="import-export" data-precision="bf16" data-hardware="" data-status="verified" data-entry="ling-3-0-tiny-base-megatron-to-hf-gpu" aria-controls="ling-3-0-tiny-base-megatron-to-hf-gpu" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Export · GPU</strong>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="pretrain" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-pretrain-h200" aria-controls="ling-3-0-tiny-base-pretrain-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Pretrain · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="sft" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-sft-h200" aria-controls="ling-3-0-tiny-base-sft-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>SFT · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="long-context" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-sft-long-context-h200" aria-controls="ling-3-0-tiny-base-sft-long-context-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>Long Context · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
    <button type="button" class="verification-combination" data-capability="lora" data-precision="bf16" data-hardware="H200" data-status="unverified" data-entry="ling-3-0-tiny-base-peft-h200" aria-controls="ling-3-0-tiny-base-peft-h200" aria-pressed="false">
      <span class="verification-combination-heading">
        <strong>LoRA · H200</strong>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </span>
      <span class="verification-combination-meta">BF16</span>
    </button>
  </div>
  <div class="verification-model-details">
    <article id="ling-3-0-tiny-base-hf-to-megatron-cpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-hf-to-megatron-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device cpu --nodes 1 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/cpu-megatron --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU import exits successfully and creates a reloadable iter_0000000 distributed checkpoint from the pinned public revision.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-hf-to-megatron-gpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-hf-to-megatron-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Import · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh import --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/gpu-megatron --torch-dtype bfloat16 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed import exits successfully on 8 GPUs at TP1/PP1/EP8/ETP1 and creates a reloadable iter_0000000 checkpoint with MLA, KDA, MoE, and MTP weights intact.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-megatron-to-hf-cpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-megatron-to-hf-cpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · CPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device cpu --nodes 1 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/cpu-megatron/iter_0000000 --hf-path work/model-verification/ling-3.0-tiny-base/cpu-hf-export --torch-dtype bfloat16 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>CPU export exits successfully and writes the complete indexed Hugging Face checkpoint. All 9,686 tensors match the pinned source exactly in keys, shapes, dtypes, and values with max_abs_diff=0.0. AutoConfig, AutoTokenizer, and BailingMoeV3ForCausalLM reload successfully under Transformers 4.56.2; the full model has 8,209,994,528 parameters.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-megatron-to-hf-gpu" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-megatron-to-hf-gpu" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Export · GPU</h4>
        <span class="verification-status verification-status--verified" title="Verified">✓ Verified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>not specified</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>2026-09-03</dd></div>
      </dl>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <div class="verification-command">
          <div class="verification-command-heading">
            <span>Command</span>
            <button type="button" class="verification-copy-command">Copy</button>
          </div>
          <pre><code class="language-bash">./scripts/conversion/convert.sh export --executor slurm --device gpu --nodes 1 --gpus-per-node 8 --hf-model inclusionAI/Ling-3.0-tiny-base --hf-revision bab7297fa02713af237e378bf21107718b8e0e1a --megatron-path work/model-verification/ling-3.0-tiny-base/gpu-megatron/iter_0000000 --hf-path work/model-verification/ling-3.0-tiny-base/gpu-hf-export --torch-dtype bfloat16 --distributed-save --save-every-n-ranks 1 --tp 1 --pp 1 --ep 8 --etp 1 --trust-remote-code</code></pre>
        </div>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Distributed export exits successfully on 8 GPUs at TP1/PP1/EP8/ETP1. All 9,686 tensors match the pinned source exactly in keys, shapes, dtypes, and values with max_abs_diff=0.0, including the source FP32 KDA and router state. BailingMoeV3ForCausalLM reloads successfully under Transformers 4.56.2 with 8,209,994,528 parameters.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-pretrain-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-pretrain-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Pretrain · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Verification requires a bounded public-data pretraining run with finite loss, no skipped or NaN iterations, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-sft-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-sft-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>SFT · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>The 8-GPU validation loaded the pinned Tiny Base checkpoint, transformed 10,000 pinned public SQuAD rows into 898 sequence-length-2048 packs at 99.12 percent efficiency, built TP1/PP1/CP1/EP8/ETP1 with the MTP layer, and completed the first forward/backward pass. It then stopped before a complete optimizer step: Bridge training_log() passed num_moe_layers to megatron.core.transformer.moe.moe_utils.track_moe_metrics(), but the tested MCore dev commit 4d197408bfd8513289cc69a9112f37798e6fdd4c did not accept that keyword. This describes the historical dev validation, not the default main pin. Verified status requires an unpatched 100-step run with finite loss, all five metrics, and a reloadable final checkpoint.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-sft-long-context-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-sft-long-context-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>Long Context · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Verification requires a public long-context Tiny Base SFT run with supported sequence packing and context parallelism, finite loss, no skipped or NaN iterations, and all five metrics.
</p>
      </section>
    </article>
    <article id="ling-3-0-tiny-base-peft-h200" class="verification-model-detail" data-entry-detail="ling-3-0-tiny-base-peft-h200" tabindex="-1">
      <header class="verification-model-detail-heading">
        <h4>LoRA · H200</h4>
        <span class="verification-status verification-status--unverified" title="Unverified">○ Unverified</span>
      </header>
      <dl class="verification-model-detail-meta">
        <div><dt>Hardware</dt><dd>H200</dd></div>
        <div><dt>Precision</dt><dd>BF16</dd></div>
        <div><dt>Last verified</dt><dd>—</dd></div>
      </dl>
      <section class="verification-recorded-metrics">
        <h5>Recorded metrics</h5>
        <dl class="verification-metric-list">
          <div>
            <dt>Initial loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Final loss</dt>
            <dd>None</dd>
          </div>
          <div>
            <dt>Step time · last 10 avg</dt>
            <dd>None ms</dd>
          </div>
          <div>
            <dt>Model throughput · last 10 avg</dt>
            <dd>None TFLOP/s/GPU</dd>
          </div>
          <div>
            <dt>Token throughput · last 10 avg</dt>
            <dd>None tokens/s/GPU</dd>
          </div>
        </dl>
      </section>
      <section class="verification-command-section">
        <h5>Exact command</h5>
        <p>No runnable command is recorded for this status.</p>
      </section>
      <section class="verification-expected-result">
        <h5>Expected result</h5>
        <p>Verification requires a public Tiny Base PEFT recipe with an audited adapter target set, finite loss, no skipped or NaN iterations, all five metrics, and a reloadable adapter checkpoint.
</p>
      </section>
    </article>
  </div>
</div>

<!-- END GENERATED VERIFIED CONFIGURATIONS -->
