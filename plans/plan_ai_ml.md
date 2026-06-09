# MominOS AI/ML Subsystem — Architecture & Training Plan

## Goal
Design, train, and produce a **specialized ~1.5B MoE language model** (8 experts, top-2 routing, ~350M active params) purpose-built for kernel/process fault diagnosis, plus all Python-side tooling: synthetic data pipeline, PyTorch training code, Q8_0 quantization, MOM1 converter, and golden-intermediate validation harness.

**Division of responsibility:**
- **You (kernel)** → Phase 0 (SSE, scheduler, disk, libm) + C inference engine (Phases 1-2)
- **Me (AI/ML)** → Model architecture design, training code, data generation, RL alignment, MoE design, quantization, MOM1 converter, validation harness

## Research Summary

**Reference MoE architectures studied:**
- **MiniCPM-MoE-8x2B** (OpenBMB): 8 experts, top-2, 13.6B total / 4B active. Uses sparse upcycling from a dense checkpoint. Load balancing loss (coefficient 0.01). Router initialized with normal distribution (mean=0, variance=0.01).
- **DeepSeek-MoE 16B** (DeepSeek): Fine-grained expert segmentation + shared expert isolation. Trained from scratch on 2T tokens. Uses Z-loss for training stability alongside load balancing loss. ~40% compute of comparable dense models.
- **SmolLM2-1.7B** (HuggingFace): Reference dense architecture — Llama-style with GQA, 24 layers, 2048 hidden dim, RoPE, SwiGLU. Good template for a single expert.

**Training recipes:**
- **Load balancing loss**: `alpha * n_experts * sum_i(f_i * P_i)` where f_i = fraction of tokens routed to expert i, P_i = avg router probability for expert i. Typical alpha = 0.01.
- **Z-loss**: `beta * 1/n * sum_i(log(sum_j exp(router_logits_ij)))^2`. Stabilizes router logits. Typical beta = 0.001.
- **DPO (Direct Preference Optimization)**: Replaces RLHF's reward model + RL with a single preference-based loss. More stable, simpler to train. Preferred over PPO for small models.
- **Sparse upcycling**: Copy a pretrained dense checkpoint, replicate FFN weights across N experts, randomly initialize router. Significantly cheaper than training MoE from scratch.

**Data generation for fault diagnosis:**
- Teacher model (e.g., Claude via API, or a capable open model) generates synthetic (fault context → diagnosis → fix) pairs
- Key format: structured fault data (vector, RIP, CR2, error code, register state, disassembly) → natural language diagnosis + suggested fix
- Can also generate hard negatives for DPO pairs (wrong diagnosis vs correct diagnosis)

**Quantization:**
- Q8_0 block format (llama.cpp standard): 32 int8 weights + 1 fp16 scale per block
- Achieves ~2x memory reduction over fp16 with minimal accuracy loss
- Well-documented, easy to convert to MOM1 format

**Hardware constraints:**
- No GPU in this sandbox — training code will be written as portable scripts
- User to run on their own hardware (recommended: 1× RTX 3090/4090 24GB)
- Data generation via API (user provides key for OpenRouter or HuggingFace)
- Small-scale CPU tests possible here for validation

---

## Architecture Design

### MominoMoE-1.5B — Config

```
d_model:       1024
n_layers:      16
n_heads:       16
n_kv_heads:    4           (GQA ratio 4:1)
head_dim:      64
d_ff:          2048        (per-expert hidden dim)
n_experts:     8
top_k:         2
n_shared_experts: 1        (DeepSeek-inspired shared expert)
vocab_size:    32000
context_len:   2048
rope_base:     10000.0
rms_eps:       1e-5
norm_first:    true        (pre-norm, Llama-style)
activation:    SwiGLU     (silu(gate(x)) * up(x))
```

**Parameter breakdown (total ~1.5B):**
- Embedding: 32000 × 1024 = 32.8M
- Per MoE layer (×16): attention(1024²×4×3≈12.6M) + MoE-8×SwiGLU(1024×2048×3×8/2=25.2M avg) + shared expert(1024×2048×3≈6.3M) + output(1024²≈1M) + RMSNorm ≈ 48M
- Total non-embedding: ~770M
- With embedding: ~800M
- Active per token: ~200M (attention + 2 experts + shared expert)
- With Q8_0 quantization: ~800MB on disk

Wait, let me recalculate more carefully:

**Per MoE layer:**
- Q projection: 1024 × 1024 = 1.05M  (n_heads * head_dim = 16*64 = 1024)
- K projection: 1024 × 256 = 0.26M  (n_kv_heads * head_dim = 4*64 = 256)
- V projection: 1024 × 256 = 0.26M
- O projection: 1024 × 1024 = 1.05M
- Attention total: ~2.6M

- MoE: 8 experts, each with gate/up/down projections:
  - gate: 1024 × 2048 = 2.1M
  - up: 1024 × 2048 = 2.1M
  - down: 2048 × 1024 = 2.1M
  - Per expert: 6.3M. 8 experts: 50.4M. Active: 2 experts = 12.6M.
  - Shared expert: gate/up/down = 6.3M
  - Router: 1024 × 8 = 8K
  - MoE total: ~56.7M

Total per layer: ~59.3M

16 layers: ~949M
Embedding: 32.8M
LM head: 1024 × 32000 = 32.8M (if untied) — let's tie embeddings to save params

Tied embedding → 32.8M shared
16 layers × ~59.3M = 949M
Total: ~982M (let's call it ~1B)

Active per token: 2.6M (attn) + 12.6M (2 experts) + 6.3M (shared expert) = ~21.5M per layer
× 16 = ~344M active params

With fp32: ~4 GB RAM for weights
With Q8_0: ~1 GB RAM for weights

This fits the spec's target of ~1-2B total, ~250-500M active.

Let me adjust n_layers to reach ~1.5B total:
18 layers: 18 × 59.3M = 1.07B + 32.8M = ~1.1B — a bit low
Actually, let me go with 20 layers: 20 × 59.3M = 1.19B + 32.8M = ~1.22B — close to 1.5B but a bit low
Or bump d_model to 1280:
d_model=1280 → head_dim=80, n_heads=16, n_kv_heads=4
Q: 1280×1280=1.64M, K: 1280×320=0.41M, V: 1280×320=0.41M, O: 1280×1280=1.64M → attn=4.1M
MoE per expert: 1280×2048×3=7.86M × 8 = 62.9M + shared 7.86M = 70.8M
Per layer: ~74.9M
16 layers: 1.2B + embedding 32000×1280=41M ≈ 1.24B
Active: (4.1M + 2×7.86M + 7.86M) × 16 = 27.68M × 16 = ~443M active

That's closer. Let me go with this. Actually, let's keep it simpler and more conservative to ensure it fits in Q8_0 under 1GB — the C engine will thank you.

Final config:
```
d_model: 1024
n_layers: 20
n_heads: 16
n_kv_heads: 4
head_dim: 64
d_ff: 2048
n_experts: 8
top_k: 2
n_shared_experts: 1
vocab_size: 32000
context_len: 2048
```

Parameters: 20 × 59.3M = 1.186B + 32.8M(embedding) = ~1.22B total
Active: ~430M per token (all experts compute each layer but only 2 + shared are active)
Q8_0 on disk: ~1.22 GB
fp16 KV cache (2048 context, 20 layers, 4 kv_heads × 64 head_dim × 2 = 4KV_cache_params = 20 × 1024 × 256 × 2bytes × 2(K+V) = ~21MB

Actually KV cache formula: context_len × n_kv_heads × head_dim × n_layers × 2 (K+V) × 2 bytes (fp16)
= 2048 × 4 × 64 × 20 × 2 × 2 = 2048 × 4 × 64 × 80 = 41.9 MB

Weights at Q8_0: 1.22 GB
KV cache fp16: ~42 MB
Total: ~1.26 GB — fits comfortably in 4GB system.

OK, on to writing the actual plan with subtasks.</think>Good, now let me write the full AI/ML plan.

<｜DSML｜tool_calls>
<｜DSML｜invoke name="write_file">
<｜DSML｜parameter name="content" string="true"># MominOS AI Subsystem — AI/ML Architecture & Training Plan

## Goal
Design, train, and deliver a **specialized ~1.2B MoE language model** (8 experts, top-2 + 1 shared expert, ~430M active params) purpose-built for kernel/process fault diagnosis. Includes all Python-side deliverables: synthetic data pipeline, PyTorch model definition, training scripts (pretrain → SFT → DPO), Q8_0 quantizer, MOM1 converter, and golden-intermediate validation harness.

**Your lane (kernel)**: SSE/FPU, scheduler, disk driver, libm, C inference engine in `src/ai/`, C tokenizer, C daemon.
**My lane**: Everything Python-side — model architecture, training, data, quantization, tooling, validation.

---

## Architecture — MominoMoE-1.2B

```
                  MominoMoE-1.2B Configuration
┌─────────────────────────────────────────────┐
│ d_model:          1024                       │
│ n_layers:         20                         │
│ n_heads:          16                         │
│ n_kv_heads:       4          (GQA 4:1)      │
│ head_dim:         64                         │
│ d_ff:             2048      (per expert)     │
│ n_experts:        8                          │
│ top_k:            2                          │
│ n_shared_experts: 1        (DeepSeek-style)  │
│ vocab_size:       32000                      │
│ context_len:      2048                       │
│ rope_base:        10000.0                    │
│ rms_eps:          1e-5                       │
│ activation:       SwiGLU                     │
│ norm:             pre-RMSNorm                │
│ embeddings:       tied (input/output share)  │
│ total params:     ~1.22B                     │
│ active params:    ~430M    (per token)       │
│ Q8_0 on disk:     ~1.22 GB                   │
│ KV cache (fp16):  ~42 MB    (2048 ctx)      │
└─────────────────────────────────────────────┘
```

**Architecture notes:**
- Standard Llama/Mixtral-style pre-norm transformer — matches what the C engine in `src/ai/` expects
- Each MoE layer = 1 shared expert (always active, captures common knowledge) + 8 routed experts (top-2 activated)
- SwiGLU FFN per expert: `silu(x @ W_gate) * (x @ W_up) @ W_down`
- Router: `softmax(x @ W_router)` → top-2 → renormalize
- Q8_0 block quantization for deployment (32 int8 + fp16 scale per block)
- Tied embeddings (input embedding = lm_head weight transpose) saves ~33M params

**Why this architecture:**
1. **8 experts + top-2** matches the spec exactly; MiniCPM-MoE-8x2B proved this ratio works well
2. **Shared expert** (DeepSeek innovation): captures cross-cutting knowledge so routed experts specialize better
3. **GQA (4:1)**: saves KV cache memory without meaningful quality loss
4. **20 layers × 1024 dim**: hits ~1.2B total comfortably fitting Q8_0 in <1.5 GB

---

## Training Pipeline — Three Stages

```
Stage 1: Pretrain (next-token prediction)
  Data: general + technical corpus (~50B tokens)
  Loss: cross-entropy + load_balancing_loss(α=0.01) + z_loss(β=0.001)
  LR: cosine decay, 3e-4 → 3e-5, 2000-step warmup
  Optimizer: AdamW (β1=0.9, β2=0.95), weight_decay=0.1
  Gradient clipping: 1.0
  Batch: ~256 sequences × 2048 tokens

Stage 2: Supervised Fine-Tuning (SFT)
  Data: synthetic (fault → diagnosis → fix) pairs — see Data section
  Format: <system_prompt><fault_context><diagnosis><fix>
  LR: 1e-5, constant, 3 epochs
  Loss: cross-entropy on response tokens only

Stage 3: DPO Alignment (Direct Preference Optimization)
  Data: (fault_context, chosen_diagnosis, rejected_diagnosis) preference pairs
  LR: 5e-6, constant, 1 epoch
  β (DPO temp): 0.1
  Generates hard negatives via teacher model + self-critique
```

---

## Data Generation Pipeline

### Synthetic Fault-Diagnosis Dataset

**Teacher**: Use a capable model (Claude 4 Sonnet via OpenRouter, or Llama 3.1 70B via HF Inference API) to generate structured training data.

**Data format per example**:
```json
{
  "fault": {
    "vector": 14,
    "name": "Page Fault",
    "rip": "0x401a2c",
    "cr2": "0x00000000",
    "error_code": {"present": 0, "write": 1, "user": 0},
    "process": "webserver",
    "pid": 7,
    "register_state": {"rax": "0x0", "rbx": "0x7f...", ...},
    "klog_snippet": "[pid 7] accept() returned -1, fd table exhausted"
  },
  "diagnosis": "Null pointer dereference in webserver's request handler. The 'conn' pointer was never assigned because accept() returned -1 (file descriptor table exhausted). The handler tried to dereference conn->socket without checking for NULL.",
  "fix": "Raise the per-process file descriptor limit, or check accept()'s return value before using 'conn'. Suggested: add 'if (conn == NULL) return;' after accept().",
  "action_type": "SUGGEST_CMD",
  "action_payload": "ulimit -n 4096"
}
```

**Generation strategy (multi-round with teacher):**
1. **Seed templates**: 50 hand-written fault patterns (null deref, buffer overflow, div-by-zero, page faults, GP faults...)
2. **Teacher expansion**: For each seed template, instruct the teacher to generate 100 variants with randomized registers, addresses, and context details
3. **Negative generation for DPO**: For each example, ask the teacher to produce 2 diagnoses — one correct, one plausible-but-wrong (wrong root cause, wrong register, wrong fix)
4. **Quality filter**: Reject samples where the teacher self-scores confidence < 0.8

**Target dataset sizes:**
- SFT: ~50,000 (fault → diagnosis + fix) examples
- DPO: ~10,000 (chosen vs rejected diagnosis) pairs
- Pretrain: general corpus (The Stack, GitHub code, technical docs, kernel source) — we provide the dataloader

---

## Deliverables — File Structure

All AI/ML work lives under `training/` (separate from kernel `src/`):

```
training/
  train_moe/                      # Main Python package
    __init__.py
    config.py                     # Model hyperparameters (MominoMoEConfig dataclass)
    model.py                      # MominoMoE PyTorch model (full MoE transformer)
    moe_layer.py                  # MoE module: router + top-2 dispatch + shared expert
    transformer_block.py          # Single transformer block (attention + MoE)
    attention.py                  # GQA attention with RoPE
    rmsnorm.py                    # RMSNorm
    swiglu.py                     # SwiGLU activation
    quant.py                      # Q8_0 quantization (forward + dequant)
    tokenizer.py                  # Load BPE tokenizer for inference pipeline
  
  training/
    __init__.py
    pretrain.py                   # Stage 1: next-token prediction training loop
    sft_train.py                  # Stage 2: supervised fine-tuning
    dpo_train.py                  # Stage 3: DPO alignment
    train_utils.py                # Shared training helpers (LR schedule, checkpointing, logging)
    dataset.py                    # Dataset classes for all 3 stages
    data_collator.py              # Collation + padding
  
  data/
    generate_dataset.py           # Synthetic data generation pipeline (calls teacher API)
    templates.py                  # Seed fault templates
    quality_filter.py             # Filter low-quality generated samples
    prompt_templates.py           # System prompts for generation + training
  
  eval/
    eval_diagnosis.py             # Evaluation harness for diagnosis quality
    golden_harness.py             # **Golden-intermediate reference generator**
    compare_logits.py             # Compare C engine logits vs PyTorch reference
  
  convert/
    convert_model.py              # **MOM1 converter** (PyTorch → .mom file with Q8_0)
    export_config.py              # Export model config as C header for the engine
  
  requirements.txt                # Python dependencies
  
tools/
  convert_model.py                # Symlink or copy of training/convert/convert_model.py
  golden_harness.py               # Symlink or copy of training/eval/golden_harness.py

docs/
  MOMINO_MOE_ARCHITECTURE.md     # Architecture reference for C engine implementer
  TRAINING_RECIPE.md              # How to run training (for the GPU-capable machine)
```

---

## Subtasks

### Subtask A: Project Scaffold + Model Architecture
**Create the full training/ directory structure and implement the model.**

- Write `training/train_moe/config.py` — dataclass with all hyperparameters, serializable to/from dict
- Write `training/train_moe/rmsnorm.py` — RMSNorm with eps
- Write `training/train_moe/attention.py` — GQA attention with RoPE, causal mask, KV cache stub
- Write `training/train_moe/swiglu.py` — SwiGLU FFN module
- Write `training/train_moe/moe_layer.py` — MoE module: router (learned linear), top-2 dispatch with auxiliary balancing loss, shared expert, efficient selective computation
- Write `training/train_moe/transformer_block.py` — Pre-norm block: RMSNorm → Attn → residual → RMSNorm → MoE → residual
- Write `training/train_moe/model.py` — Full MominoMoE: embedding → 20 blocks → final RMSNorm → lm_head (tied weights)

**Deliverable**: `momino_moe_summary()` prints param count, active params, memory estimate. Quick CPU test: forward pass on random tokens produces non-NaN output.

### Subtask B: Q8_0 Quantization + MOM1 Converter
**Implement the Q8_0 block quantization scheme and the full MOM1 binary format converter.**

- Write `training/train_moe/quant.py`:
  - `quantize_q8_0(tensor)` → (q8_data, scales) — blocks of 32 weights
  - `dequantize_q8_0(q8_data, scales, shape)` → fp32 tensor
- Write `training/convert/convert_model.py`:
  - Reads PyTorch `.pt`/`.safetensors` checkpoint
  - Quantizes all weight tensors to Q8_0 (except norms/biases which stay fp32)
  - Writes MOM1 binary blob per Section A of the spec:
    - `mom_model_header` (magic "MOM1", version, all arch fields)
    - `n_tensors` × `mom_tensor_desc` (name, dtype=2 for q8_0, shape, offset, nbytes)
    - Aligned tensor data (Q8_0 blocks)
  - Round-trip test: convert → load back in Python → verify values match to Q8_0 precision
- Write `training/convert/export_config.py`:
  - Generates a C header `src/ai/model_config.h` with all constants the C engine needs

**Deliverable**: A `.mom` file that the C engine can load. Round-trip verified in Python.

### Subtask C: Golden-Intermediate Validation Harness
**Build the critical debug tool before training even starts.**

- Write `training/eval/golden_harness.py`:
  - Load a small reference model (or just the MominoMoE with random weights)
  - Run a single fixed input through the forward pass
  - **Dump every intermediate tensor**: token embeddings, post-each-RMSNorm, post-each-attention, post-each-MoE, pre-softmax attention scores, final logits
  - Save as a golden `.npz` file with named arrays
  - Include a CLI that prints the diff at the **first** mismatching op
- Write `training/eval/compare_logits.py`:
  - Load a golden `.npz` and compare against C engine dumps
  - Reports per-tensor MSE, max absolute error, and first divergence point

**Deliverable**: Python script that generates golden intermediates + comparison tool. A note in `docs/TRAINING_RECIPE.md` describing how the C engine implementer uses this.

### Subtask D: Synthetic Data Generation Pipeline
**Build the teacher-driven data generator for fault-diagnosis pairs.**

- Write `training/data/templates.py`:
  - 50 seed fault templates covering all x86-64 exception vectors
  - Each template: type, description, register patterns, typical root causes, typical fixes
- Write `training/data/prompt_templates.py`:
  - System prompt for the teacher (explain the diagnosis task)
  - Few-shot examples of well-formatted diagnoses
  - Prompt for generating DPO preference pairs (correct vs. plausible-but-wrong diagnosis)
- Write `training/data/generate_dataset.py`:
  - CLI: `python generate_dataset.py --provider openrouter --model anthropic/claude-sonnet-4 --output data/sft_data.jsonl`
  - Generates SFT examples + DPO pairs from seed templates using the teacher API
  - Parallel generation with configurable concurrency
  - Progress tracking and resume capability
  - Quality filtering per `training/data/quality_filter.py`

**Deliverable**: Data generation scripts ready to run with a user-provided API key. Sample generated output formats documented.

### Subtask E: Training Infrastructure
**Write all three training stages as executable scripts.**

- Write `training/training/dataset.py`:
  - `PretrainDataset`: loads raw text, tokenizes, chunks to 2048-token sequences
  - `SFTDataset`: (fault_context, diagnosis, fix) → tokenized with loss mask on response
  - `DPODataset`: (fault_context, chosen_tokens, rejected_tokens) for DPO loss
- Write `training/training/train_utils.py`:
  - Cosine LR scheduler with linear warmup
  - Checkpoint manager (save/load with optimizer state, resume)
  - Metrics logging (WandB optional, CSV fallback)
  - Gradient clipping, AMP (bfloat16) support
- Write `training/training/pretrain.py`:
  - DDP-compatible training loop
  - MoE auxiliary loss (load balancing + Z-loss) added to cross-entropy
  - Evaluation on held-out validation set
  - Saves checkpoints in MOM1-ready format
- Write `training/training/sft_train.py`:
  - Load pretrained checkpoint
  - Train on fault-diagnosis pairs with response-masked loss
  - Validation: held-out diagnosis examples
- Write `training/training/dpo_train.py`:
  - Load SFT checkpoint
  - DPO loss with chosen/rejected pairs
  - β=0.1, reference model frozen

**Deliverable**: Three scripts that, given data and a GPU, can train the full model end-to-end. Includes progress bars, checkpointing, logging, and recovery.

### Subtask F: DPO Alignment — Preference Optimization
**Extend the DPO training with practical refinements for a diagnosis-focused model.**

- Implement DPO loss with label smoothing (avoids overfitting to preference noise)
- Add **iterative DPO**: train → generate new preference pairs using current model → retrain
- Add evaluation callbacks: on held-out diagnosis set, compute diagnosis accuracy (exact match and ROUGE-L)
- Monitor for reward hacking (chosen log-prob gap widening without quality gain)

**Deliverable**: DPO training loop that produces an aligned model preferring precise, actionable diagnoses over vague or incorrect ones.

### Subtask G: Documentation + Integration Spec
**Document everything so the C engine can be built against it.**

- Write `docs/MOMINO_MOE_ARCHITECTURE.md`:
  - Exact tensor names and shapes for every parameter (so the C engine knows what to load)
  - Forward pass pseudocode matching exactly what the model does
  - Attention mask rules, RoPE formula, tokenizer details
  - KV cache layout
  - Q8_0 dequant procedure
- Write `docs/TRAINING_RECIPE.md`:
  - Hardware requirements (24GB GPU recommended)
  - Step-by-step: install deps → generate data → pretrain → SFT → DPO → convert → deploy
  - Expected wall-clock time per stage
  - Troubleshooting guide (NaN loss, expert collapse, OOM)

**Deliverable**: Two documentation files that make the C engine implementer and the training operator fully self-sufficient.

---

## Evaluation & Success Criteria

| Criterion | Method | Target |
|-----------|--------|--------|
| Model forward pass | CPU test on random input | Non-NaN, non-None, correct shape |
| MOM1 round-trip | Convert → load → compare | Max error < 2× Q8_0 quantization noise |
| Golden harness | Single forward pass dumps all intermediates | Each tensor named, shaped correctly |
| SFT loss convergence | Training run (even on small data) | Loss < 0.5 on validation set |
| DPO reward accuracy | Training run | Chosen preferred > rejected in ≥70% of held-out pairs |
| Diagnosis quality | Human eval on 100 unseen fault cases | ≥80% diagnoses judged "correct and actionable" |
| Generated `.mom` file | Convert final checkpoint | Loads correctly, C engine can process |
| Tokenizer round-trip | decode(encode(s)) == s | 100% for ASCII + UTF-8 |

---

## Implementation Order

Implement subtasks in order **A → B → C → D → E → F → G**.

- **A** (model) and **B** (converter) define the contract that the C engine builds against — get these right first
- **C** (golden harness) catches bugs early — build before training starts
- **D** (data) can run in parallel once the API is configured
- **E** (training) and **F** (DPO) are the compute-heavy stages — run on GPU
- **G** (docs) captures everything for integration

---

## Hardware Requirements (User's Machine)

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Data generation | Any machine + API key | Same |
| Pretrain | 1× RTX 3090 24GB (fp16, with activation checkpointing) | 2× RTX 4090 24GB (DDP, ~3 days) |
| SFT | 1× RTX 3090 24GB (~4 hours) | Same |
| DPO | 1× RTX 3090 24GB (~2 hours) | Same |
| Q8_0 conversion | Any machine | Same |
| C engine integration | No GPU needed | Same (runs on QEMU target) |