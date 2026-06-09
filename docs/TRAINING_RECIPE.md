# MominoMoE-1.2B Training Recipe

## Overview

This document describes how to train the MominoMoE-1.2B model end-to-end across three stages:

1. **Pretrain** — next-token prediction on a general + technical corpus
2. **SFT** — supervised fine-tuning on synthetic fault-diagnosis pairs
3. **DPO** — direct preference optimization with preference pairs

After training, the model is converted to MOM1 binary format (Q8_0 quantized) for deployment on the C inference engine.

---

## Hardware Requirements

### Minimum (Single GPU)

| Component | Requirement |
|-----------|-------------|
| GPU | 1× NVIDIA RTX 3090/4090 (24GB VRAM) or 1× NVIDIA L4 (23GB VRAM) |
| RAM | 32 GB system RAM |
| Disk | 50 GB free for datasets + checkpoints |
| Precision | fp16 / bf16 with gradient checkpointing |

### Recommended (Multi-GPU)

| Component | Requirement |
|-----------|-------------|
| GPU | 2× NVIDIA RTX 4090 (24GB) or 2× NVIDIA L4 |
| RAM | 64 GB system RAM |
| Interconnect | NVLink or PCIe 4.0 |
| Precision | fp16 with DDP |

### Stage-Specific Memory Estimates

| Stage | GPU Memory (fp16) | Time (1× L4) | Time (2× L4) |
|-------|-------------------|-------------|-------------|
| Pretrain (50B tokens) | ~22 GB | ~14 days | ~7 days |
| SFT (50k examples, 3 epochs) | ~22 GB | ~4 hours | ~2 hours |
| DPO (10k pairs, 1 epoch) | ~22 GB | ~2 hours | ~1 hour |
| MOM1 conversion | ~6 GB | ~5 minutes | — |

**Memory-efficient settings for 24GB GPU:**
```
--batch-size 4 --grad-accum-steps 8 --max-seq-len 2048
Activation checkpointing enabled by default
AMP (fp16) enabled by default
```

---

## Environment Setup

### Dependencies

```bash
# Python 3.10+ required
pip install torch>=2.12.0  # CUDA 12.x
pip install transformers>=5.x
pip install datasets>=3.x
pip install accelerate>=1.x
pip install peft>=0.14.x
pip install bitsandbytes>=0.45.x
pip install trl>=0.15.x
pip install wandb  # optional, for experiment tracking
pip install sentencepiece  # for tokenizer
```

### Quick Verify

```python
import torch
print(torch.__version__)        # Should be 2.12.0+
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # NVIDIA L4 or similar
```

---

## Step 1: Data Preparation

Before training begins, generate the synthetic fault-diagnosis dataset.

### Generate SFT Data

Requires an API key for the teacher model (OpenRouter or HuggingFace Inference API).

```bash
cd /root/MominOS/training

# Generate SFT fault-diagnosis pairs from 56 seed templates
python data/generate_dataset.py \
  --mode sft \
  --api-type openrouter \
  --model-tier sonnet \
  --max-samples 50000 \
  --output-dir data/raw/

# Generate DPO preference pairs (chosen vs rejected diagnoses)
python data/generate_dataset.py \
  --mode dpo \
  --api-type openrouter \
  --model-tier sonnet \
  --max-samples 10000 \
  --output-dir data/raw/

# Quality filter the generated data
python data/generate_dataset.py \
  --mode augment \
  --source data/raw/sft_data.jsonl \
  --output-dir data/processed/

# Expected output files:
#   data/processed/sft_train.jsonl   (~50,000 examples)
#   data/processed/sft_eval.jsonl    (~5,000 examples)
#   data/processed/dpo_train.jsonl   (~10,000 examples)
#   data/processed/dpo_eval.jsonl    (~1,000 examples)
```

### Pretrain Data

For pretraining, use a text corpus tokenized into 2048-token chunks as `.npy` files.

```bash
# Expected format: numpy array of shape (num_sequences, 2048) with int32 token IDs
# Example: data/pretrain/train.npy, data/pretrain/eval.npy
```

---

## Step 2: Pretraining

Pretrain the MoE model from scratch (or from a dense checkpoint via sparse upcycling).

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW | β1=0.9, β2=0.95, eps=1e-8 |
| Learning rate | 3e-4 | Peak LR, cosine decay to 3e-5 |
| Warmup | 2,000 steps | Linear warmup |
| Weight decay | 0.1 | Applied to all weights except norms/biases |
| Gradient clip | 1.0 | Global norm |
| Batch size (effective) | 256 seq × 2048 tokens = 524,288 tokens | 8 GPUs × 4 batch × 8 grad accum → adjusted for setup |
| Precision | fp16 (AMP) | with GradScaler |
| MoE aux loss coeff | 0.01 | Load balancing |
| MoE z-loss coeff | 0.001 | Router stability |

### Command

**Single GPU (with gradient accumulation):**

```bash
cd /root/MominOS/training

source venv/bin/activate

python training/pretrain.py \
  --data-path /root/MominOS/training/data/pretrain/train.npy \
  --output-dir /root/MominOS/training/checkpoints/pretrain \
  --batch-size 4 \
  --grad-accum-steps 64 \
  --max-steps 500000 \
  --max-seq-len 2048 \
  --lr 3e-4 \
  --warmup-steps 2000 \
  --weight-decay 0.1 \
  --fp16
```

**Multi-GPU (DDP):**

```bash
cd /root/MominOS/training

torchrun --nproc_per_node=2 training/pretrain.py \
  --data-path /root/MominOS/training/data/pretrain/train.npy \
  --output-dir /root/MominOS/training/checkpoints/pretrain \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --max-steps 250000 \
  --max-seq-len 2048 \
  --lr 3e-4 \
  --warmup-steps 2000 \
  --weight-decay 0.1 \
  --fp16 \
  --ddp
```

### Checkpointing

Checkpoints are saved every `--save-every` steps (default 5000) to `--output-dir`.

```
checkpoints/pretrain/
├── checkpoint-5000/
│   ├── model.pt         # Model state dict
│   ├── optimizer.pt     # Optimizer state
│   ├── scheduler.pt     # LR scheduler
│   ├── config.json       # Model config
│   └── training_state.json  # Step, loss, metrics
├── checkpoint-10000/
└── ...
```

To resume from a checkpoint, use `--resume`:

```bash
python training/pretrain.py \
  ... \
  --resume /root/MominOS/training/checkpoints/pretrain/checkpoint-10000
```

### Monitoring

- **Loss**: Should consistently decrease. Initial loss ≈ ln(vocab_size) ≈ 10.37
- **Perplexity**: Should drop from ~vocab_size (32000) toward single digits
- **MoE load balancing**: `min_expert_load` and `max_expert_load` should converge toward 0.125 (1/8 uniform)
- **Z-loss**: Should remain small (< 1.0)
- **Gradient norm**: Should stay near 1.0 (clipped)

### Expected Wall Time (50B tokens)

| Configuration | Time |
|-------------|------|
| 1× L4 (batch 4, grad accum 64, ~8 tok/s/GPU) | ~14 days |
| 2× L4 (batch 4, grad accum 32, ~16 tok/s) | ~7 days |
| 4× L4 (batch 4, grad accum 16, ~32 tok/s) | ~4 days |

---

## Step 3: Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model on fault-diagnosis pairs. Only response tokens contribute to loss.

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Pretrained checkpoint | From Step 2 |
| Learning rate | 2e-5 | Constant (no decay) |
| Warmup | 200 steps | Short warmup for fine-tuning |
| Epochs | 3 | On 50k examples |
| Batch size (effective) | 32 | Per-device 4 × grad accum 8 |
| Loss masking | Response-only | `--response-only-loss` (default: True) |
| Max seq len | 2048 | Truncate/pad |

### Command

```bash
cd /root/MominOS/training

source venv/bin/activate

python training/sft_train.py \
  --data-path /root/MominOS/training/data/processed/sft_train.jsonl \
  --output-dir /root/MominOS/training/checkpoints/sft \
  --init-from /root/MominOS/training/checkpoints/pretrain/checkpoint-50000/model.pt \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --max-steps 20000 \
  --max-seq-len 2048 \
  --lr 2e-5 \
  --warmup-steps 200 \
  --response-only-loss \
  --fp16 \
  --eval-every 500 \
  --save-every 1000
```

### SFT Data Format

JSONL file where each line contains:

```json
{"text": "<system_prompt>Analyze this fault: Page fault at RIP=0x401a2c...</system_prompt>", "response": "<diagnosis>The fault is a null pointer dereference...</diagnosis><fix>Check accept() return value...</fix>"}
```

The model computes loss only on the `response` portion.

### Expected Metrics

| Metric | Expected Value |
|--------|---------------|
| Training loss | < 0.5 |
| Validation loss | < 1.0 |
| Diagnosis accuracy (exact) | > 60% on held-out |
| Diagnosis ROUGE-L | > 0.7 |

### Dry-Run (Verify Setup)

```bash
python training/sft_train.py \
  --data-path /root/MominOS/training/data/dummy_sft_data.json \
  --dry-run \
  --max-steps 1 \
  --batch-size 2 \
  --max-seq-len 128
```

Expected output:
```
[INFO] Dry run mode — will run 1 step and exit
Model: 1,217,864,704 trainable params
Loaded 3 SFT samples from ...
Step 0: loss=10.xx, response tokens=31
Dry run OK
```

---

## Step 4: DPO Alignment

Align the SFT model using preference pairs (chosen vs rejected diagnoses).

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | SFT checkpoint | From Step 3 |
| Learning rate | 5e-6 | Very small — DPO is sensitive |
| β (DPO temperature) | 0.1 | Controls how much to penalize rejected |
| Label smoothing | 0.1 | Optional — reduces overconfidence |
| IPO mode | Optional | Use `--ipo` instead of standard DPO |
| Batch size (effective) | 16 | 4 per device × 4 grad accum |
| Epochs | 1 | On 10k pairs |
| Max seq len | 2048 | |

### Standard DPO

```bash
cd /root/MominOS/training

source venv/bin/activate

python training/dpo_train.py \
  --data-path /root/MominOS/training/data/processed/dpo_train.jsonl \
  --output-dir /root/MominOS/training/checkpoints/dpo \
  --init-from /root/MominOS/training/checkpoints/sft/checkpoint-10000/model.pt \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --max-steps 5000 \
  --max-seq-len 2048 \
  --lr 5e-6 \
  --beta 0.1 \
  --label-smoothing 0.1 \
  --fp16
```

### DPO with Label Smoothing

Label smoothing prevents the model from becoming overconfident (useful when preference data has noise):

```bash
python training/dpo_train.py \
  ... \
  --label-smoothing 0.1
```

### IPO (Identity Preference Optimization)

Alternative to DPO that is more stable but slower to converge:

```bash
python training/dpo_train.py \
  ... \
  --ipo \
  --beta 0.5
```

### DPO Data Format

JSONL with chosen/rejected pairs:

```json
{
  "prompt": "Analyze this fault: Page fault at RIP=0x401a2c, CR2=0x0...",
  "chosen": "Null pointer dereference: the conn pointer was NULL because accept() failed. Check return values.",
  "rejected": "Memory corruption: the page table entry was invalid. The fix is to increase swap space."
}
```

### Iterative DPO (Advanced)

The training script includes a placeholder `generate_new_pairs()` function. For production iterative DPO:

1. Train initial DPO model
2. Use the current model to generate multiple candidate diagnoses for each fault
3. Score them (using heuristic: correct root cause + actionable fix) → select best/worst
4. Add new preference pairs to training data
5. Retrain from SFT checkpoint with augmented data
6. Repeat for 2-3 iterations

Expected iteration schedule:
```
Iteration 0: SFT checkpoint → trained on original 10k pairs
  ↳ Generate 5k new pairs using the Iteration 0 model
Iteration 1: SFT checkpoint → trained on 15k pairs (original + new)
  ↳ Generate 5k more pairs using the Iteration 1 model
Iteration 2: SFT checkpoint → trained on 20k pairs (final model)
```

### Reward Hacking Monitor

The training script automatically detects reward hacking with `RewardStats`:

```python
# Warning signs logged automatically
#   "rapid_reward_growth" — margin growing > 0.5/step
#   "high_reward_variance" — variance > 10.0
#   "overconfident" — accuracy > 98% with low loss
```

If detected:
1. Reduce learning rate (5e-6 → 1e-6)
2. Increase label smoothing (0.1 → 0.2)
3. Add more rejected samples with plausible-sounding wrong diagnoses
4. Switch to IPO loss with `--ipo`

### Expected Metrics (after DPO)

| Metric | Before DPO (SFT only) | After DPO |
|--------|----------------------|-----------|
| Chosen reward | — | > 0.5 |
| Rejected reward | — | < -0.5 |
| Reward margin | — | > 1.0 |
| Preference accuracy | ~50% | > 70% |
| Diagnosis quality (human eval) | ~70% correct | > 80% correct |

---

## Step 5: Model Conversion to MOM1 Binary

Convert the trained PyTorch checkpoint to the MOM1 binary format with Q8_0 quantization for the C inference engine.

### Convert Checkpoint to .mom

```bash
cd /root/MominOS/training

source venv/bin/activate

# Convert with Q8_0 quantization (for deployment)
python convert/convert_model.py \
  --checkpoint /root/MominOS/training/checkpoints/dpo/checkpoint-5000/model.pt \
  --output /root/MominOS/training/models/momino_moe_v1.mom \
  --quantize

# Convert without quantization (for verification/testing)
python convert/convert_model.py \
  --checkpoint /root/MominOS/training/checkpoints/dpo/checkpoint-5000/model.pt \
  --output /root/MominOS/training/models/momino_moe_v1_fp32.mom \
  --no-quantize

# Verify the converted file
python convert/convert_model.py \
  --checkpoint /root/MominOS/training/checkpoints/dpo/checkpoint-5000/model.pt \
  --output /tmp/test.mom \
  --quantize \
  --verify

# Round-trip test (quantize → dequantize → compare)
python convert/convert_model.py \
  --checkpoint /root/MominOS/training/checkpoints/dpo/checkpoint-5000/model.pt \
  --output /tmp/rt_test.mom \
  --roundtrip
```

### Generate C Header

```bash
cd /root/MominOS/training

source venv/bin/activate

python convert/export_config.py
# Outputs: /root/MominOS/src/ai/model_config.h
```

The generated header contains all architecture constants, Q8_0 parameters, and tensor name macros for the C inference engine.

### MOM1 Binary Format

The output `.mom` file has this structure:

```
┌─────────────────────────────────┐
│ mom_model_header (128 bytes)    │ ← Magic "MOM1", version, all arch fields
├─────────────────────────────────┤
│ mom_tensor_desc[0] (64 bytes)   │ ← name, dtype, shape, offset, nbytes
│ mom_tensor_desc[1] (64 bytes)   │
│ ...                              │
├─────────────────────────────────┤
│ Tensor data (Q8_0 blocks,       │
│  aligned to 64 bytes)           │
│ [block 0: 32 int8 + 1 fp16]    │
│ [block 1: 32 int8 + 1 fp16]    │
│ ...                              │
└─────────────────────────────────┘
```

**MOM1 file sizes:**

| Format | File Size |
|--------|-----------|
| fp32 (no quantize) | ~4.87 GB |
| Q8_0 (quantized) | ~1.22 GB |

---

## Step 6: Deployment

### Copy to Target

```bash
# Copy to the C inference engine directory
cp /root/MominOS/training/models/momino_moe_v1.mom /root/MominOS/src/ai/models/
cp /root/MominOS/src/ai/model_config.h /root/MominOS/src/ai/
```

### Verify with Golden Harness

The golden harness validates the Python model output against the C engine.

```bash
cd /root/MominOS/training

source venv/bin/activate

# Generate golden intermediates from Python model
python eval/golden_harness.py \
  --checkpoint /root/MominOS/training/checkpoints/dpo/checkpoint-5000/model.pt \
  --output /root/MominOS/training/eval/golden_output.npz \
  --batch 2 \
  --seq_len 64 \
  --seed 42

# Expected: 586 tensors saved, 133 MB .npz file
# Contains: embed, logits, all 20 layers (attention inputs/outputs,
#   MoE router outputs, expert outputs, block outputs)

# Compare with C engine dump (once available)
python eval/compare_logits.py \
  --golden /root/MominOS/training/eval/golden_output.npz \
  --c-engine-dir /root/MominOS/src/ai/dumps/
```

---

## End-to-End Pipeline Script

For convenience, a single script can run the full pipeline:

```bash
#!/bin/bash
# train_full_pipeline.sh — Run from /root/MominOS/training

set -e

# Configuration
DATA_DIR="/root/MominOS/training/data"
CHECKPOINT_DIR="/root/MominOS/training/checkpoints"
MODEL_DIR="/root/MominOS/training/models"
PRETRAIN_DATA="${DATA_DIR}/pretrain/train.npy"
SFT_DATA="${DATA_DIR}/processed/sft_train.jsonl"
DPO_DATA="${DATA_DIR}/processed/dpo_train.jsonl"

# Activate environment
source /root/MominOS/venv/bin/activate

# Step 2: Pretrain
echo "=== STAGE 1: PRETRAIN ==="
python training/pretrain.py \
  --data-path ${PRETRAIN_DATA} \
  --output-dir ${CHECKPOINT_DIR}/pretrain \
  --batch-size 4 \
  --grad-accum-steps 64 \
  --max-steps 500000 \
  --max-seq-len 2048 \
  --lr 3e-4 \
  --fp16

# Step 3: SFT
echo "=== STAGE 2: SFT ==="
PRETRAIN_LAST=$(ls -d ${CHECKPOINT_DIR}/pretrain/checkpoint-* | sort -t- -k2 -n | tail -1)
python training/sft_train.py \
  --data-path ${SFT_DATA} \
  --output-dir ${CHECKPOINT_DIR}/sft \
  --init-from ${PRETRAIN_LAST}/model.pt \
  --batch-size 4 \
  --grad-accum-steps 8 \
  --max-steps 20000 \
  --lr 2e-5 \
  --fp16

# Step 4: DPO
echo "=== STAGE 3: DPO ==="
SFT_LAST=$(ls -d ${CHECKPOINT_DIR}/sft/checkpoint-* | sort -t- -k2 -n | tail -1)
python training/dpo_train.py \
  --data-path ${DPO_DATA} \
  --output-dir ${CHECKPOINT_DIR}/dpo \
  --init-from ${SFT_LAST}/model.pt \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --max-steps 5000 \
  --lr 5e-6 \
  --beta 0.1 \
  --label-smoothing 0.1 \
  --fp16

# Step 5: Convert
echo "=== STAGE 4: CONVERT ==="
DPO_LAST=$(ls -d ${CHECKPOINT_DIR}/dpo/checkpoint-* | sort -t- -k2 -n | tail -1)
mkdir -p ${MODEL_DIR}
python convert/convert_model.py \
  --checkpoint ${DPO_LAST}/model.pt \
  --output ${MODEL_DIR}/momino_moe_v1.mom \
  --quantize
python convert/export_config.py

# Step 6: Verify
echo "=== VERIFICATION ==="
python convert/convert_model.py --checkpoint ${DPO_LAST}/model.pt --output /tmp/verify.mom --verify
python eval/golden_harness.py --checkpoint ${DPO_LAST}/model.pt --output ${MODEL_DIR}/golden.npz

echo "=== PIPELINE COMPLETE ==="
echo "Model: ${MODEL_DIR}/momino_moe_v1.mom"
echo "Config: /root/MominOS/src/ai/model_config.h"
echo "Golden: ${MODEL_DIR}/golden.npz"
```

---

## Troubleshooting

### NaN Loss

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN in first 10 steps | Learning rate too high | Reduce `--lr` by 10× |
| Loss = NaN after stable training | Gradient explosion | Reduce `--grad-clip` to 0.5, check for data issues |
| Loss = NaN in MoE layers only | Router logits diverging | Increase `--z-loss-coeff` (0.001 → 0.01) |
| Loss = NaN after resume | LR scheduler reset | Use `--resume` instead of fresh `--init-from` |

### Expert Collapse

| Symptom | Likely Cause | Fix |
|---------|-------------|------|
| One expert gets all tokens | Load balancing loss too weak | Increase `--aux-loss-coeff` (0.01 → 0.1) |
| All experts get equal load | Load balancing loss too strong | Decrease `--aux-loss-coeff` (0.01 → 0.001) |
| Dead expert (zero tokens) | Router never selects it | Check router init; lower `--router-init-std` |
| Expert load oscillates | Learning rate too high | Reduce LR |

### OOM (Out of Memory)

| Symptom | Fix |
|---------|-----|
| CUDA OOM during forward | Reduce `--batch-size` (4 → 2 → 1) |
| CUDA OOM during backward | Reduce `--max-seq-len` (2048 → 1024) |
| CUDA OOM with checkpointing | Use `--cpu-offload` or reduce batch size further |
| System RAM OOM | Reduce `--num-workers` to 2 |

### Slow Training

| Symptom | Likely Cause | Fix |
|---------|-------------|------|
| GPU utilization < 50% | Data loading bottleneck | Increase `--num-workers` (2 → 4) |
| GPU utilization < 20% | Batch size too small | Increase `--batch-size` if memory permits |
| High CPU usage | Too many workers | Reduce `--num-workers` to 2 |
| Slow MoE forward | Expert routing overhead | Normal for MoE; can fuse with torch.compile |

### DPO-Specific Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|------|
| Reward margin grows but accuracy drops | Reward hacking | Add `--label-smoothing 0.2`, reduce `--lr` |
| Chosen and rejected both have negative reward | Reference model issue | Verify SFT checkpoint loads correctly |
| Accuracy stays at 50% | Preference data too noisy | Increase data quality threshold in `quality_filter.py` |
| Loss increases during DPO | Beta too low | Increase `--beta` (0.1 → 0.2) |

### MOM1 Conversion

| Symptom | Likely Cause | Fix |
|---------|-------------|------|
| "Tensor not found" during verify | Tensor name mismatch | Check `map_tensor_name()` in `convert_model.py` |
| Output .mom file too large | Skipped quantization | Use `--quantize` flag |
| C engine can't load .mom | Wrong dtype or alignment | Run `--verify` flag first |

---

## Quick Verification Commands

Run these after each training stage to verify correctness:

```bash
# 1. Verify model imports and param count
source venv/bin/activate
python -c '
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
m = MominoMoE(MominoMoEConfig())
print(m.summary())
'

# 2. Verify forward pass (random input)
python -c '
import torch
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
m = MominoMoE(MominoMoEConfig())
m.eval()
x = torch.randint(0, 32000, (2, 64))
out = m(x)
logits = out["logits"]
print(f"Shape: {logits.shape}")
print(f"Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
print(f"NaN: {torch.isnan(logits).any().item()}")
print(f"Inf: {torch.isinf(logits).any().item()}")
'

# 3. Verify Q8_0 round-trip
python -c '
from train_moe.quant import round_trip_test
import torch
t = torch.randn(1024, 1024)
r = round_trip_test(t)
print(f"MSE: {r[\"mse\"]:.8f}, Max error: {r[\"max_error\"]:.6f}")
assert r["mse"] < 1e-4, "Q8_0 round-trip failed"
print("Q8_0 round-trip OK")
'

# 4. Verify golden harness
python eval/golden_harness.py --checkpoint /path/to/model.pt \
  --output /tmp/golden_test.npz --batch 2 --seq_len 64 --seed 42

# 5. Verify data pipeline
python -c '
from data.templates import FAULT_TEMPLATES, count_templates
print(f"Templates: {len(FAULT_TEMPLATES)}")
ct = count_templates()
if isinstance(ct, dict):
    print(f"Template count: {sum(ct.values())}")
else:
    print(f"Template count: {ct}")
'

# 6. Verify training scripts import
python -c '
from training.pretrain import main as pm
from training.sft_train import main as sm
from training.dpo_train import main as dm
from training.dataset import SFTDataset
print("All training imports OK")
'

# 7. Verify C header generation
python convert/export_config.py
wc -c /root/MominOS/src/ai/model_config.h
```

---

## Model Architecture Summary

```
MominoMoE-1.2B
├── Embedding: 32,000 × 1,024 (= 32.8M)
├── 20× TransformerBlock
│   ├── RMSNorm (1,024)
│   ├── GQA Attention (16 heads, 4 KV heads, RoPE)
│   │   ├── q_proj: 1,024 → 1,024
│   │   ├── k_proj: 1,024 → 256
│   │   ├── v_proj: 1,024 → 256
│   │   └── o_proj: 1,024 → 1,024
│   ├── RMSNorm (1,024)
│   └── MoE Layer
│       ├── Router: 1,024 → 8
│       ├── 1× Shared Expert (SwiGLU: 1,024 → 2,048 × 3)
│       └── 8× Routed Experts (SwiGLU: 1,024 → 2,048 × 3)
├── RMSNorm (1,024)
└── LM Head: 1,024 → 32,000 (tied with embedding)

Total: ~1.22B parameters
Active per token: ~430M
Q8_0 on disk: ~1.22 GB
KV cache (fp16): ~42 MB
```