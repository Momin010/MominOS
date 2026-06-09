# MominoMoE OS Copilot Agent Training Plan

## Goal
Train MominoMoE-1.2B (MoE 8 experts top-2 + shared, ~430M active) from random weights into a functional OS-level AI copilot agent. All training, benchmarking, and evaluation runs entirely on the GCP VM (mominos-ai-train, NVIDIA L4 23GB, 200GB free disk). No dataset download to local sandbox.

## Research Summary
- **Toucan-1.5M SFT**: 119,287 agent trajectories — messages are JSON strings with roles (user, assistant, tool_call, tool_response). Covers 495 MCPs with 2,000+ tools. Best dataset for teaching multi-tool function calling.
- **AgentBank intercode_bash**: 200 NL→bash trajectories with ReAct-style thought/action/observation loops.
- **AgentBank other subsets**: ~30K total across webarena, webshop, alfworld, apps, hotpotqa, etc.
- **MominoMoE-1.2B**: Verified on GCP — 1,217.9M total / 430.1M active params, forward pass [1,64,32000] logits, no NaN/Inf. L4 has 23GB VRAM, can train with fp16 + gradient checkpointing + micro-batch.
- **Training scripts**: sft_train.py and dpo_train.py verified on GCP with correct imports and relative paths.

## Approach
Build pipeline in 4 phases, all executing on the GCP VM:

### Phase 1: Dataset Download & Conversion (on GCP VM)
Write `training/data/prepare_agent_data.py` that:
1. Downloads Toucan-1.5M SFT subset (119K samples) via HuggingFace `datasets`
2. Converts JSON messages → MominoMoE prompt+response JSONL format suitable for SFTDataset
3. Downloads AgentBank intercode_bash + other NL→shell subsets
4. Converts to same format
5. Splits into train/validation/test (90/5/5)
6. Saves as JSONL files on GCP disk
7. Output: `data/agent_train.jsonl`, `data/agent_val.jsonl`, `data/agent_test.jsonl`

### Phase 2: Pre-Training Benchmark (untrained model)
Write `training/data/agent_benchmark.py` that loads the raw (untrained) model and evaluates on:
1. **Tool call accuracy**: 50 synthetic tool-use scenarios — measure JSON parseability, correct function name, correct arguments
2. **Bash command generation**: 200 AgentBank intercode_bash test trajectories — measure exact-match accuracy on bash commands
3. **Multi-turn coherence**: 50 multi-turn scenarios — measure context retention across 3-5 turns
4. **Output format compliance**: % of outputs that parse as valid JSON/structured format
5. **Response entropy**: token-level entropy as diversity baseline
6. **System log analysis**: 20 kernel log snippets — measure whether model identifies the fault type

### Phase 3: SFT Training (on GCP L4)
Run `training/training/sft_train.py` on agent data:
- 10 epochs on agent_train.jsonl (100K samples across Toucan + AgentBank)
- fp16 mixed precision, gradient checkpointing, gradient accumulation
- micro-batch size determined by VRAM testing
- Validation every 500 steps on agent_val.jsonl
- Save best checkpoint by validation loss

### Phase 4: DPO Training (on GCP L4)
Run `training/training/dpo_train.py`:
- Use fault diagnosis templates expanded via teacher API (OpenRouter) to generate chosen/rejected pairs
- Label smoothing, reward hacking monitor
- DPO for preference tuning on correct vs. incorrect diagnoses

### Phase 5: Post-Training Benchmark
Run the exact same agent_benchmark.py from Phase 2 on the trained checkpoint.
Produce side-by-side comparison table.

## Deliverables
All on GCP VM at `~/MominOS/`:
| File | Description |
|------|-------------|
| `training/data/prepare_agent_data.py` | Dataset download + conversion pipeline |
| `training/data/agent_benchmark.py` | Agent capability benchmark harness |
| `training/data/agent_train.jsonl` | Converted training data |
| `training/data/agent_val.jsonl` | Validation data |
| `training/data/agent_test.jsonl` | Test data (from intercode_bash held-out + synthetic) |
| `checkpoints/sft_best.pt` | Best SFT checkpoint |
| `checkpoints/dpo_best.pt` | Best DPO checkpoint |
| `benchmark_results/before_training.json` | Pre-training benchmark results |
| `benchmark_results/after_sft.json` | Post-SFT benchmark results |
| `benchmark_results/after_dpo.json` | Post-DPO benchmark results |
| `benchmark_results/comparison_report.md` | Side-by-side comparison with tables |

## Evaluation Criteria
- Tool call accuracy improves from ~random (5%) to ≥60% after SFT, ≥70% after DPO
- Bash command exact-match ≥50% after SFT
- Format compliance ≥80% after training
- Multi-turn completion rate ≥60% after training
- System log fault identification ≥50% after training

## Notes
- ALL downloads happen on GCP VM — nothing downloaded to local sandbox
- Use HuggingFace `datasets` library's streaming to avoid downloading full 22GB Toucan to disk
- Set HF_TOKEN on GCP for faster downloads
- Model is 1.22B params — fp16 training uses ~6GB + optimizer states ~8GB = fits in L4 23GB with gradient checkpointing
- All Python scripts will be written to local `/root/MominOS/` then SCP'd to GCP