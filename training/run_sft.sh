#!/usr/bin/env bash
# Crash-resumable SFT launcher for MominoMoE on the L4.
# - 8-bit AdamW (via bitsandbytes) keeps a 1.2B model under the 23GB budget.
# - expandable_segments avoids the allocator fragmentation that caused the OOM.
# - Auto-resumes from the newest checkpoint after any crash/preemption.
# - Logs everything to a timestamped file (symlinked as latest.log).
# Safe to run detached in tmux; survives SSH/laptop disconnect.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/MominOS/training
source ~/MominOS/venv/bin/activate

OUT="$HOME/MominOS/checkpoints/sft"
mkdir -p "$OUT"
LOG="$OUT/train_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG" "$OUT/latest.log"

echo "[launcher] $(date) starting. Logging to $LOG" | tee -a "$LOG"

fails=0
last_ckpt_seen=""
while true; do
    latest=$(ls -1t "$OUT"/checkpoint_step*.pt 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        RESUME=(--resume "$latest")
        echo "[launcher] $(date) resuming from $latest" | tee -a "$LOG"
    else
        RESUME=()
        echo "[launcher] $(date) fresh start" | tee -a "$LOG"
    fi

    python3 training/sft_train.py \
        --data-path data/agent_train.jsonl \
        --val-data-path data/agent_val.jsonl \
        --output-dir "$OUT" \
        --batch-size 2 --grad-accum-steps 8 \
        --max-steps 10000 --max-seq-len 2048 \
        --lr 5e-5 --warmup-steps 100 \
        --eval-every 500 --save-every 250 --fp16 \
        "${RESUME[@]}" >> "$LOG" 2>&1
    code=$?

    if [ $code -eq 0 ]; then
        echo "[launcher] $(date) training completed successfully (exit 0)." | tee -a "$LOG"
        break
    fi

    newest=$(ls -1t "$OUT"/checkpoint_step*.pt 2>/dev/null | head -1)
    if [ "$newest" != "$last_ckpt_seen" ] && [ -n "$newest" ]; then
        fails=0
        last_ckpt_seen="$newest"
    else
        fails=$((fails + 1))
    fi

    if [ $fails -ge 5 ]; then
        echo "[launcher] $(date) exited $code; 5 consecutive failures with no checkpoint progress. Giving up." | tee -a "$LOG"
        break
    fi

    echo "[launcher] $(date) exited $code; retrying in 30s (consecutive no-progress failures: $fails)." | tee -a "$LOG"
    sleep 30
done

echo "[launcher] $(date) launcher done." | tee -a "$LOG"
