#!/usr/bin/env python3
"""Single-shot batch test - starts completely fresh each time."""
import torch, os, sys, gc

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
assert batch_size in [1, 2, 4, 8]

sys.path.insert(0, '.')
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.train_utils import configure_optimizer, get_scaler

torch.cuda.empty_cache()
gc.collect()

config = MominoMoEConfig()
model = MominoMoE(config)
model.gradient_checkpointing_enable()
device = torch.device('cuda')
model = model.to(device)

# Simulate 2048-token sequences
seq_len = 2048
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_mem:.1f}GB')

torch.cuda.reset_peak_memory_stats()

# First check baseline (model only)
model_mem = torch.cuda.max_memory_allocated() / 1e9
print(f'After model load: {model_mem:.2f}GB')

# Create optimizer
torch.cuda.reset_peak_memory_stats()
optimizer = configure_optimizer(model, learning_rate=5e-5, weight_decay=0.1)
opt_mem = torch.cuda.max_memory_allocated() / 1e9
print(f'After optimizer: {opt_mem:.2f}GB (delta={opt_mem-model_mem:.2f}GB)')

# Create inputs using byte tokenizer for valid token IDs
from training.byte_tokenizer import ByteTokenizer
tok = ByteTokenizer()
text = "Install nginx and configure HTTPS. " * 100
tokens = tok.encode(text, max_length=seq_len)
if len(tokens) < seq_len:
    tokens = tokens + [0] * (seq_len - len(tokens))
tokens = torch.tensor(tokens[:seq_len], device=device)
prompt_len = len(tok.encode("Install nginx and configure HTTPS. " * 30, max_length=seq_len))

input_ids = tokens.unsqueeze(0).expand(batch_size, -1)
labels = tokens.clone().unsqueeze(0).expand(batch_size, -1)
labels[:, :min(prompt_len, seq_len)] = -100  # mask prompt tokens

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

try:
    # Training step
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        out = model(input_ids, labels=labels)
        loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    
    peak = torch.cuda.max_memory_allocated() / 1e9
    free = total_mem - peak
    print(f'Training step: peak={peak:.2f}GB, free={free:.2f}GB, loss={loss.item():.4f}')
    
    if free > 0.5:
        print(f'RESULT: batch={batch_size} WORKS (free={free:.1f}GB)')
    else:
        print(f'RESULT: batch={batch_size} MARGINAL (free={free:.1f}GB)')
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print(f'RESULT: batch={batch_size} OOM')
    else:
        print(f'RESULT: batch={batch_size} ERROR: {e}')
except Exception as e:
    print(f'RESULT: batch={batch_size} ERROR: {e}')