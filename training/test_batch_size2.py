#!/usr/bin/env python3
"""Clean test for max batch with Adam optimizer states on L4 23GB."""
import torch, os, sys, gc
sys.path.insert(0, '.')
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.byte_tokenizer import ByteTokenizer

torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()

config = MominoMoEConfig()
model = MominoMoE(config)
model.gradient_checkpointing_enable()

device = torch.device('cuda')
model = model.to(device)

tok = ByteTokenizer()

prompt = 'Install nginx.' * 50
resp = 'I will install nginx.' * 25
full = prompt + resp
prompt_ids = tok.encode(prompt, max_length=2048)
full_ids = tok.encode(full, max_length=2048)
if len(full_ids) > 2048:
    full_ids = full_ids[:2048]
labels = [-100] * min(len(prompt_ids), 2048) + full_ids[min(len(prompt_ids), 2048):]
labels = labels[:2048]
full_ids = full_ids[:2048]
full_ids = full_ids + [0] * (2048 - len(full_ids))
labels = labels + [-100] * (2048 - len(labels))

total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {total_mem:.1f}GB')
print(f'Seq len: 2048, seq filled: {len(full_ids) - full_ids.count(0)} tokens')

for bs in [1, 2]:
    print(f'\n--- batch={bs} with Adam, grad_accum=1 ---')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    
    # Fresh model to avoid stale state issues
    m = MominoMoE(config)
    m.gradient_checkpointing_enable()
    m = m.to(device)
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=5e-5)
    
    batch_ids = torch.tensor([full_ids] * bs, device=device)
    batch_labels = torch.tensor([labels] * bs, device=device)
    
    try:
        for step in range(5):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = m(batch_ids, labels=batch_labels)
                loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f'  step {step}: loss={loss.item():.4f}, peak_mem={peak:.2f}GB')
        
        peak = torch.cuda.max_memory_allocated() / 1e9
        free = total_mem - peak
        print(f'\n  Result: peak={peak:.2f}GB / {total_mem:.1f}GB, free={free:.2f}GB -- batch={bs} {"WORKS" if free > 1 else "MARGINAL"}')
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  OOM at batch={bs}')
        else:
            print(f'  Error: {e}')
        torch.cuda.empty_cache()
    
    del m, optimizer
    torch.cuda.empty_cache()
    gc.collect()

print('\nDone.')