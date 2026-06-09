#!/usr/bin/env python3
"""Test max batch size for SFT training on L4 23GB"""
import torch, os, sys, time
sys.path.insert(0, '.')
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from training.byte_tokenizer import ByteTokenizer

config = MominoMoEConfig()
model = MominoMoE(config)
model.gradient_checkpointing_enable()

device = torch.device('cuda')
model = model.to(device)

tok = ByteTokenizer()

prompt = 'Install the nginx web server on this system, configure it to serve a static site from /var/www, and enable HTTPS with a self-signed certificate.' * 10
resp = 'I will install and configure nginx for you.' * 5
full = prompt + resp
prompt_ids = tok.encode(prompt, max_length=2048)
full_ids = tok.encode(full, max_length=2048)
if len(full_ids) > 2048:
    full_ids = full_ids[:2048]
labels = [-100] * min(len(prompt_ids), 2048) + full_ids[min(len(prompt_ids), 2048):]
labels = labels[:2048]
full_ids = full_ids[:2048]
# Pad
full_ids = full_ids + [0] * (2048 - len(full_ids))
labels = labels + [-100] * (2048 - len(labels))

total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {torch.cuda.get_device_name(0)}, Total VRAM: {total_mem:.1f}GB')

for bs in [1, 2, 3, 4]:
    torch.cuda.reset_peak_memory_stats()
    try:
        batch_ids = torch.tensor([full_ids] * bs, device=device)
        batch_labels = torch.tensor([labels] * bs, device=device)
        
        # Forward + backward
        with torch.amp.autocast('cuda'):
            out = model(batch_ids, labels=batch_labels)
            loss = out['loss']
        loss.backward()
        model.zero_grad()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / 1e9
        free_mem = total_mem - peak
        print(f'batch={bs}: peak={peak:.2f}GB, free={free_mem:.2f}GB, OK')
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'batch={bs}: OOM')
            torch.cuda.empty_cache()
        else:
            print(f'batch={bs}: error={e}')
    except Exception as e:
        print(f'batch={bs}: error={e}')

# Now with optimizer states (Adam)
print('\n--- Testing with Adam optimizer states ---')
model = MominoMoE(config)
model.gradient_checkpointing_enable()
model = model.to(device)

for bs in [1, 2]:
    torch.cuda.reset_peak_memory_stats()
    try:
        batch_ids = torch.tensor([full_ids] * bs, device=device)
        batch_labels = torch.tensor([labels] * bs, device=device)
        
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Simulate a training step
        for _ in range(2):
            with torch.amp.autocast('cuda'):
                out = model(batch_ids, labels=batch_labels)
                loss = out['loss']
            loss.backward()
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / 1e9
        free_mem = total_mem - peak
        print(f'batch={bs} (Adam): peak={peak:.2f}GB, free={free_mem:.2f}GB, OK')
        del opt
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'batch={bs} (Adam): OOM')
            torch.cuda.empty_cache()
        else:
            print(f'batch={bs}: error={e}')
    except Exception as e:
        print(f'batch={bs}: error={e}')

print('\nDone.')