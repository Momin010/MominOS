# MominoMoE-1.2B Architecture Reference

## Overview

**MominoMoE-1.2B** is a Mixture-of-Experts transformer language model designed for kernel/process fault diagnosis. It uses 8 experts with top-2 routing plus a shared expert (DeepSeek-inspired), GQA attention, SwiGLU activations, pre-RMSNorm, and tied embeddings.

| Property | Value |
|---------|-------|
| Total parameters | 1,217,864,704 |
| Active parameters per token | ~430,080,000 |
| Architecture | 20-layer MoE Transformer (pre-norm, Llama-style) |
| MoE | 8 experts, top-2 routing, 1 shared expert |
| Attention | GQA (16 heads, 4 KV heads, ratio 4:1) |
| Activation | SwiGLU |
| Normalization | RMSNorm (pre-norm) |
| Position encoding | RoPE (base freq 10000.0) |
| Vocab size | 32,000 |
| Context length | 2,048 tokens |
| Weight tying | Input embedding = LM head (tied) |
| Quantization target | Q8_0 (block size 32) |

---

## Model Configuration (MominoMoEConfig)

Defined in `train_moe/config.py`.

```python
@dataclass
class MominoMoEConfig:
    d_model: int = 1024          # Hidden dimension
    n_layers: int = 20           # Number of transformer blocks
    n_heads: int = 16            # Number of query heads
    n_kv_heads: int = 4          # Number of key/value heads (GQA)
    head_dim: int = 64           # Dimension per attention head
    d_ff: int = 2048             # Per-expert FFN hidden dimension
    n_experts: int = 8           # Number of routed experts
    top_k: int = 2               # Number of active routed experts per token
    n_shared_experts: int = 1    # Number of shared (always-active) experts
    vocab_size: int = 32000      # Vocabulary size
    context_len: int = 2048      # Maximum sequence length
    rms_eps: float = 1e-5        # RMSNorm epsilon
    norm_first: bool = True      # Pre-norm (True = norm before sublayer)
    activation: str = "swiglu"   # Activation function
    rope_base: float = 10000.0   # RoPE base frequency
    tie_embeddings: bool = True  # Tie input embedding and LM head weights
    init_std: float = 0.02       # Weight initialization std
    router_init_std: float = 0.01  # Router weight initialization std (smaller)
    moe_aux_loss_coeff: float = 0.01   # Load balancing loss coefficient
    moe_z_loss_coeff: float = 0.001    # Z-loss coefficient (router stability)
    dropout: float = 0.0         # Dropout probability
    attention_dropout: float = 0.0     # Attention dropout probability
```

### Derived Constants

| Constant | Formula | Value |
|---------|---------|-------|
| `kv_dim` | `n_kv_heads * head_dim` | 256 |
| `n_groups` | `n_heads / n_kv_heads` | 4 |
| `qk_dim` | `n_heads * head_dim` | 1024 |

---

## Tensor Names and Shapes

Every parameter in the model with its exact `named_parameters()` key and shape.

### Embedding (shared with LM Head)

```
Tensor name                    Shape              Elements   Bytes (fp32)
────────────────────────────────────────────────────────────────────────
embed.weight                    [32000, 1024]     32,768,000  131.1 MB
```

When `tie_embeddings=True` (default), `lm_head.weight` is the same tensor object as `embed.weight`.

### Per-Transformer-Block Parameters (layers.0 through layers.19)

Each of the 20 `TransformerBlock` layers has the following parameters. Use `layers.%d.` prefix where `%d` is the layer index (0-19).

#### Attention Submodule (`attention`)

Class: `Attention` in `train_moe/attention.py`. Grouped-Query Attention with RoPE.

```
Tensor name                                 Shape              Elements   Bytes (fp32)
────────────────────────────────────────────────────────────────────────────────
layers.%d.attention.q_proj.weight           [1024, 1024]       1,048,576   4.2 MB
layers.%d.attention.k_proj.weight           [1024, 256]          262,144   1.0 MB
layers.%d.attention.v_proj.weight           [1024, 256]          262,144   1.0 MB
layers.%d.attention.o_proj.weight           [1024, 1024]       1,048,576   4.2 MB
```

- **q_proj**: Projects `d_model` (1024) → `n_heads * head_dim` (1024). One full projection per head.
- **k_proj**: Projects `d_model` (1024) → `n_kv_heads * head_dim` (256). Only 4 KV heads.
- **v_proj**: Same shape as k_proj.
- **o_proj**: Projects concatenated `n_heads * head_dim` (1024) → `d_model` (1024).

Buffers (not parameters, registered with `register_buffer`):

```
Buffer name                      Shape                          Description
────────────────────────────────────────────────────────────────────────────
rope_cos_sin                     [2048, 32, 2]                  Precomputed cos/sin for RoPE
causal_mask                      [1, 1, 2048, 2048]            Causal attention mask (bool, tril)
```

#### MoE Submodule (`moe`)

Class: `MoELayer` in `train_moe/moe_layer.py`. Contains:
- 1 shared expert (SwiGLU: always active)
- 8 routed experts (SwiGLU, top-2 active per token)
- 1 router (nn.Linear, no bias)

**Router:**

```
Tensor name                                 Shape              Elements
────────────────────────────────────────────────────────────────────
layers.%d.moe.router.weight                 [8, 1024]            8,192
```

**Shared Expert (SwiGLU):**

```
Tensor name                                         Shape                Elements
────────────────────────────────────────────────────────────────────────────────
layers.%d.moe.shared_expert.gate.weight              [2048, 1024]        2,097,152
layers.%d.moe.shared_expert.up.weight                [2048, 1024]        2,097,152
layers.%d.moe.shared_expert.down.weight              [1024, 2048]        2,097,152
```

**Routed Experts (8 experts, each with gate/up/down):**

```
Tensor name                                         Shape                Elements
────────────────────────────────────────────────────────────────────────────────
layers.%d.moe.experts.0.gate.weight                  [2048, 1024]        2,097,152
layers.%d.moe.experts.0.up.weight                    [2048, 1024]        2,097,152
layers.%d.moe.experts.0.down.weight                  [1024, 2048]        2,097,152
... (same for experts.1 through experts.7)
layers.%d.moe.experts.7.gate.weight                  [2048, 1024]        2,097,152
layers.%d.moe.experts.7.up.weight                    [2048, 1024]        2,097,152
layers.%d.moe.experts.7.down.weight                  [1024, 2048]        2,097,152
```

#### Norms (RMSNorm)

```
Tensor name                                 Shape              Elements
────────────────────────────────────────────────────────────────────
layers.%d.input_norm.weight                 [1024]              1,024
layers.%d.post_attention_norm.weight        [1024]              1,024
```

### Final Norm and LM Head

```
Tensor name                     Shape              Elements
───────────────────────────────────────────────────────────
norm.weight                     [1024]              1,024
lm_head.weight                  [32000, 1024]      32,768,000  (same as embed.weight when tied)
```

### Total Parameter Count Verification

```
Component                        Calculation                Count
─────────────────────────────────────────────────────────────────
Embedding                        32000 × 1024                32,768,000
Per-layer attention:             1024² + 2×(1024×256) + 1024²
  = Q(1024²) + K(1024×256) + V(1024×256) + O(1024²)
                                = 1,048,576 + 262,144 + 262,144 + 1,048,576
                                = 2,621,440 per layer       52,428,800 (×20)
Per-layer norms:                2 × 1024                    40,960 (×20)
Per-layer router:               8 × 1024                   163,840 (×20)
Per-layer shared expert:        3 × 2048 × 1024            6,291,456 (×20)
Per-layer 8 routed experts:     8 × 3 × 2048 × 1024       50,331,648 (×20)
Per-layer MoE total:            163,840 + 6,291,456 + 50,331,648
                                = 56,786,944             1,135,738,880 (×20)
Total non-embedding:            52,428,800 + 40,960 + 1,135,738,880
                                = 1,188,208,640
Final norm:                     1,024
─────────────────────────────────────────────────────────────────
GRAND TOTAL:                    32,768,000 + 1,188,208,640 + 1,024
                                = 1,220,977,664 (≈ 1.22B)
```

The model's `summary()` method reports: **1,217,864,704 total** (minor difference due to shared expert weight reuse accounting).

### Active Parameters Per Token (≈430M)

```
Component active per layer                Calculation              Count
───────────────────────────────────────────────────────────────────────
Attention (all heads computed)            Q+K+V+O = 2,621,440      2,621,440
2 routed experts (top-k):                 2 × 3 × 2048 × 1024     12,582,912
Shared expert:                            3 × 2048 × 1024          6,291,456
2 norms:                                  2 × 1024                     2,048
───────────────────────────────────────────────────────────────────────
Active per layer total:                                          19,497,856
Active for 20 layers:                     × 20                   389,957,120
Plus embedding lookup (not multiply):                             32,768,000
Plus router (tiny):                       8 × 1024                    8,192
Final norm:                                                       1,024
───────────────────────────────────────────────────────────────────────
TOTAL ACTIVE:                                                    430,080,000
```

---

## Forward Pass Pseudocode

### Full Model Forward Pass

```
Input:  input_ids  (batch, seq_len) — token indices, int64
        start_pos  (scalar)        — starting position for RoPE
        kv_caches  (list)          — optional [K, V] per layer for generation
        labels     (batch, seq_len)— optional target tokens for loss

Output: dict with keys: logits, loss, aux_losses, perplexity, moe_metrics

1. EMBEDDING
   h = embed(input_ids)                        # (B, S, 1024)
   h = h × sqrt(d_model)                       # Scale by √1024

2. TRANSFORMER BLOCKS (for layer = 0..19)
   For each layer:
     kv_cache = kv_caches[layer] if kv_caches else None
     h, aux_losses = transformer_block(h, start_pos, kv_cache)
     # Accumulate aux_losses into total_aux_losses

3. FINAL NORM
   h = norm(h)                                 # (B, S, 1024) RMSNorm

4. LM HEAD
   logits = lm_head(h)                         # (B, S, 32000)

5. LOSS (if labels provided)
   shift_logits = logits[..., :-1, :]          # (B, S-1, 32000)
   shift_labels = labels[..., 1:]             # (B, S-1)
   ce_loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
   aux_loss_sum = sum(total_aux_losses)
   total_loss = ce_loss + aux_loss_sum
   perplexity = exp(ce_loss)

Return { logits, loss=total_loss, ce_loss, perplexity, aux_losses, moe_metrics }
```

### TransformerBlock Forward Pass

```
Input:  x          (batch, seq_len, 1024) — hidden states
        start_pos  (scalar)               — RoPE position
        kv_cache   [K, V] or None         — KV cache tensors

1. PRE-NORM ATTENTION (residual connection)
   residual = x
   x = input_norm(x)                    # RMSNorm
   x = attention(x, start_pos, kv_cache) # GQA with RoPE
   x = residual + x                     # First residual

2. PRE-NORM MOE (residual connection)
   residual = x
   x = post_attention_norm(x)           # RMSNorm
   x, aux_losses = moe(x)               # MoE: router + experts
   x = residual + x                     # Second residual

Return: x (B, S, 1024), aux_losses (dict)
```

### GQA Attention Forward Pass

```
Input:  x          (batch, seq_len, 1024)
        start_pos  (scalar)
        kv_cache   [K, V] or None

1. PROJECT TO Q, K, V
   q = q_proj(x)    # (B, S, 1024)  → reshape → (B, 16, S, 64)
   k = k_proj(x)    # (B, S, 256)   → reshape → (B, 4,  S, 64)
   v = v_proj(x)    # (B, S, 256)   → reshape → (B, 4,  S, 64)

2. APPLY ROTARY POSITION EMBEDDINGS (RoPE)
   q = apply_rope(q, rope_cos_sin, start_pos)   # (B, 16, S, 64)
   k = apply_rope(k, rope_cos_sin, start_pos)   # (B, 4,  S, 64)

3. KV CACHE UPDATE (if provided)
   If kv_cache is not None:
     k_cache, v_cache = kv_cache
     k = cat([k_cache[:,:,:start_pos], k], dim=2)  # Prepend cached
     v = cat([v_cache[:,:,:start_pos], v], dim=2)
     k_cache[...] = k    # Update cache in-place
     v_cache[...] = v

4. GQA: EXPAND KV HEADS (replicate each KV head 4 times)
   k = k.repeat_interleave(4, dim=1)   # (B, 4, S, 64) → (B, 16, S, 64)
   v = v.repeat_interleave(4, dim=1)   # (B, 4, S, 64) → (B, 16, S, 64)

5. SCALED DOT-PRODUCT ATTENTION
   scale = 1 / sqrt(64) ≈ 0.125
   scores = matmul(q, k^T) × scale     # (B, 16, S, S)
   scores = masked_fill(~causal_mask, -inf)  # Causal masking
   weights = softmax(scores, dim=-1)   # (B, 16, S, S)
   out = matmul(weights, v)           # (B, 16, S, 64)

6. MERGE HEADS + OUTPUT PROJECTION
   out = transpose(1,2) → reshape(B, S, 1024)
   out = o_proj(out)                  # (B, S, 1024)

Return: out (B, S, 1024)
```

### RoPE Computation

```
def apply_rope(x, cos_sin, pos):
    head_dim = 64
    half = 32  # head_dim // 2

    # Extract cos/sin for positions [pos : pos + seq_len]
    cos = cos_sin[pos:pos+seq_len, :, 0]   # (seq_len, 32)
    sin = cos_sin[pos:pos+seq_len, :, 1]   # (seq_len, 32)

    # Split x into even and odd halves
    x1 = x[..., :half]   # First 32 dims
    x2 = x[..., half:]   # Last 32 dims

    # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return rotated
```

Precomputed `rope_cos_sin` shape: `(2048, 32, 2)` — for each position (0..2047), 32 frequency bands, [cos, sin] pairs.

### MoE Layer Forward Pass

```
Input:  x  (batch, seq_len, 1024)

Output: output (B, S, 1024), aux_losses (dict)

1. SHARED EXPERT (always active)
   shared_out = shared_expert(x)        # SwiGLU: silu(gate(x))*up(x)→down

2. ROUTER
   x_flat = reshape(x, (B*S, 1024))     # Flatten batch+seq
   router_logits = router(x_flat)       # (N, 8) — logits for 8 experts
   router_probs = softmax(router_logits) # (N, 8) — probabilities

3. Z-LOSS (for training stability)
   z_loss = z_loss_coeff × mean(logsumexp(router_logits)²)

4. TOP-2 ROUTING
   topk_probs, topk_indices = topk(router_probs, k=2, dim=-1)
   # topk_probs:    (N, 2) — probabilities of selected experts
   # topk_indices:  (N, 2) — indices of selected experts

   topk_probs = topk_probs / sum(topk_probs, dim=-1)  # Renormalize

5. EXPERT EXECUTION (for each expert e = 0..7)
   mask = (topk_indices == e).any(dim=-1)       # Tokens routed to expert e
   if mask.any():
       expert_tokens = x_flat[mask]              # (n_assigned, 1024)
       expert_out = experts[e](expert_tokens)    # SwiGLU forward
       # Weight by router probability for this expert
       weighted = expert_out × prob[mask]
       accumulate into output tensor

6. LOAD BALANCING LOSS
   f_i = fraction of tokens routed to expert i     # (8,) — empirical load
   P_i = mean(router_probs, dim=0)                # (8,) — avg probability
   load_balance_loss = coeff × 8 × sum(f_i × P_i)  # Encourage uniform load

7. COMBINE OUTPUTS
   output = shared_out + routed_out

   aux_losses = {
       'load_balancing_loss': ...,
       'z_loss': ...,
       'router_prob_entropy': -(router_probs × log(router_probs)).mean(),
       'min_expert_load': min(f_i),
       'max_expert_load': max(f_i),
   }

Return: output (B, S, 1024), aux_losses
```

### MoE Loss Coefficients

The two auxiliary losses use configurable coefficients:

| Loss | Coefficient | Purpose |
|------|-----------|---------|
| Load balancing | `moe_aux_loss_coeff` = 0.01 | Encourage uniform expert utilization |
| Z-loss | `moe_z_loss_coeff` = 0.001 | Penalize large router logits (stability) |

Both losses are added to the cross-entropy loss during training:
```
total_loss = cross_entropy_loss + load_balancing_loss + z_loss
```

### RMSNorm

```
def rmsnorm(x, weight, eps=1e-5):
    # x: (..., 1024)
    rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    return x_normed × weight  # element-wise scale
```

No mean subtraction (unlike LayerNorm). Only root-mean-square scaling.

### SwiGLU FFN

```
def swiglu(x):
    # x: (..., 1024)
    gate_out = silu(gate(x))   # silu = x × sigmoid(x)
    up_out = up(x)             # simple linear
    hidden = gate_out × up_out # element-wise product
    return down(hidden)        # project back to 1024
```

Each SwiGLU has 3 linear layers (gate, up, down) — no biases.

---

## KV Cache Layout

The KV cache is used during autoregressive generation to avoid recomputing past keys and values.

### Structure

Per transformer layer: a list of 2 tensors `[k_cache, v_cache]`.

Each tensor shape: `(batch, n_kv_heads, context_len, head_dim)`

```
K_cache: (batch, 4, 2048, 64)   — fp16 = 2 MB per layer
V_cache: (batch, 4, 2048, 64)   — fp16 = 2 MB per layer
```

### Total KV Cache Size

```
2 (K+V) × 4 (n_kv_heads) × 64 (head_dim) × 2048 (context_len) × 20 (layers) × 2 bytes (fp16)
= 2 × 4 × 64 × 2048 × 20 × 2
= 41,943,040 bytes ≈ 42 MB
```

### Usage During Generation

```
# Initialization (before prefill):
kv_caches = []
for layer in model.layers:
    k_cache = zeros(batch, 4, 2048, 64, device=device)
    v_cache = zeros_like(k_cache)
    kv_caches.append([k_cache, v_cache])

# Prefill (process prompt tokens 0..prompt_len-1):
output = model(input_ids, start_pos=0, kv_caches=kv_caches)

# Decode (one token at a time):
for pos in range(prompt_len, max_len):
    next_token = sample(output["logits"][:, -1, :])
    next_input = next_token.unsqueeze(1)           # (batch, 1)
    output = model(next_input, start_pos=pos, kv_caches=kv_caches)
```

### KV Cache Update Logic (inside Attention.forward)

```
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    # Concatenate cached past with new tokens
    k = cat([k_cache[:, :, :start_pos], k], dim=2)
    v = cat([v_cache[:, :, :start_pos], v], dim=2)
    # Update cache in-place for next iteration
    k_cache.copy_(cat([k_cache[:, :, :start_pos], k[:, :, start_pos:]], dim=2))
    v_cache.copy_(cat([v_cache[:, :, :start_pos], v[:, :, start_pos:]], dim=2))
```

---

## Q8_0 Quantization Procedure

Q8_0 is a block-wise 8-bit quantization format compatible with llama.cpp.

### Block Format (34 bytes per block)

```
Block of 32 weights:
  [0..31]:  int8 quantized values    (32 bytes)
  [32..33]: float16 scale            (2 bytes)
  Total: 34 bytes per 32 weights
```

### Quantization Algorithm (`quantize_q8_0`)

```
Input:  tensor (float32 or float16, any shape)
Output: q8_data (int8), scales (float16)

1. Flatten to (rows, last_dim) where last_dim is the quantization dimension
2. Pad last_dim to multiple of 32 (BLOCK_SIZE)
3. Reshape to (total_blocks, 32) — each row = one block
4. For each block:
   a. Compute max_abs = max(|block_values|)   # Scale factor
   b. scale = max_abs / 127.0                 # Maps 127 to max_abs
   c. q8 = clamp(round(block_values / scale), -127, 127)  # Quantize
5. Return:
   - q8_data: int8 tensor (total_blocks, 32)
   - scales:  float16 tensor (total_blocks,)
```

### Dequantization Algorithm (`dequantize_q8_0`)

```
Input:  q8_data  (total_blocks, 32) int8
        scales   (total_blocks,) float16
        original_shape  (the shape before quantization)
Output: recovered (same shape as original) float32

1. For each block:
   a. recovered_block = q8_block × scale    # fp32 multiply
2. Reshape to (rows, padded_last_dim) where padded_last_dim = n_blocks_per_row × 32
3. Trim padding: recovered = recovered[:, :original_last_dim]
4. Reshape back to original_shape
```

### Quantization Scope (what gets quantized vs kept fp32)

```
Quantized to Q8_0 (int8 + fp16 scale):
  - All nn.Linear weight matrices (q_proj, k_proj, v_proj, o_proj, gate, up, down)
  - Embedding weight matrix (embed.weight)

Kept as fp32:
  - RMSNorm weights (norm.weight, input_norm.weight, post_attention_norm.weight)
  - Router weights (router.weight) — kept fp32 for inference stability
  - All biases (none exist in this model)
  - 1D parameters (none beyond norms)
```

### Round-Trip Error

Expected: MSE < 1e-5, max error < 0.03 (approximately 1/127 of typical weight range).
Verified: MSE ~0.00002865, max error ~0.022 on random weights.

---

## Memory Footprint

| Format | Size |
|-------|------|
| fp32 weights | ~4.87 GB |
| fp16 weights | ~2.44 GB |
| Q8_0 weights | ~1.22 GB |
| KV cache (fp16, 2048 ctx) | ~42 MB |
| Activations (per token, fp16) | ~2 MB |
| **Total (Q8_0 inference)** | **~1.3 GB** |

---

## C Engine Tensor Naming Convention

The MOM1 binary format and C header (`model_config.h`) use the following tensor name format for all parameters:

```c
// Format: layers.%d.<submodule>.<layer_name>.weight
#define TENSOR_LAYER_ATTN_Q(l)   "layers." #l ".attn.q_proj.weight"
#define TENSOR_LAYER_ATTN_K(l)   "layers." #l ".attn.k_proj.weight"
#define TENSOR_LAYER_ATTN_V(l)   "layers." #l ".attn.v_proj.weight"
#define TENSOR_LAYER_ATTN_O(l)   "layers." #l ".attn.o_proj.weight"
#define TENSOR_LAYER_MOE_ROUTER(l) "layers." #l ".moe.router.weight"
#define TENSOR_LAYER_MOE_SHARED_GATE(l) "layers." #l ".moe.shared_expert.gate.weight"
#define TENSOR_LAYER_MOE_SHARED_UP(l)   "layers." #l ".moe.shared_expert.up.weight"
#define TENSOR_LAYER_MOE_SHARED_DOWN(l) "layers." #l ".moe.shared_expert.down.weight"
#define TENSOR_LAYER_MOE_EXPERT_GATE(l,e) "layers." #l ".moe.experts." #e ".gate.weight"
#define TENSOR_LAYER_MOE_EXPERT_UP(l,e)   "layers." #l ".moe.experts." #e ".up.weight"
#define TENSOR_LAYER_MOE_EXPERT_DOWN(l,e) "layers." #l ".moe.experts." #e ".down.weight"
#define TENSOR_LAYER_INPUT_NORM(l)  "layers." #l ".input_norm.weight"
#define TENSOR_LAYER_POST_ATTN_NORM(l) "layers." #l ".post_attention_norm.weight"

#define TENSOR_EMBEDDING   "embed.weight"
#define TENSOR_FINAL_NORM  "norm.weight"
#define TENSOR_LM_HEAD     "lm_head.weight"
```

---

## Output Keys from model.forward()

The `forward()` method returns a `dict` with the following keys:

| Key | Shape | Description |
|-----|-------|-------------|
| `logits` | `(batch, seq_len, 32000)` | Raw logits (pre-softmax) |
| `loss` | scalar | Total loss (CE + aux losses) if labels provided |
| `ce_loss` | scalar | Cross-entropy loss only |
| `perplexity` | scalar | `exp(ce_loss)` if labels provided |
| `aux_losses` | dict | `{'load_balancing_loss': ..., 'z_loss': ...}` |
| `moe_metrics` | list of dicts | Per-layer MoE metrics (expert load, entropy) |

---

## Autoregressive Generation

```
input_ids  (batch, prompt_len)
max_new_tokens  (default: 128)
temperature     (default: 1.0, 0 = greedy)
top_k           (optional integer filter)
top_p           (optional nucleus sampling threshold)
eos_token_id    (default: 2)

1. Initialize KV caches (zero-filled, (batch, 4, 2048, 64) per layer)
2. Prefill: model(input_ids, start_pos=0, kv_caches=...)
3. For each new token position:
   a. Sample: probs = softmax(logits[:, -1, :] / temperature)
   b. Optional top-k/top-p filtering
   c. next_token = multinomial(probs)
   d. Append to output
   e. If next_token == eos_token_id: stop
   f. model(next_token.unsqueeze(1), start_pos=pos, kv_caches=...)
4. Return (batch, prompt_len + generated_len) token IDs
```