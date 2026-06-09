# MominOS AI Subsystem — Implementation Spec (handoff)

This document specifies the embedded LLM subsystem for MominOS: an on-device
Mixture-of-Experts model that automatically diagnoses and proposes fixes when a
process or the kernel faults. It is written so a separate engineer/agent can
execute it without further context.

**Read this whole file before writing code.** The single biggest failure mode is
jumping straight to "write an LLM" while the OS still lacks floating point, RAM,
a disk, a filesystem, and a heap. Those are hard blockers, listed in Phase 0.

---

## 0. Current state of MominOS (what you're building on)

Already working (do not rebuild):
- BIOS boot -> stage 2 -> long mode (64-bit), GDT, 4-level paging, first 2 MB identity-mapped.
- PMM: bitmap physical allocator, `pmm_alloc()/pmm_free()`, ~128 MB tracked.
- IDT: 256 vectors, C dispatcher, clean fault reports (vector, RIP, error code).
- PIC remapped (0x20/0x28), PIT @100 Hz tick, PS/2 keyboard, EOI + `sti` working.
- VMM: in progress (`vmm_map/unmap`, direct-map of physical RAM).
- Serial (COM1) + VGA text drivers.

NOT present yet (and required — see Phase 0):
- Floating point / SIMD in the kernel (currently compiled `-mno-sse -mno-mmx`).
- A kernel heap (`kmalloc/kfree`).
- A scheduler / threads.
- A disk driver and a filesystem.
- Any libm (no `expf`, `sqrtf`, etc.).
- Boot configured for multi-GB RAM.

The model is ~1-2 GB on disk. None of the above is optional.

---

## 1. Target definition ("done" looks like this)

A process segfaults. Within a few seconds, the console prints a human-readable
diagnosis and a concrete suggested fix, generated entirely on-device by the
embedded model. No network. Example:

```
[AI] process 'webserver' (pid 7) faulted: #PF write to 0x0, rip=0x401a2c
[AI] diagnosis: null pointer dereference in request handler; 'conn' was never
     assigned because accept() returned -1 (fd table exhausted).
[AI] suggested fix: raise the fd limit, or check accept()'s return before use.
[AI] (apply suggested config change? y/N)
```

Model spec (target): ~1-2 B total params, 8 experts, top-2 routing
(~250-500 M active params/token), int8 weights, 4-8 tok/s on CPU.

---

## 2. Architecture — three layers

```
+-------------------------------------------------------------+
| Layer 3: Error-diagnosis daemon                              |
|   fault hook -> context builder -> inference -> action gate  |
+-------------------------------------------------------------+
| Layer 2: Model + tokenizer                                  |
|   on-disk format, weights, BPE tokenizer, prompt template   |
+-------------------------------------------------------------+
| Layer 1: Inference engine (tensor kernels + transformer)   |
|   quantized matmul, attention+KV cache, MoE router, sampler |
+-------------------------------------------------------------+
| Layer 0: OS prerequisites (SSE, heap, disk, FS, threads)   |
+-------------------------------------------------------------+
```

Build bottom-up. Each layer has acceptance criteria; do not advance until met.

---

## Phase 0 — OS prerequisites (blockers)

### 0.1 Enable SSE/SSE2 (mandatory), AVX2 (strongly recommended)
The kernel is currently built with `-mno-sse`. Inference needs float math and,
for the perf target, SIMD. In `kernel_entry` (long mode, before `kmain`):

```asm
; enable x87 + SSE
mov rax, cr0
and ax, 0xFFFB          ; clear CR0.EM (bit 2)
or  ax, 0x0002          ; set CR0.MP (bit 1)
mov cr0, rax
mov rax, cr4
or  rax, (1<<9)|(1<<10) ; CR4.OSFXSR | CR4.OSXMMEXCPT
mov cr4, rax
```
For AVX (optional, big speedup): also set `CR4.OSXSAVE` (bit 18), then `xsetbv`
XCR0 = bits 0,1,2 (x87|SSE|AVX). Verify support with CPUID first.
Then change the build: drop `-mno-sse -mno-sse2` for the inference translation
units (keep `-mno-red-zone`). Confirm with CPUID before using AVX2/FMA paths.

Acceptance: a kernel function that does `float a=1.5f; a*=2;` and SIMD vector add
runs without #UD/#NM.

### 0.2 Boot with real RAM
Run QEMU with `-m 4G` (or 8G). Confirm PMM reports multi-GB free. The PMM bitmap
must cover the configured RAM (current bitmap covers 4 GB; bump if you go higher).

### 0.3 Kernel heap (`kmalloc/kfree`)
Build on VMM + PMM. Start simple: a linked-list/bump-then-free allocator over a
large virtual region backed by `pmm_alloc` pages mapped via `vmm_map`. Must
handle large (multi-hundred-MB) allocations for weights and KV cache.

Acceptance: allocate/free 256 MB in chunks without corruption; alignment to 64 B
available (for SIMD loads).

### 0.4 Threads / scheduler (minimum: kernel threads)
Inference is long-running; it must not freeze the tick/keyboard. Minimum viable:
cooperative or PIT-preemptive kernel threads with context switch (save/restore
GPRs + RSP, and FXSAVE/XSAVE area for SIMD state). Full user processes can come
later, but the AI daemon needs to be its own schedulable thread.

Acceptance: two kernel threads alternate; the AI thread can run a long loop while
the heartbeat dot keeps printing.

### 0.5 Disk driver + filesystem (to load the 1-2 GB model)
Pick the simplest path that works under QEMU:
- Driver: `virtio-blk` (cleanest under QEMU) or ATA PIO (simplest to write, slow).
- Storage: you can skip a real FS initially by placing the model on a raw
  partition/second disk and reading it by LBA. For a real system, implement a
  read-only FAT32 or ext2 reader.
Recommendation: ATA PIO read + raw-LBA model blob first (fastest to get inference
running), then upgrade to virtio-blk + a real FS.

Acceptance: read a known 1 GB blob from disk into a heap buffer and checksum it.

### 0.6 libm-lite
You need at least: `expf`, `sqrtf` (use the SSE `sqrtss` instruction), `sinf`,
`cosf` (for RoPE — or precompute tables), and integer round. Port `expf` from a
known-correct minimax implementation (musl/cephes style: range-reduce x to
`k*ln2 + r`, evaluate a poly for `e^r`, scale by `2^k`). Do NOT hand-roll exp
without testing against reference values — softmax accuracy depends on it.

Acceptance: `expf`/`sqrtf` match reference to <1e-5 over the input ranges used.

---

## Phase 1 — Inference engine

### 1.1 On-disk model format (define this exactly — see Section A below)
A flat, mmap-friendly blob: header (config) + tensor table + tensor data.
Quantization: Q8_0-style blocks (32 int8 weights + one fp16 scale per block).
This is simple, proven (llama.cpp), and accurate enough.

### 1.2 Tensor kernels (the performance-critical core)
Implement and unit-test each in isolation against reference outputs:
- `rmsnorm(x, weight, eps)`: `x_i * rsqrt(mean(x^2)+eps) * w_i`.
- `matmul_q8`(W[out,in] quantized, x[in] fp32) -> y[out] fp32:
  - v0 (correctness): dequantize each Q8_0 block to fp32, fp32 dot product.
  - v1 (speed): quantize activations per-row to int8, do int32 dot via AVX2
    (`vpdpbusd` if VNNI, else `vpmaddubsw`+`vpmaddwd`), then dequant with
    `act_scale * weight_scale`. This is where the tok/s target is won or lost.
- `rope(q, k, pos)`: rotate dim pairs by `pos / base^(2i/d)`, base=10000.
- `attention`: scaled dot-product, causal mask, stable softmax, weighted sum of
  V; support GQA (n_kv_heads < n_heads -> repeat KV heads).
- `swiglu_ffn(x)`: `down( silu(gate(x)) * up(x) )`, `silu(x)=x*sigmoid(x)`.
- `softmax`: subtract row max before exp (numerical stability).

### 1.3 MoE block
Per token, per layer:
1. `router_logits = matmul(W_router[n_experts, d_model], x)`.
2. `g = softmax(router_logits)`.
3. Select top-2 experts; renormalize their two gate weights to sum to 1.
4. `y = g0*expert[i0].ffn(x) + g1*expert[i1].ffn(x)` (each expert is a SwiGLU FFN).
5. Only the 2 selected experts are computed/touched — this is the whole point of
   MoE: ~250-500 M active params instead of 1-2 B.

### 1.4 Transformer forward pass
Per layer: `x += attention(rmsnorm(x))`; `x += moe(rmsnorm(x))`. Final RMSNorm,
then `logits = matmul(lm_head, x)`. Standard Llama/Mixtral-style pre-norm.

### 1.5 KV cache
Allocate `[n_layers][context_len][n_kv_heads*head_dim]` for K and V (fp16 to
save RAM). Append per generated token; attention reads the whole prefix.

### 1.6 Sampler + generation loop
Greedy (argmax) first for determinism while debugging. Then temperature + top-p.
Loop: feed prompt tokens (prefill), then autoregressively generate until EOS or
max tokens.

Acceptance: feed a fixed prompt, get coherent text matching a reference run of
the same weights in PyTorch/llama.cpp (logits of token 0 should match to ~1e-2;
greedy decode should produce the same sequence).

### 1.7 Golden-intermediate validation harness (build this EARLY, not last)
End-to-end "greedy decode matches reference" tells you it's broken but not WHERE,
and a from-scratch 20+ layer transformer will be broken somewhere. Before trusting
the full forward pass:
1. In the reference runtime (PyTorch/llama.cpp), run ONE fixed input with hooks
   that dump every intermediate tensor to disk: post-embedding, post each RMSNorm,
   post-attention, post each matmul, post-MoE, final logits.
2. In MominOS, dump the same intermediates and diff against the golden tensors.
3. Find the FIRST op that diverges — that is your bug, localized.
This harness is the single highest-leverage thing in this spec. Treat it as part
of Phase 1, not a final gate.

### 1.8 Do not trust this doc's architecture defaults — derive them from the checkpoint
"Llama/Mixtral-style" is an assumption, not a guarantee. Real checkpoints vary:
attention/FFN bias terms, QK-norm, activation choice, RoPE theta, GQA ratio, tied
vs untied embeddings, norm placement. Read the chosen model's actual config and
match the forward pass to IT. A correct engine + a wrong convention = silent
garbage that is miserable to debug.

---

## Phase 2 — Tokenizer

Implement byte-level BPE (GPT-2/Llama style):
- Ship `vocab` (token -> id) and `merges` (ordered merge rules) as data files.
- Encode: bytes -> initial symbols -> repeatedly apply the highest-priority merge
  present until none apply.
- Decode: ids -> byte sequences -> UTF-8.
This is fiddly but bounded (~300 lines C). Test round-trip encode/decode on known
strings against the reference tokenizer.

Acceptance: `decode(encode(s)) == s` for ASCII + UTF-8 samples; token ids match
the reference tokenizer exactly.

---

## Phase 3 — The model itself (decision required)

There is no off-the-shelf 1-2 B / 8-expert / top-2 MoE that drops in cleanly, so
pick a path:

- **Path A (fastest to a demo): bootstrap with a small DENSE model.** Convert an
  existing ~1 B dense model (e.g. a small Llama-architecture model) to the Q8_0
  format and validate the engine end-to-end. The engine is written MoE-capable;
  a dense model is just "1 expert, top-1". This de-risks Layers 1-2 before you
  ever touch MoE weights.
- **Path B (the vision): a custom small MoE specialized for error diagnosis.**
  Distill from a large teacher: generate a synthetic corpus of
  (fault context -> diagnosis -> fix) pairs (the teacher can be Claude),
  fine-tune/distill a small MoE, quantize to int8, convert to the format. This
  keeps the model tiny because it's specialized, not general-purpose.

Recommendation: do **A**, prove the whole stack works, then do **B**. Treat model
training as a parallel ML workstream; the OS engine only needs the format frozen.

Acceptance (A): engine produces the same greedy decode as the reference runtime.
Acceptance (B): on a held-out set of real MominOS faults, diagnoses are correct.

---

## Phase 4 — Error-diagnosis daemon (the actual product)

### 4.0 Scope v1 to KERNEL faults — process faults wait on the process stack
Process-crash diagnosis (pid, argv, process-exit path) needs a scheduler, ELF
loader, and userspace — none of which exist yet. Do NOT block Phase 4 on them.
**v1 diagnoses kernel exceptions only**, which are hookable TODAY via the existing
IDT C dispatcher. This gives a demoable milestone now; add process-fault capture
later once the process stack lands.

### 4.1 Fault capture
Hook the existing IDT C dispatcher (v1) and the future process-exit path (v2). On
a CPU exception (or, later, abnormal process exit), push an event onto a ring buffer:
- fault type/vector, RIP, faulting address (CR2 for #PF), error code,
- full register snapshot (already captured by the ISR frame),
- the disassembled faulting instruction (ship a tiny x86 length/decode helper),
- offending process name/pid + its argv,
- the last N lines of the kernel log ring buffer (add one if absent),
- relevant file snippets if identifiable (e.g. a config the process opened).

### 4.2 Context builder
Format the event into the model's prompt template (a fixed system prompt that
explains the model is a kernel diagnostician + the structured fault data). Keep
within context length; truncate logs oldest-first.

### 4.3 Inference
Run the engine (on the AI thread from 0.4) on the prompt; stream the diagnosis to
the console/log.

### 4.4 Action gate (safety — do not skip)
The model SUGGESTS; it must not silently mutate the system. Define a restricted
action vocabulary the model can emit (e.g. `SUGGEST_CMD`, `EDIT_CONFIG <file>`,
`RESTART <svc>`, `RAISE_LIMIT <name> <val>`). Every action is shown and requires
confirmation (or runs only in an explicit "autopilot" mode the user opts into).
Never let the model issue raw arbitrary writes.

Acceptance: an induced null-deref in a test process yields a correct diagnosis and
a sensible, gated suggested fix on the console.

---

## Section A — Model file format (freeze this first)

```
struct mom_model_header {            // little-endian
    char     magic[4];               // "MOM1"
    uint32_t version;                // 1
    uint32_t arch;                   // 0 = dense, 1 = moe
    uint32_t d_model;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;             // == n_heads if no GQA
    uint32_t head_dim;
    uint32_t d_ff;                   // per-expert FFN hidden size
    uint32_t n_experts;              // 1 if dense
    uint32_t top_k;                  // 2 for MoE, 1 for dense
    uint32_t vocab_size;
    uint32_t context_len;
    float    rope_base;              // 10000.0
    float    rms_eps;                // 1e-5
    uint32_t n_tensors;
    // followed by n_tensors x mom_tensor_desc, then 64-byte-aligned data
};

struct mom_tensor_desc {
    char     name[64];               // e.g. "blk.0.attn_q.weight"
    uint32_t dtype;                  // 0=fp32, 1=fp16, 2=q8_0
    uint32_t n_dims;
    uint64_t shape[4];
    uint64_t offset;                 // byte offset into data section
    uint64_t nbytes;
};
```
Q8_0 block layout (matches llama.cpp Q8_0): for every 32 weights, store
`fp16 scale` then `32 x int8`. Dequant: `w[i] = q[i] * scale`.

Provide a Python converter (PyTorch/safetensors -> this format) as a host-side
tool in `tools/convert_model.py`. Validate by round-tripping a tiny model.

---

## Section B — Honest performance & risk notes

- 4-8 tok/s is achievable only with a tuned int8 SIMD matmul (AVX2/VNNI) and
  cache blocking. A naive fp32 matmul in a freestanding kernel will likely be
  <1 tok/s at first. Get correctness first, then optimize matmul; that one kernel
  is ~90% of compute time.
- Multi-core (SMP) would roughly multiply throughput by core count but SMP
  bring-up (APIC, AP startup, per-core scheduling) is a large separate project.
  Single-core first.
- The KV cache and weights dominate RAM. 1-2 GB weights + KV cache means boot
  with >=4 GB and confirm the PMM/VMM handle it.
- Biggest schedule risk is Phase 0, not the model. Budget accordingly.
- fp16 storage (KV cache, Q8_0 scales) needs fp16<->fp32 conversion: use the F16C
  instructions (`vcvtph2ps`/`vcvtps2ph`) if CPUID reports them, else a small
  software routine. Don't forget this — half the data is fp16.
- Any interrupt handler that touches float will clobber the interrupted thread's
  XMM registers. Keep ISRs FP-free, OR have them FXSAVE/FXRSTOR. The
  context-switch FXSAVE (0.4) only covers thread preemption, not nested ISR FP use.

---

## Section C — Suggested repo layout for the new work

```
src/ai/
  tensor.c/.h        # rmsnorm, matmul_q8, rope, softmax, swiglu, attention
  moe.c/.h           # router + top-2 dispatch
  model.c/.h         # format loader, forward pass, KV cache
  tokenizer.c/.h     # byte-level BPE
  sampler.c/.h       # greedy / temp / top-p
  infer.c/.h         # generation loop (public API: ai_generate(prompt)->text)
src/ai/diag/
  capture.c/.h       # fault hook -> event ring buffer
  context.c/.h       # event -> prompt
  daemon.c/.h        # AI thread, action gate
src/libm/
  mathf.c/.h         # expf, sqrtf, sinf/cosf or tables
tools/
  convert_model.py   # host-side: checkpoint -> MOM1 format
docs/
  AI_SUBSYSTEM.md    # this file
```

---

## Build order checklist (give this to the agent verbatim)

1. [ ] Enable SSE2 in kernel; build inference TUs with FP. (0.1)
2. [ ] Boot with 4 GB; confirm PMM. (0.2)
3. [ ] kmalloc/kfree. (0.3)
4. [ ] Kernel threads + context switch with FXSAVE. (0.4)
5. [ ] Disk read (ATA PIO + raw LBA). (0.5)
6. [ ] libm-lite (expf, sqrtf). (0.6)
7. [ ] Freeze MOM1 format + Python converter. (Section A)
8. [ ] Tensor kernels, unit-tested vs reference. (1.2)
9. [ ] Transformer + KV cache + sampler; greedy decode matches reference. (1.3-1.6)
10. [ ] MoE router + top-2. (1.3)
11. [ ] Byte-level BPE tokenizer, round-trip tested. (Phase 2)
12. [ ] Convert a small dense model (Path A); end-to-end coherent generation. (Phase 3A)
13. [ ] Fault capture -> context -> inference -> gated action. (Phase 4)
14. [ ] (Later) distill custom MoE (Path B); int8 SIMD matmul for tok/s target.
```
```
