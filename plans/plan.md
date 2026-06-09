# MominOS AI Subsystem — Implementation Plan

## Goal
Build the full AI subsystem for MominOS as specified in `docs/AI_SUBSYSTEM.md`: an on-device LLM inference engine with MoE support, BPE tokenizer, and an error-diagnosis daemon, all running within the kernel as a schedulable thread, with the end-to-end demo: a kernel fault triggers an on-device diagnosis printed to console.

## Current State
Working: boot, GDT, paging, PMM, VMM, IDT/ISR, PIC/PIT, keyboard, serial, VGA, kernel heap (`kmalloc/kfree`).
Not yet working: SSE/FPU in kernel, 4GB boot, disk driver, scheduler, libm, any AI code.

## Research Summary
The spec is self-contained and defines everything precisely. Key facts confirmed from the existing codebase:
- Build: gcc, nasm, ld — all available. Makefile exists with clean targets.
- QEMU available for testing. Current boot memory: 512 MB (needs bump to 4G).
- kernel_entry.asm has NO FPU init code — CR0.EM not cleared, no CR4.FXSAVE bits.
- PMM bitmap covers up to 4 GB (MAX_PAGES = 4GB/4KB = 1,048,576).
- VMM supports 4-level paging with 2MB huge pages and page-level splitting.
- MOM1 format and Q8_0 quantization follow llama.cpp conventions (well-documented).
- No GPU available in sandbox — all work is CPU-only (expected for kernel work).
- Python3 available for `tools/convert_model.py` and golden-interval reference generation.

## Approach
Build strictly bottom-up per the spec's phase ordering. Phase 0 first (OS prerequisites), then Phase 1 (inference engine), Phase 2 (tokenizer), Phase 3 (model conversion), Phase 4 (daemon). Each phase has measurable acceptance criteria. We do not advance until criteria are met.

Key decisions:
- **SSE/SSE2 only** initially (AVX requires CPUID check + XSAVE); will add AVX path as optimization.
- **ATA PIO** for disk driver (simplest to write, no virtio knowledge needed); raw LBA model read first.
- **Cooperative kernel threads** (PIT-based preemption is more complex but we set up the context switch with FXSAVE from the start for correctness).
- **Small dense model first** (Path A from spec) for engine validation; MoE later.
- **Use existing llama.cpp Q8_0** block format for easy cross-validation.

## Subtasks

### Phase 0 — OS Prerequisites

**Subtask 0.1: Enable SSE/SSE2 in kernel**
- Add FPU init code in `kernel_entry.asm` (clear CR0.EM, set CR0.MP, set CR4.OSFXSR+OSXMMEXCPT).
- Create separate compile flags for inference TUs (drop `-mno-sse -mno-sse2`) while keeping kernel TUs FP-safe.
- Add a test function that does float ops and SIMD vector add, call it from kmain to verify no #UD.
- Update Makefile to allow per-TU CFLAGS.

**Subtask 0.2: Boot with 4 GB RAM**
- Change QEMU run target from `-m 512M` to `-m 4G`.
- Confirm PMM reports ~4 GB free pages.

**Subtask 0.3: Kernel heap** — DONE (kheap.c exists, self-test included). Verify it's working by calling `kheap_self_test()` from `kmain`.

**Subtask 0.4: Kernel threads + context switch**
- Implement `thread_create(func, arg)` — allocates stack + initial context frame.
- Implement `thread_yield()` — save GPRs + RSP, switch to next thread.
- Add FXSAVE/FXRSTOR in context switch (fxsave area sized 512 bytes, 16-byte aligned).
- Implement a simple round-robin scheduler with a linked list of threads.
- Acceptance: AI thread runs inference loop while heartbeat dot continues printing.

**Subtask 0.5: Disk driver (ATA PIO)**
- Write ATA PIO driver: read sectors by LBA (28-bit LBA addressing).
- Implement `ata_read_sectors(lba, count, buffer)`.
- Provide raw LBA-based model reading function `ata_read_blob(lba_start, buffer, byte_count)`.
- Acceptance: read a known sector and verify against a reference checksum.

**Subtask 0.6: libm-lite**
- Implement `expf(x)` using range reduction + minimax polynomial (musl/cephes algorithm).
- Implement `sqrtf(x)` using SSE `sqrtss` intrinsic.
- Implement `sinf(x)` and `cosf(x)` for RoPE — or precompute sin/cos table.
- Acceptance: `expf`/`sqrtf` match reference to <1e-5 over relevant ranges.
- Place in `src/libm/mathf.c`.

### Phase 1 — Inference Engine

**Subtask 1.1: MOM1 model format + Python converter**
- Define exact struct layouts for `mom_model_header`, `mom_tensor_desc`, Q8_0 block in C headers (`src/ai/model.h`).
- Write `tools/convert_model.py` — reads a PyTorch/safetensors checkpoint (or GGUF via llama.cpp conversion tools), converts to MOM1 binary blob with Q8_0 quantization.
- Include round-trip test: convert a tiny model, load it back in Python to verify.
- Acceptance: C headers match Python writer exactly.

**Subtask 1.2: Tensor kernels**
- Implement in `src/ai/tensor.c`:
  - `rmsnorm(x, weight, eps, out, n)`
  - `matmul_q8(w_q8, w_scales, x_fp32, out_fp32, m, n, k)` — v0 (dequant+fp32 dot), v1 (int8 SIMD)
  - `rope(q, k, pos, head_dim, theta_base)` — rotation per dim pair
  - `scaled_dot_product_attention(q, k, v, out, n_heads, n_kv_heads, seq_len, head_dim, causal)`
  - `swiglu_ffn(x, gate_w, up_w, down_w, hidden_dim)` — silu(gate(x)) * up(x) → down
  - `softmax(x, n)` — numerically stable (max subtract)
- Each function: C implementation first (non-SIMD), unit-tested against reference Python outputs.
- Acceptance: each kernel matches reference computation to 1e-5 (fp32).

**Subtask 1.3: MoE block**
- `moe_router(x, router_w, n_experts, top_k, gate_indices_out, gate_weights_out, d_model)`
- `moe_forward(x, experts, n_experts, top_k, gate_indices, gate_weights, d_model, d_ff)`
- Only compute top-k selected experts. Place in `src/ai/moe.c`.

**Subtask 1.4: Transformer forward pass**
- `transformer_forward(tokens, model, kv_cache, n_tokens)` in `src/ai/model.c`
- Per layer: RMSNorm → attention (with KV cache) → residual add → RMSNorm → MoE → residual add.
- Final RMSNorm → lm_head matmul → logits.
- Support configurable GQA ratio.
- Acceptance: logits for token 0 match reference runtime to ~1e-2.

**Subtask 1.5: KV cache**
- Allocate K/V buffers per layer: `[context_len][n_kv_heads * head_dim]` fp16.
- `kv_cache_append(layer, k_slice, v_slice)` — write at current position.
- `kv_cache_read(layer, pos)` — read back stored K/V.
- Use fp16 storage with F16C conversion (or software fallback).

**Subtask 1.6: Sampler + generation loop**
- Greedy (argmax) first. Then temperature + top-p.
- Generation loop in `src/ai/infer.c`: `ai_generate(prompt_tokens, max_tokens, output_buffer)`.
- Prefill prompt in one forward pass, then autoregressive decode.
- Acceptance: greedy decode matches reference runtime exactly.

**Subtask 1.7: Golden-intermediate validation harness**
- In Python (using reference runtime), run a fixed prompt through a small model.
- Dump every intermediate tensor (post-embedding, post-each-RMSNorm, post-attention, post-FFN, logits).
- In C, dump the same intermediates and diff against the golden tensors.
- Pinpoints the FIRST divergence — critical debugging tool.

### Phase 2 — Tokenizer

**Subtask 2.1: Byte-level BPE tokenizer**
- `src/ai/tokenizer.c`: `tokenizer_encode(str) → tokens`, `tokenizer_decode(tokens) → str`.
- Load vocab (string→id) and merges (ordered pairs) from data files.
- Byte-level encoding per GPT-2/Llama spec.
- Acceptance: `decode(encode(s)) == s` for ASCII + UTF-8; token IDs match reference tokenizer.

### Phase 3 — Model

**Subtask 3.1: Bootstrap with small dense model (Path A)**
- Pick a small Llama-architecture model (e.g., TinyLlama-1.1B or SmolLM-135M).
- Convert to MOM1 Q8_0 format using `tools/convert_model.py`.
- Load and run in MominOS engine.
- Acceptance: engine produces same greedy decode as reference runtime.

### Phase 4 — Error-Diagnosis Daemon

**Subtask 4.1: Fault capture**
- Hook `isr_handler()` in `idt.c` to push fault events to a ring buffer.
- Capture: vector, RIP, CR2 (for #PF), error code, register snapshot (already in frame).
- Place in `src/ai/diag/capture.c`.

**Subtask 4.2: Context builder**
- Format captured event into prompt template: system prompt (diagnostician role) + structured fault data.
- Truncate oldest log entries to stay within context length.
- Place in `src/ai/diag/context.c`.

**Subtask 4.3: Inference thread**
- AI daemon as a kernel thread (using scheduler from 0.4).
- Poll fault ring buffer; when event arrives, run inference and stream diagnosis.
- Place in `src/ai/diag/daemon.c`.

**Subtask 4.4: Action gate**
- Define restricted action vocabulary (SUGGEST_CMD, EDIT_CONFIG, RAISE_LIMIT, RESTART).
- Show suggestion and require y/N confirmation before applying.
- Never silently mutate system.
- Acceptance: induced null-deref yields correct diagnosis + gated fix suggestion.

## Deliverables

| File Path | Description |
|-----------|-------------|
| `src/kernel/kernel_entry.asm` | FPU init in entry code |
| `src/libm/mathf.c` + `.h` | expf, sqrtf, sinf/cosf |
| `src/kernel/sched.c` + `.h` | Kernel threads + context switch |
| `src/kernel/ata.c` + `.h` | ATA PIO disk driver |
| `src/ai/model.h` | MOM1 format definitions |
| `src/ai/tensor.c` + `.h` | RMSNorm, matmul_q8, RoPE, attention, SwiGLU, softmax |
| `src/ai/moe.c` + `.h` | MoE router + forward |
| `src/ai/model.c` + `.h` | Format loader, transformer forward, KV cache |
| `src/ai/tokenizer.c` + `.h` | Byte-level BPE tokenizer |
| `src/ai/sampler.c` + `.h` | Greedy/temp/top-p sampler |
| `src/ai/infer.c` + `.h` | Generation loop |
| `src/ai/diag/capture.c` + `.h` | Fault ring buffer |
| `src/ai/diag/context.c` + `.h` | Event→prompt builder |
| `src/ai/diag/daemon.c` + `.h` | AI daemon thread |
| `tools/convert_model.py` | PyTorch→MOM1 converter |
| `tools/golden_harness.py` | Golden-intermediate reference generator |

## Evaluation Criteria
1. Build compiles without errors (`make all`)
2. QEMU boots with `-m 4G`, all init messages print
3. SSE float ops run without #UD
4. kheap self-test passes on boot
5. Context switch test: two threads alternate
6. ATA PIO reads verify checksum
7. expf/sqrtf match reference to <1e-5
8. Tensor kernels match reference to 1e-5
9. Greedy decode matches reference runtime
10. Tokenizer round-trips: `decode(encode(s)) == s`
11. Induced fault → diagnosis printed (Phase 4)

## Notes
- No GPU available; all kernel code is CPU-only.
- Build with `make all` and run with `make run` (after updating QEMU memory flag).
- For AVX, add runtime CPUID check before using — safe fallback to SSE-only paths.
- The golden-intermediate harness (1.7) should be built EARLY to avoid debugging blind.
- MoE is optional for Phase 1 validation — test with dense (n_experts=1) first.