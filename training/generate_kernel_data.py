#!/usr/bin/env python3
"""
Synthetic kernel fault diagnosis training data generator.
Produces prompt/response pairs in MominOS harness envelope format.
"""
import random, json, os, argparse

random.seed(42)

CWDS  = ["/", "/bin", "/home/user", "/tmp", "/root", "/proc", "/usr/bin"]
PATHS = ["/bin/sh", "/bin/ls", "/bin/cat", "/usr/bin/grep",
         "/usr/lib/libc.so", "/etc/passwd", "/tmp/work", "/proc/self/maps"]

PAGE_FAULT_ERR = {
    0: "read, non-present page, kernel",
    2: "write, non-present page, kernel",
    4: "read, non-present page, user",
    5: "read, protection violation, user",
    6: "write, non-present page, user",
    7: "write, protection violation, user",
}

SYSCALL_DEFS = [
    ("SYS_READ",   lambda: f"{random.randint(3,7)} {random.choice([64,256,512,4096])}",  lambda: random.randint(0, 4096)),
    ("SYS_WRITE",  lambda: f"{random.randint(3,7)} {random.choice([16,64,128])}",        lambda: random.randint(0, 128)),
    ("SYS_OPEN",   lambda: f"{random.choice(PATHS)} 0",                                  lambda: random.randint(3, 9)),
    ("SYS_CLOSE",  lambda: str(random.randint(3, 9)),                                    lambda: 0),
    ("SYS_SPAWN",  lambda: random.choice(PATHS),                                          lambda: random.randint(4, 20)),
    ("SYS_GETPID", lambda: "",                                                            lambda: random.randint(1, 50)),
    ("SYS_MMAP",   lambda: f"{_uaddr():#x} {random.choice([0x1000,0x4000,0x10000]):#x} 0x3",
                                                                                          lambda: _uaddr()),
    ("SYS_MUNMAP", lambda: f"{_uaddr():#x} {0x1000:#x}",                                 lambda: 0),
    ("SYS_SBRK",   lambda: f"{random.choice([0x1000,0x4000]):#x}",                       lambda: _uaddr()),
]

LOG_TEMPLATES = [
    lambda: f"  [VMM] mapped {_uaddr():#x}->{_uaddr():#x}",
    lambda: f"  [SCHED] thread {random.randint(1,10)} running",
    lambda: f"  [VFS] opened {random.choice(PATHS)}",
    lambda: f"  [VFS] read {random.randint(64,4096)} bytes from fd {random.randint(3,7)}",
    lambda: f"  [HEAP] allocated {random.randint(16,4096)} bytes at {_uaddr():#x}",
    lambda: f"  [ELF] loaded {random.choice(PATHS)} at {_kaddr():#x}",
    lambda: f"  [PMM] page {_uaddr():#x} allocated",
    lambda: f"  [VMM] unmapped {_uaddr():#x}",
    lambda: f"  [IRQ] timer tick {random.randint(1000,9999)}",
    lambda: f"  [TTY] flushed {random.randint(1,80)} chars",
]

def _kaddr():
    return 0xFFFF800000000000 + random.randint(0, 0x1FFFFF)

def _uaddr():
    bases = [0x400000, 0x7fff0000, 0x10000000, 0x20000000]
    return random.choice(bases) + random.randint(0, 0xFFFF)

def _bad_addr():
    return random.choice([0, 0x8, 0x10, 0x18, 0x1000, 0xDEAD, 0xBAAD0000, 0xFFFF])

def syscall_history(n=4):
    lines = []
    for _ in range(n):
        name, args_fn, ret_fn = random.choice(SYSCALL_DEFS)
        args = args_fn()
        ret  = ret_fn()
        lines.append(f"  {name} {args} -> {ret}")
    return lines

def log_lines(n=5):
    return [random.choice(LOG_TEMPLATES)() for _ in range(n)]

def build_prompt(vector, fname, err, rip, rsp, rax, rdi, rsi, cr2, tid, cwd, scalls, logs):
    cr2_str = f"cr2={cr2:#018x} " if vector == 14 else ""
    lines = [
        "[SYSTEM] MominOS kernel fault diagnostician. Analyze and suggest a fix.",
        "",
        f"[FAULT] vector={vector} ({fname}) err={err:#06x} rip={rip:#018x} {cr2_str}tid={tid} cwd={cwd}",
        "",
        f"[REGISTERS] rax={rax:#018x} rdi={rdi:#018x} rsi={rsi:#018x} rsp={rsp:#018x}",
        "",
        "[RECENT_SYSCALLS]",
    ] + scalls + ["", "[LOG]"] + logs + ["", "[QUERY] Diagnose this fault and suggest a corrective action."]
    return "\n".join(lines)


# ── Fault generators ──────────────────────────────────────────────────────────

def gen_page_fault_null():
    err  = random.choice([4, 6])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    cr2  = random.randint(0, 0x18)
    rax  = random.randint(0, 0xFFFF)
    prompt = build_prompt(14, "Page Fault", err, rip, rsp, rax, cr2, 0, cr2, tid, cwd,
                          syscall_history(random.randint(2,6)), log_lines(random.randint(3,7)))
    response = (
        f"**Fault Type:** Page Fault — Null/Near-Null Pointer Dereference\n"
        f"**Root Cause:** cr2={cr2:#x} is a near-null address. "
        f"Error {err:#x} ({PAGE_FAULT_ERR[err]}). "
        f"A pointer was never initialized or was set to NULL before this access.\n"
        f"**Instruction:** RIP={rip:#x}. RSP={rsp:#x} — stack appears valid.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** Add a NULL check before every dereference. "
        f"If the pointer is expected to be valid here, trace its initialization path — "
        f"it was likely not set before use or was freed early."
    )
    return prompt, response

def gen_page_fault_bad_ptr():
    err  = random.choice([4, 6])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    cr2  = _bad_addr()
    rax  = random.randint(0, 0xFFFF)
    prompt = build_prompt(14, "Page Fault", err, rip, rsp, rax, cr2, 0, cr2, tid, cwd,
                          syscall_history(random.randint(2,6)), log_lines(random.randint(3,7)))
    response = (
        f"**Fault Type:** Page Fault — Invalid / Garbage Pointer\n"
        f"**Root Cause:** cr2={cr2:#x} is not in any mapped VMA. "
        f"Error {err:#x} ({PAGE_FAULT_ERR[err]}). "
        f"The pointer value is garbage — likely an uninitialized stack variable, "
        f"corrupted by an out-of-bounds write, or result of bad pointer arithmetic.\n"
        f"**Instruction:** RIP={rip:#x}.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** Check pointer arithmetic for off-by-one errors. "
        f"Verify the allocation returned a valid pointer before use. "
        f"Inspect recent writes near the pointer variable for buffer overruns."
    )
    return prompt, response

def gen_page_fault_stack_overflow():
    err  = random.choice([4, 6])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rsp  = _uaddr() & ~0xFFF
    cr2  = rsp - random.randint(1, 16)
    rip  = _uaddr()
    rax  = random.randint(0, 0xFFFF)
    prompt = build_prompt(14, "Page Fault", err, rip, rsp, rax, cr2, 0, cr2, tid, cwd,
                          syscall_history(random.randint(2,5)), log_lines(random.randint(3,6)))
    response = (
        f"**Fault Type:** Page Fault — Stack Overflow\n"
        f"**Root Cause:** cr2={cr2:#x} is just below RSP={rsp:#x} (delta={rsp-cr2} bytes). "
        f"The stack grew past the guard page. "
        f"Error {err:#x} ({PAGE_FAULT_ERR[err]}).\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** Reduce stack usage — move large local arrays to heap (malloc/kmalloc). "
        f"Check for unbounded recursion. "
        f"If the function is deeply recursive, add a depth counter and bail early."
    )
    return prompt, response

def gen_page_fault_use_after_free():
    err  = random.choice([4, 6])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    cr2  = _uaddr()
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    rax  = random.randint(0, 0xFFFF)
    scalls = syscall_history(random.randint(2, 5))
    # inject a MUNMAP to hint at use-after-free
    scalls.insert(random.randint(0, len(scalls)), f"  SYS_MUNMAP {cr2:#x} 0x1000 -> 0")
    prompt = build_prompt(14, "Page Fault", err, rip, rsp, rax, cr2, 0, cr2, tid, cwd,
                          scalls, log_lines(random.randint(3,6)))
    response = (
        f"**Fault Type:** Page Fault — Use-After-Free\n"
        f"**Root Cause:** cr2={cr2:#x} was accessed after it was unmapped/freed "
        f"(note SYS_MUNMAP {cr2:#x} in recent syscalls). "
        f"Error {err:#x} ({PAGE_FAULT_ERR[err]}). A dangling pointer is being used.\n"
        f"**Instruction:** RIP={rip:#x}.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** Set the pointer to NULL immediately after free/munmap. "
        f"Audit all code paths that share this pointer. "
        f"Consider using a ref-counted wrapper to prevent premature deallocation."
    )
    return prompt, response

def gen_page_fault_prot_violation():
    err  = random.choice([5, 7])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    cr2  = _uaddr()
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    rax  = random.randint(0, 0xFFFF)
    prompt = build_prompt(14, "Page Fault", err, rip, rsp, rax, cr2, 0, cr2, tid, cwd,
                          syscall_history(random.randint(2,5)), log_lines(random.randint(3,6)))
    write_fault = err in [3, 7]
    response = (
        f"**Fault Type:** Page Fault — Permission Violation\n"
        f"**Root Cause:** {'Write' if write_fault else 'Read'} access to {cr2:#x} denied by page table. "
        f"Error {err:#x} ({PAGE_FAULT_ERR[err]}). "
        f"The page is present but {'not writable' if write_fault else 'not readable'} at the current privilege level.\n"
        f"**Instruction:** RIP={rip:#x}.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** {'Check mmap/mprotect flags — the region was mapped read-only. If a write is needed, remap with PROT_WRITE.' if write_fault else 'The page is mapped but user-mode cannot read it — likely a kernel address. Check privilege level before accessing.'}"
    )
    return prompt, response

def gen_gpf():
    err  = random.choice([0, 0x02, 0x04, 0x08, 0x10])
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = random.choice([_kaddr(), _uaddr()])
    rsp  = _kaddr()
    rax  = random.randint(0, 0xFFFF)
    rdi  = _uaddr()
    rsi  = random.randint(0, 0x1000)
    fault_class = random.choice(["privileged_instr", "bad_selector", "misaligned", "kernel_access"])
    prompt = build_prompt(13, "General Protection Fault", err, rip, rsp, rax, rdi, rsi, 0, tid, cwd,
                          syscall_history(random.randint(2,5)), log_lines(random.randint(3,6)))
    if fault_class == "privileged_instr":
        response = (
            f"**Fault Type:** General Protection Fault — Privileged Instruction\n"
            f"**Root Cause:** User-mode code at RIP={rip:#x} executed a ring-0 instruction "
            f"(cli/hlt/in/out/wrmsr). Error {err:#x}.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** User processes cannot execute privileged instructions. "
            f"Terminate the process. If this is kernel code, verify the CPL was 0. "
            f"Audit the ELF loader — this code should not be in user space."
        )
    elif fault_class == "bad_selector":
        idx = err >> 3
        response = (
            f"**Fault Type:** General Protection Fault — Invalid Segment Selector\n"
            f"**Root Cause:** Descriptor at GDT/LDT index {idx} is invalid or not present. "
            f"Error {err:#x} encodes the selector. RIP={rip:#x}.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Verify GDT entry {idx} is correctly initialized. "
            f"This can also occur after a bad iretq frame restores a corrupted CS or SS register."
        )
    elif fault_class == "misaligned":
        response = (
            f"**Fault Type:** General Protection Fault — Misaligned Access\n"
            f"**Root Cause:** Instruction at RIP={rip:#x} performed a misaligned memory access "
            f"that the CPU rejected. Error {err:#x}.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Check for packed structs or manual pointer arithmetic producing unaligned pointers. "
            f"SIMD (SSE/AVX) loads require 16/32-byte alignment. "
            f"Use __attribute__((aligned(16))) or _mm_loadu variants."
        )
    else:
        response = (
            f"**Fault Type:** General Protection Fault — Kernel Memory Access from User Mode\n"
            f"**Root Cause:** User-mode code at RIP={rip:#x} attempted to access a "
            f"kernel virtual address. Error {err:#x}.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Never pass kernel addresses to user-mode code. "
            f"Check for a syscall that leaked a kernel pointer into userspace. "
            f"Validate all pointer arguments to syscalls against user-accessible VMA ranges."
        )
    return prompt, response

def gen_div_zero():
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    rax  = 0
    rdi  = random.randint(0, 0xFFFF)
    rsi  = random.randint(0, 0xFFFF)
    prompt = build_prompt(0, "Divide by Zero", 0, rip, rsp, rax, rdi, rsi, 0, tid, cwd,
                          syscall_history(random.randint(1,4)), log_lines(random.randint(2,5)))
    divisor_src = random.choice(["user input", "a counter that reached zero", "an uninitialized variable"])
    response = (
        f"**Fault Type:** Divide by Zero (vector 0 — #DE)\n"
        f"**Root Cause:** Integer division at RIP={rip:#x} with a zero divisor. "
        f"RAX={rax:#x} was the dividend. The divisor likely came from {divisor_src}.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** Guard the division: check the divisor is non-zero before div/idiv. "
        f"If the divisor comes from external input, validate it at the syscall boundary. "
        f"Return an error code or use a default value rather than dividing."
    )
    return prompt, response

def gen_invalid_opcode():
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = _uaddr()
    rsp  = (_uaddr() & ~0xFFF) + 0xFF8
    rax  = random.randint(0, 0xFFFFFFFF)
    rdi  = random.randint(0, 0xFFFF)
    rsi  = random.randint(0, 0xFFFF)
    fault_class = random.choice(["corrupt_text", "stack_smash", "wrong_cpu"])
    prompt = build_prompt(6, "Invalid Opcode", 0, rip, rsp, rax, rdi, rsi, 0, tid, cwd,
                          syscall_history(random.randint(1,4)), log_lines(random.randint(2,5)))
    if fault_class == "corrupt_text":
        response = (
            f"**Fault Type:** Invalid Opcode (vector 6 — #UD) — Corrupt Code Segment\n"
            f"**Root Cause:** CPU fetched an undefined byte sequence at RIP={rip:#x}. "
            f"The code page was overwritten, the RIP jumped to data, or the ELF is malformed.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Verify ELF integrity. Check if any code path writes to the text segment (W^X violation). "
            f"If RIP points to a heap or stack address, there is a wild function pointer or corrupted vtable."
        )
    elif fault_class == "stack_smash":
        response = (
            f"**Fault Type:** Invalid Opcode (vector 6 — #UD) — Corrupted Return Address\n"
            f"**Root Cause:** RIP={rip:#x} jumped to an invalid location — "
            f"likely a stack buffer overflow overwrote the return address with garbage. "
            f"The CPU hit an undefined byte sequence at that address.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Enable stack canaries. Audit recent function calls for buffer overflows. "
            f"This may be a security vulnerability — an attacker could redirect RIP to shellcode."
        )
    else:
        response = (
            f"**Fault Type:** Invalid Opcode (vector 6 — #UD) — Unsupported CPU Extension\n"
            f"**Root Cause:** Instruction at RIP={rip:#x} uses a CPU extension "
            f"(AVX-512, AMX, etc.) not supported by this processor.\n"
            f"**Thread:** tid={tid}, cwd={cwd}.\n"
            f"**Fix:** Check CPUID for supported features before executing optional ISA extensions. "
            f"Rebuild with -march=x86-64-v2 or appropriate baseline to avoid newer instructions."
        )
    return prompt, response

def gen_double_fault():
    tid  = random.randint(1, 12)
    cwd  = random.choice(CWDS)
    rip  = _kaddr()
    rsp  = _kaddr() & ~0xF   # kernel RSP
    rax  = random.randint(0, 0xFFFF)
    rdi  = random.randint(0, 0xFFFF)
    rsi  = random.randint(0, 0xFFFF)
    prompt = build_prompt(8, "Double Fault", 0, rip, rsp, rax, rdi, rsi, 0, tid, cwd,
                          syscall_history(random.randint(1,3)), log_lines(random.randint(2,5)))
    response = (
        f"**Fault Type:** Double Fault (vector 8 — #DF) — CRITICAL\n"
        f"**Root Cause:** A fault occurred while the CPU was handling another fault, "
        f"and the nested fault could not be handled. "
        f"This almost always means the kernel stack is exhausted (RSP={rsp:#x} may be at or past "
        f"the stack bottom) or the IDT/TSS is corrupt.\n"
        f"**Thread:** tid={tid}, cwd={cwd}.\n"
        f"**Fix:** This is a kernel-level crisis. "
        f"Check kernel stack size — a deep interrupt nesting chain or recursive kernel call "
        f"exhausted the stack. Verify the TSS.rsp0 is pointing to a valid kernel stack. "
        f"If the IDT entry for the original fault was corrupt, fix the IDT setup."
    )
    return prompt, response


# ── Driver ────────────────────────────────────────────────────────────────────

GENERATORS = [
    (gen_page_fault_null,          0.20),
    (gen_page_fault_bad_ptr,       0.18),
    (gen_page_fault_stack_overflow,0.12),
    (gen_page_fault_use_after_free,0.12),
    (gen_page_fault_prot_violation,0.10),
    (gen_gpf,                      0.14),
    (gen_div_zero,                 0.07),
    (gen_invalid_opcode,           0.05),
    (gen_double_fault,             0.02),
]

def generate(n_train=10000, n_val=1000):
    fns, weights = zip(*GENERATORS)
    total = n_train + n_val
    samples = []
    for _ in range(total):
        fn = random.choices(fns, weights=weights)[0]
        p, r = fn()
        samples.append({"prompt": p, "response": r})
    random.shuffle(samples)
    return samples[:n_train], samples[n_train:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  default="training/data")
    parser.add_argument("--n-train",  type=int, default=10000)
    parser.add_argument("--n-val",    type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Generating {args.n_train} train + {args.n_val} val samples...")
    train, val = generate(args.n_train, args.n_val)

    train_path = os.path.join(args.out_dir, "kernel_train.jsonl")
    val_path   = os.path.join(args.out_dir, "kernel_val.jsonl")

    with open(train_path, "w") as f:
        for s in train: f.write(json.dumps(s) + "\n")
    with open(val_path, "w") as f:
        for s in val:   f.write(json.dumps(s) + "\n")

    print(f"Train: {len(train)} → {train_path}")
    print(f"Val:   {len(val)}   → {val_path}")
    print("\n--- Sample prompt ---")
    print(train[0]["prompt"])
    print("\n--- Sample response ---")
    print(train[0]["response"])

if __name__ == "__main__":
    main()
