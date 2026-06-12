#include "context.h"
#include "capture.h"

static diag_event_t snap[DIAG_RING_SIZE];

static const char *exception_names[32] = {
    "Divide Error",
    "Debug",
    "NMI",
    "Breakpoint",
    "Overflow",
    "Bound Range",
    "Invalid Opcode",
    "Device Not Available",
    "Double Fault",
    "Coprocessor Segment",
    "Invalid TSS",
    "Segment Not Present",
    "Stack Segment",
    "General Protection",
    "Page Fault",
    "Reserved",
    "x87 Floating Point",
    "Alignment Check",
    "Machine Check",
    "SIMD Floating Point",
    "Virtualization",
    "Control Protection",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Hypervisor Injection",
    "VMM Communication",
    "Security",
    "Reserved",
};

static const char *syscall_names[12] = {
    "?",
    "SYS_WRITE",
    "SYS_READ",
    "SYS_OPEN",
    "SYS_CLOSE",
    "?",
    "SYS_EXIT",
    "SYS_SPAWN",
    "SYS_WAITPID",
    "SYS_READDIR",
    "SYS_CHDIR",
    "SYS_GETCWD",
};

static uint32_t append(char *buf, uint32_t cap, uint32_t pos, const char *s) {
    while (*s && pos + 1 < cap)
        buf[pos++] = *s++;
    if (pos < cap)
        buf[pos] = 0;
    return pos;
}

static uint32_t append_hex(char *buf, uint32_t cap, uint32_t pos, uint64_t val) {
    static const char hex[] = "0123456789ABCDEF";

    pos = append(buf, cap, pos, "0x");
    for (int i = 60; i >= 0; i -= 4) {
        if (pos + 1 >= cap)
            break;
        buf[pos++] = hex[(val >> i) & 0xF];
    }
    if (pos < cap)
        buf[pos] = 0;
    return pos;
}

static uint32_t append_u32(char *buf, uint32_t cap, uint32_t pos, uint32_t val) {
    char tmp[11];
    int n = 0;

    if (val == 0) {
        tmp[n++] = '0';
    } else {
        while (val > 0 && n < 10) {
            tmp[n++] = (char)('0' + (val % 10));
            val /= 10;
        }
    }
    while (n > 0 && pos + 1 < cap)
        buf[pos++] = tmp[--n];
    if (pos < cap)
        buf[pos] = 0;
    return pos;
}

int diag_build_context(char *buf, uint32_t capacity) {
    uint32_t n = diag_ring_snapshot(snap, DIAG_RING_SIZE);
    uint32_t pos = 0;
    int fault_idx = -1;

    if (capacity < 200)
        return -1;

    /* snapshot is most-recent-first, so the first fault found is the latest */
    for (uint32_t i = 0; i < n; i++) {
        if (snap[i].type == DIAG_EVT_FAULT) {
            fault_idx = (int)i;
            break;
        }
    }
    if (fault_idx < 0)
        return -1;

    {
        const diag_fault_t *f = &snap[fault_idx].fault;
        uint64_t fault_tick = snap[fault_idx].tick;

        pos = append(buf, capacity, pos,
                     "[SYSTEM] MominoMoE kernel fault diagnostician. "
                     "Analyze the fault and suggest a corrective action.\n\n");

        pos = append(buf, capacity, pos, "[FAULT] vector=");
        pos = append_u32(buf, capacity, pos, (uint32_t)f->vector);
        pos = append(buf, capacity, pos, " (");
        pos = append(buf, capacity, pos,
                     f->vector < 32 ? exception_names[f->vector] : "?");
        pos = append(buf, capacity, pos, ") err=");
        pos = append_hex(buf, capacity, pos, f->error_code);
        pos = append(buf, capacity, pos, " rip=");
        pos = append_hex(buf, capacity, pos, f->rip);
        pos = append(buf, capacity, pos, " cr2=");
        pos = append_hex(buf, capacity, pos, f->cr2);
        pos = append(buf, capacity, pos, " tid=");
        pos = append_u32(buf, capacity, pos, f->tid);
        pos = append(buf, capacity, pos, " cwd=");
        pos = append(buf, capacity, pos, f->cwd);
        pos = append(buf, capacity, pos, "\n\n");

        pos = append(buf, capacity, pos, "[REGISTERS] rax=");
        pos = append_hex(buf, capacity, pos, f->rax);
        pos = append(buf, capacity, pos, " rdi=");
        pos = append_hex(buf, capacity, pos, f->rdi);
        pos = append(buf, capacity, pos, " rsi=");
        pos = append_hex(buf, capacity, pos, f->rsi);
        pos = append(buf, capacity, pos, " rsp=");
        pos = append_hex(buf, capacity, pos, f->rsp);
        pos = append(buf, capacity, pos, "\n\n");

        if (pos <= capacity - 150) {
            uint32_t emitted = 0;

            pos = append(buf, capacity, pos, "[RECENT_SYSCALLS]\n");
            for (uint32_t i = 0; i < n && emitted < 8; i++) {
                if (snap[i].type != DIAG_EVT_SYSCALL)
                    continue;
                if (snap[i].tick > fault_tick)
                    continue;
                if (pos > capacity - 150)
                    break;

                {
                    const diag_syscall_t *s = &snap[i].syscall;
                    const char *name = (s->number < 12)
                                           ? syscall_names[s->number]
                                           : "?";

                    pos = append(buf, capacity, pos, "  ");
                    pos = append(buf, capacity, pos, name);
                    pos = append(buf, capacity, pos, " ");
                    pos = append_hex(buf, capacity, pos, s->args[0]);
                    pos = append(buf, capacity, pos, " ");
                    pos = append_hex(buf, capacity, pos, s->args[1]);
                    pos = append(buf, capacity, pos, " ");
                    pos = append_hex(buf, capacity, pos, s->args[2]);
                    pos = append(buf, capacity, pos, " -> ");
                    pos = append_hex(buf, capacity, pos, s->retval);
                    pos = append(buf, capacity, pos, "\n");
                }
                emitted++;
            }
        }

        if (pos <= capacity - 150) {
            uint32_t emitted = 0;

            pos = append(buf, capacity, pos, "\n[LOG]\n");
            for (uint32_t i = 0; i < n && emitted < 10; i++) {
                if (snap[i].type != DIAG_EVT_LOG)
                    continue;
                if (pos > capacity - 150)
                    break;

                pos = append(buf, capacity, pos, "  ");
                pos = append(buf, capacity, pos, snap[i].log.text);
                pos = append(buf, capacity, pos, "\n");
                emitted++;
            }
        }

        pos = append(buf, capacity, pos,
                     "\n[QUERY] Diagnose this fault and suggest a corrective action.\n");
    }

    if (pos < capacity)
        buf[pos] = 0;
    else
        buf[capacity - 1] = 0;
    return (int)pos;
}
