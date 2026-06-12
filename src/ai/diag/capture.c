#include "capture.h"
#include "idt.h"
#include "timer.h"
#include "serial.h"

static diag_event_t ring[DIAG_RING_SIZE];
static volatile uint32_t head = 0;   /* next write slot, wraps mod DIAG_RING_SIZE */
static volatile uint32_t count = 0;  /* capped at DIAG_RING_SIZE */

static inline uint64_t irq_save(void) {
    uint64_t flags;
    __asm__ volatile ("pushfq; pop %0; cli" : "=r"(flags) : : "memory");
    return flags;
}

static inline void irq_restore(uint64_t flags) {
    __asm__ volatile ("push %0; popfq" : : "r"(flags) : "memory");
}

/* manual byte copy/zero: no libc memcpy/memset in a freestanding kernel */
static void copy_bytes(char *dst, const char *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        dst[i] = src[i];
}

static void zero_bytes(char *dst, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        dst[i] = 0;
}

/* IRQ-safe push of one event into the ring (byte-by-byte, no struct assign) */
static void push_event(const diag_event_t *ev) {
    uint64_t flags = irq_save();
    copy_bytes((char *)&ring[head], (const char *)ev, sizeof(diag_event_t));
    head = (head + 1) % DIAG_RING_SIZE;
    if (count < DIAG_RING_SIZE)
        count++;
    irq_restore(flags);
}

static char  _logbuf[97];
static uint32_t _logpos = 0;

void diag_init(void) {
    uint64_t flags = irq_save();
    zero_bytes((char *)ring, sizeof(ring));
    head  = 0;
    count = 0;
    zero_bytes(_logbuf, sizeof(_logbuf));
    _logpos = 0;
    irq_restore(flags);
    serial_print("[DIAG] capture init\n");
}

void diag_capture_fault(struct isr_frame *frame, uint32_t tid, const char *cwd, uint64_t cr2) {
    diag_event_t ev;
    uint32_t i;

    zero_bytes((char *)&ev, sizeof(ev));
    ev.type = DIAG_EVT_FAULT;
    ev.tick = timer_ticks();

    ev.fault.vector     = frame->vector;
    ev.fault.error_code = frame->error_code;
    ev.fault.rip        = frame->rip;
    ev.fault.rsp        = frame->rsp;
    ev.fault.rax        = frame->rax;
    ev.fault.rdi        = frame->rdi;
    ev.fault.rsi        = frame->rsi;
    ev.fault.cr2        = cr2;
    ev.fault.tid        = tid;

    for (i = 0; i < 63 && cwd && cwd[i]; i++)
        ev.fault.cwd[i] = cwd[i];
    ev.fault.cwd[i] = 0;

    push_event(&ev);
}

void diag_capture_syscall(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t retval, uint32_t tid) {
    diag_event_t ev;

    zero_bytes((char *)&ev, sizeof(ev));
    ev.type = DIAG_EVT_SYSCALL;
    ev.tick = timer_ticks();

    ev.syscall.number  = n;
    ev.syscall.args[0] = a1;
    ev.syscall.args[1] = a2;
    ev.syscall.args[2] = a3;
    ev.syscall.retval  = retval;
    ev.syscall.tid     = tid;

    push_event(&ev);
}

void diag_log_write(const char *text) {
    while (*text) {
        char c = *text++;

        if (c == '\n') {
            diag_event_t ev;
            uint32_t i;

            _logbuf[_logpos] = 0;
            zero_bytes((char *)&ev, sizeof(ev));
            ev.type = DIAG_EVT_LOG;
            ev.tick = timer_ticks();
            for (i = 0; i <= _logpos && i < 96; i++)
                ev.log.text[i] = _logbuf[i];
            ev.log.text[95] = 0;
            push_event(&ev);
            _logpos = 0;
        } else {
            /* append char; flush when buffer is full before adding */
            if (_logpos == 95) {
                diag_event_t ev;
                uint32_t i;

                _logbuf[95] = 0;
                zero_bytes((char *)&ev, sizeof(ev));
                ev.type = DIAG_EVT_LOG;
                ev.tick = timer_ticks();
                for (i = 0; i < 96; i++)
                    ev.log.text[i] = _logbuf[i];
                push_event(&ev);
                _logpos = 0;
            }
            _logbuf[_logpos++] = c;
        }
    }
}

uint32_t diag_ring_snapshot(diag_event_t *out, uint32_t max_out) {
    uint64_t flags = irq_save();
    uint32_t avail = count;
    uint32_t n = (avail < max_out) ? avail : max_out;

    /* copy most-recent-first: walk backwards from the slot before head */
    for (uint32_t i = 0; i < n; i++) {
        uint32_t idx = (head + DIAG_RING_SIZE - 1 - i) % DIAG_RING_SIZE;
        copy_bytes((char *)&out[i], (const char *)&ring[idx], sizeof(diag_event_t));
    }

    irq_restore(flags);
    return n;
}
