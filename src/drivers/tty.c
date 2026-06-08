#include "tty.h"
#include "serial.h"
#include "sched.h"

#define TTY_BUF_SIZE 1024

/* Ring buffer of cooked input bytes. A reader is released once at least
   one complete line ('\n') has been committed to the buffer. */
static char ring[TTY_BUF_SIZE];
static volatile uint32_t head;       /* write index (producer) */
static volatile uint32_t tail;       /* read index (consumer) */
static volatile uint32_t committed;  /* bytes available to readers */
static volatile uint32_t lines;      /* number of complete lines ready */
static volatile uint32_t edit;       /* bytes in the in-progress line */

static struct thread *waiter;

static inline uint64_t irq_save(void) {
    uint64_t flags;

    __asm__ volatile ("pushfq; pop %0; cli" : "=r"(flags) : : "memory");
    return flags;
}

static inline void irq_restore(uint64_t flags) {
    __asm__ volatile ("push %0; popfq" : : "r"(flags) : "memory");
}

void tty_init(void) {
    head = 0;
    tail = 0;
    committed = 0;
    lines = 0;
    edit = 0;
    waiter = 0;
    serial_print("[TTY] initialized\n");
}

void tty_feed(char c) {
    uint64_t flags = irq_save();

    if (c == '\b' || c == 127) {
        /* erase one char from the in-progress (uncommitted) line */
        if (edit > 0) {
            head = (head + TTY_BUF_SIZE - 1) % TTY_BUF_SIZE;
            edit--;
            serial_print("\b \b");
        }
        irq_restore(flags);
        return;
    }

    if (c == '\r')
        c = '\n';

    /* drop input if the buffer is full */
    if (((head + 1) % TTY_BUF_SIZE) == tail) {
        irq_restore(flags);
        return;
    }

    ring[head] = c;
    head = (head + 1) % TTY_BUF_SIZE;
    serial_putc(c);

    if (c == '\n') {
        /* commit the whole in-progress line, including the newline */
        committed += edit + 1;
        edit = 0;
        lines++;
        if (waiter != 0) {
            sched_wake(waiter);
            waiter = 0;
        }
    } else {
        edit++;
    }

    irq_restore(flags);
}

size_t tty_read(char *buf, size_t size) {
    uint64_t flags;
    size_t n = 0;

    if (size == 0)
        return 0;

    flags = irq_save();

    /* block until at least one complete line is available */
    while (lines == 0) {
        waiter = sched_current_thread();
        irq_restore(flags);
        sched_block();
        flags = irq_save();
    }

    /* copy out at most one line worth of committed bytes */
    while (n < size && committed > 0) {
        char c = ring[tail];
        tail = (tail + 1) % TTY_BUF_SIZE;
        committed--;
        buf[n++] = c;
        if (c == '\n') {
            lines--;
            break;
        }
    }

    irq_restore(flags);
    return n;
}
