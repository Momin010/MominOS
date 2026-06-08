#include "timer.h"
#include "serial.h"
#include "sched.h"

#define PIT_BASE_HZ 1193182
#define PIT_COMMAND 0x43
#define PIT_CHANNEL0 0x40

static uint64_t ticks;
static uint32_t tick_hz;

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

void timer_init(uint32_t hz) {
    uint32_t divisor;

    if (hz == 0)
        hz = 100;

    divisor = PIT_BASE_HZ / hz;
    tick_hz = hz;
    ticks = 0;

    outb(PIT_COMMAND, 0x36);
    outb(PIT_CHANNEL0, divisor & 0xFF);
    outb(PIT_CHANNEL0, (divisor >> 8) & 0xFF);

    serial_print("[PIT] initialized\n");
}

void timer_irq(void) {
    ticks++;
    sched_tick();

    if (tick_hz != 0 && (ticks % tick_hz) == 0)
        serial_putc('.');
}

uint64_t timer_ticks(void) {
    return ticks;
}
