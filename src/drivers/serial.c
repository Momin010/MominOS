#include "serial.h"
#include "tty.h"
#include "../ai/diag/capture.h"

#define COM1 0x3F8

static inline void outb(unsigned short port, unsigned char val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline unsigned char inb(unsigned short port) {
    unsigned char ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

void serial_init(void) {
    outb(COM1 + 1, 0x00);  /* disable interrupts */
    outb(COM1 + 3, 0x80);  /* DLAB on */
    outb(COM1 + 0, 0x03);  /* 38400 baud lo */
    outb(COM1 + 1, 0x00);  /* 38400 baud hi */
    outb(COM1 + 3, 0x03);  /* 8n1 */
    outb(COM1 + 2, 0xC7);  /* FIFO enable */
    outb(COM1 + 4, 0x0B);  /* RTS/DSR + OUT2 (gates the IRQ line to the PIC) */
    outb(COM1 + 1, 0x01);  /* enable "received data available" interrupt (IRQ4) */
}

/* COM1 RX interrupt: drain every byte the UART has buffered into the line
   discipline. Reading the RBR clears the data-ready interrupt; we loop until
   the Line Status Register reports no more data so a burst isn't left behind. */
void serial_irq(void) {
    while (inb(COM1 + 5) & 0x01)
        tty_feed((char)inb(COM1));
}

void serial_putc(char c) {
    while ((inb(COM1 + 5) & 0x20) == 0);
    outb(COM1, c);
}

void serial_print(const char *s) {
    diag_log_write(s);
    while (*s) {
        if (*s == '\n')
            serial_putc('\r');
        serial_putc(*s++);
    }
}

void serial_print_hex(unsigned long val) {
    const char *hex = "0123456789ABCDEF";
    serial_print("0x");
    for (int i = 60; i >= 0; i -= 4)
        serial_putc(hex[(val >> i) & 0xF]);
}
