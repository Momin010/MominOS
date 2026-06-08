#include "serial.h"

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
    outb(COM1 + 4, 0x0B);  /* RTS/DSR */
}

void serial_putc(char c) {
    while ((inb(COM1 + 5) & 0x20) == 0);
    outb(COM1, c);
}

void serial_print(const char *s) {
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
