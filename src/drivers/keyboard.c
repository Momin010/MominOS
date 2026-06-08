#include "keyboard.h"
#include "serial.h"
#include "tty.h"

#define KEYBOARD_DATA 0x60

static const char scancode_ascii[128] = {
    0,  27, '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '0', '-', '=', '\b', '\t',
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i',
    'o', 'p', '[', ']', '\n', 0,  'a', 's',
    'd', 'f', 'g', 'h', 'j', 'k', 'l', ';',
    '\'', '`', 0,  '\\', 'z', 'x', 'c', 'v',
    'b', 'n', 'm', ',', '.', '/', 0,  '*',
    0,  ' ',
};

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

void keyboard_init(void) {
    serial_print("[KBD] initialized\n");
}

void keyboard_irq(void) {
    uint8_t scancode = inb(KEYBOARD_DATA);

    if (scancode & 0x80)
        return;

    if (scancode < sizeof(scancode_ascii) && scancode_ascii[scancode])
        tty_feed(scancode_ascii[scancode]);
}

