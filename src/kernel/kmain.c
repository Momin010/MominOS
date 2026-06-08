#include "serial.h"
#include "vga.h"
#include "pmm.h"
#include "idt.h"

void kmain(void) {
    serial_putc('2');
    serial_init();
    serial_putc('3');
    serial_print("[MominOS] Kernel alive\n");
    serial_print("[MominOS] About to init PMM\n");

    pmm_init();
    serial_print("[PMM] initialized\n");
    serial_print("[PMM] free pages: ");
    serial_print_hex(pmm_free_pages());
    serial_print("\n");

    idt_init();

    vga_clear();
    vga_set_color(0x0A);
    vga_print("MominOS 64-bit\n");
    vga_set_color(0x0F);
    vga_print("Kernel running in Long Mode\n");
    vga_print("Serial: COM1 active\n");

    serial_print("[MominOS] VGA initialized\n");
    serial_print("[MominOS] Halting (no scheduler yet)\n");

    while (1)
        __asm__ volatile ("hlt");
}
