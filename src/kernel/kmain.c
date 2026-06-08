#include "serial.h"
#include "vga.h"

void kmain(void) {
    serial_init();
    serial_print("[MominOS] Kernel alive\n");

    vga_clear();
    vga_set_color(0x0A);  /* bright green */
    vga_print("MominOS 64-bit\n");
    vga_set_color(0x0F);
    vga_print("Kernel running in Long Mode\n");
    vga_print("Serial: COM1 active\n");

    serial_print("[MominOS] VGA initialized\n");
    serial_print("[MominOS] Halting (no scheduler yet)\n");

    while (1)
        __asm__ volatile ("hlt");
}
