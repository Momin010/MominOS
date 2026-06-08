#include "serial.h"
#include "vga.h"
#include "pmm.h"
#include "vmm.h"
#include "kheap.h"
#include "sched.h"
#include "idt.h"
#include "keyboard.h"
#include "pic.h"
#include "timer.h"
#include "ata.h"
#include "vfs.h"
#include "arch.h"
#include "elf.h"

static void print_worker(void *arg) {
    char c = (char)(uint64_t)arg;
    uint64_t last = 0;

    while (1) {
        uint64_t ticks = timer_ticks();
        uint64_t bucket = ticks / 25;

        if (bucket != last) {
            serial_putc(c);
            last = bucket;
            sched_yield();
        }
    }
}

static void busy_worker(void *arg) {
    volatile uint64_t counter = (uint64_t)arg;

    while (1)
        counter++;
}

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

    vmm_init();
    serial_print("[VMM] phys(0x10000)=");
    serial_print_hex(vmm_phys(0x10000));
    serial_print("\n");

    kheap_init();

    arch_init();
    sched_init();

    idt_init();
    pic_remap();
    pic_mask_all();
    timer_init(100);
    keyboard_init();
    pic_clear_mask(0);
    pic_clear_mask(1);
    __asm__ volatile ("sti");
    serial_print("[IRQ] enabled\n");

    if (ata_init()) {
        if (vfs_mount_root()) {
            if (!vfs_self_test())
                serial_print("[VFS] self-test failed\n");
            if (!elf_spawn("/init"))
                serial_print("[ELF] spawn failed\n");
        }
    }

    thread_create(print_worker, (void *)(uint64_t)'A');
    thread_create(print_worker, (void *)(uint64_t)'B');
    thread_create(busy_worker, 0);

    vga_clear();
    vga_set_color(0x0A);
    vga_print("MominOS 64-bit\n");
    vga_set_color(0x0F);
    vga_print("Kernel running in Long Mode\n");
    vga_print("Serial: COM1 active\n");

    serial_print("[MominOS] VGA initialized\n");
    serial_print("[MominOS] Scheduler running\n");

    while (1)
        __asm__ volatile ("hlt");
}
