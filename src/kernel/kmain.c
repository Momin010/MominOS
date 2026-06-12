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
#include "tty.h"
#include "memmap.h"
#include "../ai/diag/capture.h"

void kmain(uint64_t mb_info_phys) {
    serial_putc('2');
    serial_init();
    serial_putc('3');
    serial_print("[MominOS] Kernel alive\n");

    /* Parse the boot-protocol memory map into the architecture-neutral
       mem_regions table before bringing up the physical allocator. */
    memmap_parse_multiboot2(mb_info_phys);

    serial_print("[MominOS] About to init PMM\n");

    pmm_init(mem_regions, mem_region_count);
    serial_print("[PMM] initialized\n");
    serial_print("[PMM] free pages: ");
    serial_print_hex(pmm_free_pages());
    serial_print("\n");

    vmm_init(mem_regions, mem_region_count);
    serial_print("[VMM] phys(KERNEL_VMA+0x10000)=");
    serial_print_hex(vmm_phys(VMM_KERNEL_VMA + 0x10000));
    serial_print("\n");

    kheap_init();

    arch_init();
    sched_init();
    diag_init();

    idt_init();
    pic_remap();
    pic_mask_all();
    timer_init(100);
    keyboard_init();
    tty_init();
    pic_clear_mask(0);
    pic_clear_mask(1);
    pic_clear_mask(4);
    __asm__ volatile ("sti");
    serial_print("[IRQ] enabled\n");

    if (ata_init()) {
        if (vfs_mount_root()) {
            if (!vfs_self_test())
                serial_print("[VFS] self-test failed\n");
            /* launch the interactive shell as the first user process */
            if (elf_load_process("/bin/sh", 0, 0) == 0)
                serial_print("[ELF] shell spawn failed\n");
        }
    }

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
