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

/* Boot self-test: inject canned keystrokes into the TTY so the ring-3
   program's blocking read() exercises the real IRQ-wake path with no
   physical typing. Gives the userspace reader time to block first. */
static void tty_feed_str(const char *s) {
    while (*s) {
        tty_feed(*s);
        s++;
    }
}

/* Boot self-test driver: once the shell is up and blocked on stdin, feed
   it a canned command sequence so an automated `make run` demonstrates
   the whole read -> tokenize -> spawn -> waitpid chain with no typing.
   Each command gets time to run before the next line is delivered. */
static void boot_input_test(void *arg) {
    (void)arg;
    const char *cmds[] = {
        "ls /\n",
        "cat hello.txt\n",
        "echo hi from shell\n",
        "ls /bin\n",
        0,
    };
    uint64_t start = timer_ticks();

    /* let the shell reach its first blocking read() */
    while (timer_ticks() - start < 150)
        sched_yield();

    for (int i = 0; cmds[i] != 0; i++) {
        serial_print("[TEST] >> ");
        serial_print(cmds[i]);
        tty_feed_str(cmds[i]);
        start = timer_ticks();
        while (timer_ticks() - start < 100)
            sched_yield();
    }

    while (1)
        sched_yield();
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
    tty_init();
    pic_clear_mask(0);
    pic_clear_mask(1);
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

    thread_create(boot_input_test, 0);

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
