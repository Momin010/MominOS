/* Higher-half kernel link script.
 *
 * The kernel is LINKED at a high virtual base (KERNEL_VMA, the -2GB area =
 * PML4[511]) but LOADED at physical 1MB (LMA). AT() keeps every section's
 * load address low while its virtual address is high, so GRUB loads the image
 * at 1MB and the CPU runs it at the high VMA once the boot page tables map
 * KERNEL_VMA -> phys 0.
 *
 * Convention: phys = virt - KERNEL_VMA for all kernel image symbols. In
 * particular kernel_phys_end is exported as a PHYSICAL address so the PMM can
 * reserve the loaded image correctly. */

KERNEL_VMA = 0xFFFFFFFF80000000;

ENTRY(_start)
SECTIONS
{
    . = KERNEL_VMA + 1M;

    .multiboot : AT(ADDR(.multiboot) - KERNEL_VMA) {
        *(.multiboot)
    }
    .text : AT(ADDR(.text) - KERNEL_VMA) {
        *(.text*)
    }
    .rodata : AT(ADDR(.rodata) - KERNEL_VMA) {
        *(.rodata*)
    }
    .data : AT(ADDR(.data) - KERNEL_VMA) {
        *(.data*)
    }
    .bss : AT(ADDR(.bss) - KERNEL_VMA) {
        *(.bss*)
        *(COMMON)
    }

    /* Physical end of the loaded image (for the PMM to reserve). */
    kernel_phys_end = . - KERNEL_VMA;
}
