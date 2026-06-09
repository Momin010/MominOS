#ifndef MEMMAP_H
#define MEMMAP_H

#include <stdint.h>

/* Architecture-neutral physical memory map. The boot/arch layer fills this in
   from whatever the firmware/boot protocol provides (Multiboot2 on x86 BIOS,
   the device tree on ARM later). Portable code (pmm, vmm) only ever reads this
   generic form and never the boot-protocol-specific structures. */

#define MEM_TYPE_AVAILABLE 1   /* usable RAM */

#define MEM_MAX_REGIONS 64

struct mem_region {
    uint64_t base;
    uint64_t length;
    uint32_t type;
};

extern struct mem_region mem_regions[MEM_MAX_REGIONS];
extern uint32_t mem_region_count;

/* Append a region to the global map (used by the boot-protocol parser). */
void memmap_add(uint64_t base, uint64_t length, uint32_t type);

/* Parse a Multiboot2 information structure (physical pointer) into mem_regions.
   x86-specific; lives in the boot layer. */
void memmap_parse_multiboot2(uint64_t mbi_phys);

#endif
