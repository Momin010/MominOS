#include "memmap.h"
#include "serial.h"

struct mem_region mem_regions[MEM_MAX_REGIONS];
uint32_t mem_region_count = 0;

void memmap_add(uint64_t base, uint64_t length, uint32_t type) {
    if (mem_region_count >= MEM_MAX_REGIONS)
        return;
    mem_regions[mem_region_count].base = base;
    mem_regions[mem_region_count].length = length;
    mem_regions[mem_region_count].type = type;
    mem_region_count++;
}

/* --- Multiboot2 (x86 BIOS via GRUB) --- */

#define MB2_TAG_END     0
#define MB2_TAG_MMAP    6

struct mb2_tag {
    uint32_t type;
    uint32_t size;
};

struct mb2_mmap_tag {
    uint32_t type;
    uint32_t size;
    uint32_t entry_size;
    uint32_t entry_version;
    /* entries follow */
};

struct mb2_mmap_entry {
    uint64_t base;
    uint64_t length;
    uint32_t type;     /* 1 = available */
    uint32_t reserved;
};

void memmap_parse_multiboot2(uint64_t mbi_phys) {
    uint8_t *base = (uint8_t *)mbi_phys;

    /* First 8 bytes: total_size (u32), reserved (u32). Tags start at +8. */
    uint32_t total_size = *(uint32_t *)base;
    uint8_t *p = base + 8;
    uint8_t *end = base + total_size;

    while (p < end) {
        struct mb2_tag *tag = (struct mb2_tag *)p;

        if (tag->type == MB2_TAG_END)
            break;

        if (tag->type == MB2_TAG_MMAP) {
            struct mb2_mmap_tag *mmap = (struct mb2_mmap_tag *)p;
            uint32_t entry_size = mmap->entry_size;
            uint8_t *e = p + sizeof(struct mb2_mmap_tag);
            uint8_t *tag_end = p + tag->size;

            while (e + entry_size <= tag_end) {
                struct mb2_mmap_entry *me = (struct mb2_mmap_entry *)e;
                memmap_add(me->base, me->length, me->type);
                e += entry_size;
            }
        }

        /* Tags are 8-byte aligned. */
        p += (tag->size + 7) & ~7u;
    }

    serial_print("[MMAP] regions: ");
    serial_print_hex(mem_region_count);
    serial_print("\n");
}
