#include "pmm.h"
#include "memmap.h"
#include "serial.h"

/* Provided by the linker: physical end of the loaded kernel image. */
extern uint8_t kernel_phys_end[];

#define PAGE_SIZE 4096
#define MAX_PAGES (4ULL * 1024 * 1024 * 1024 / PAGE_SIZE)
#define BITMAP_SIZE (MAX_PAGES / 8)

static uint8_t bitmap[BITMAP_SIZE];
static uint64_t total_pages = 0;
static uint64_t free_pages = 0;

static inline void bitmap_set(uint64_t bit) {
    bitmap[bit / 8] |= (1 << (bit % 8));
}

static inline void bitmap_clear(uint64_t bit) {
    bitmap[bit / 8] &= ~(1 << (bit % 8));
}

static inline int bitmap_test(uint64_t bit) {
    return bitmap[bit / 8] & (1 << (bit % 8));
}

static void mark_region(uint64_t base, uint64_t length, int used) {
    uint64_t start_page = (base + PAGE_SIZE - 1) / PAGE_SIZE;
    uint64_t end_page = (base + length) / PAGE_SIZE;

    if (start_page >= end_page) return;
    if (end_page > MAX_PAGES) end_page = MAX_PAGES;

    for (uint64_t p = start_page; p < end_page; p++) {
        if (used) {
            if (!bitmap_test(p)) {
                bitmap_set(p);
                if (free_pages > 0) free_pages--;
            }
        } else {
            if (bitmap_test(p)) {
                bitmap_clear(p);
                free_pages++;
            }
        }
    }
}

void pmm_init(const struct mem_region *regions, uint32_t count) {
    for (uint64_t i = 0; i < BITMAP_SIZE; i++)
        bitmap[i] = 0xFF;
    free_pages = 0;

    serial_print("[PMM] memory regions: ");
    serial_print_hex(count);
    serial_print("\n");

    for (uint32_t i = 0; i < count; i++) {
        uint64_t base = regions[i].base;
        uint64_t length = regions[i].length;
        uint32_t type = regions[i].type;

        serial_print("[PMM] Entry ");
        serial_print_hex(i);
        serial_print(": base=");
        serial_print_hex(base);
        serial_print(" len=");
        serial_print_hex(length);
        serial_print(" type=");
        serial_print_hex(type);
        serial_print("\n");

        if (type == MEM_TYPE_AVAILABLE) {
            mark_region(base, length, 0);
            total_pages += length / PAGE_SIZE;
        }
    }

    /* Reserve low memory (IVT/BIOS/boot page tables) and the loaded kernel
       image itself (now linked at 1MB, so it sits inside a usable region).
       Round the kernel end up to a full page: mark_region truncates the end
       down, which would otherwise leave the kernel's final partial page free
       and let pmm_alloc hand out memory that overlaps the kernel image. */
    uint64_t kend = ((uint64_t)kernel_phys_end + PAGE_SIZE - 1) & ~(uint64_t)(PAGE_SIZE - 1);
    mark_region(0, 0x100000, 1);
    mark_region(0x100000, kend - 0x100000, 1);

    serial_print("[PMM] Total usable pages: ");
    serial_print_hex(total_pages);
    serial_print("\n");
    serial_print("[PMM] Free pages: ");
    serial_print_hex(free_pages);
    serial_print("\n");
}

void pmm_reserve(uint64_t base, uint64_t length) {
    mark_region(base, length, 1);
}

uint64_t pmm_alloc(void) {
    for (uint64_t i = 0; i < MAX_PAGES; i++) {
        if (!bitmap_test(i)) {
            bitmap_set(i);
            free_pages--;
            return i * PAGE_SIZE;
        }
    }
    return 0;
}

void pmm_free(uint64_t addr) {
    uint64_t page = addr / PAGE_SIZE;
    if (page < MAX_PAGES && bitmap_test(page)) {
        bitmap_clear(page);
        free_pages++;
    }
}

uint64_t pmm_free_pages(void) {
    return free_pages;
}
