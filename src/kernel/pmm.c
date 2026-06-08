#include "pmm.h"
#include "serial.h"

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

void pmm_init(void) {
    for (uint64_t i = 0; i < BITMAP_SIZE; i++)
        bitmap[i] = 0xFF;
    free_pages = 0;

    uint16_t *count_ptr = (uint16_t *)0x5FF8;
    uint16_t entry_count = *count_ptr;
    uint8_t *entries = (uint8_t *)0x6000;

    serial_print("[PMM] E820 entry count: ");
    serial_print_hex(entry_count);
    serial_print("\n");

    for (uint16_t i = 0; i < entry_count; i++) {
        uint64_t *entry = (uint64_t *)(entries + i * 24);
        uint64_t base = entry[0];
        uint64_t length = entry[1];
        uint32_t type = *(uint32_t *)&entry[2];

        serial_print("[PMM] Entry ");
        serial_print_hex(i);
        serial_print(": base=");
        serial_print_hex(base);
        serial_print(" len=");
        serial_print_hex(length);
        serial_print(" type=");
        serial_print_hex(type);
        serial_print("\n");

        if (type == 1) {
            mark_region(base, length, 0);
            total_pages += length / PAGE_SIZE;
        }
    }

    mark_region(0, 0x100000, 1);

    serial_print("[PMM] Total usable pages: ");
    serial_print_hex(total_pages);
    serial_print("\n");
    serial_print("[PMM] Free pages: ");
    serial_print_hex(free_pages);
    serial_print("\n");
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
