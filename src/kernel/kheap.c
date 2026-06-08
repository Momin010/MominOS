#include "kheap.h"
#include "pmm.h"
#include "serial.h"
#include "vmm.h"

#define PAGE_SIZE 4096ULL
#define KHEAP_ALIGN 64ULL
#define KHEAP_BASE 0xFFFF800000000000ULL
#define KHEAP_MAX_SIZE (4ULL * 1024 * 1024 * 1024)
#define KHEAP_INITIAL_SIZE (1024ULL * 1024)
#define KHEAP_MAGIC 0xC0FFEE42U
#define KHEAP_FREED_MAGIC 0xDEADFACEU
#define KHEAP_MIN_SPLIT KHEAP_ALIGN
#define KHEAP_TEST_LARGE_SIZE (256ULL * 1024 * 1024)

typedef struct heap_block {
    uint64_t size;
    struct heap_block *next;
    struct heap_block *prev;
    uint32_t magic;
    uint32_t free;
} heap_block_t;

static heap_block_t *heap_head;
static uint64_t heap_brk = KHEAP_BASE;
static uint64_t heap_limit = KHEAP_BASE + KHEAP_MAX_SIZE;

static uint64_t align_up(uint64_t value, uint64_t align) {
    return (value + align - 1) & ~(align - 1);
}

static inline uint64_t irq_save(void) {
    uint64_t flags;

    __asm__ volatile ("pushfq; pop %0; cli" : "=r"(flags) : : "memory");
    return flags;
}

static inline void irq_restore(uint64_t flags) {
    __asm__ volatile ("push %0; popfq" : : "r"(flags) : "memory");
}

static void *mem_zero(void *ptr, size_t size) {
    uint8_t *out = ptr;

    for (size_t i = 0; i < size; i++)
        out[i] = 0;
    return ptr;
}

static uint64_t header_size(void) {
    return align_up(sizeof(heap_block_t), KHEAP_ALIGN);
}

static void heap_halt(const char *message) {
    serial_print(message);
    while (1)
        __asm__ volatile ("cli; hlt");
}

static int block_valid(heap_block_t *block) {
    return block != 0 && block->magic == KHEAP_MAGIC;
}

static heap_block_t *last_block(void) {
    heap_block_t *block = heap_head;

    if (block == 0)
        return 0;

    while (block->next != 0)
        block = block->next;
    return block;
}

static heap_block_t *next_adjacent(heap_block_t *block) {
    uint64_t next_addr = (uint64_t)block + header_size() + block->size;

    if (next_addr >= heap_brk)
        return 0;
    return (heap_block_t *)next_addr;
}

static void split_block(heap_block_t *block, uint64_t size) {
    uint64_t hsize = header_size();
    uint64_t remaining = block->size - size;
    heap_block_t *split;

    if (remaining < hsize + KHEAP_MIN_SPLIT)
        return;

    split = (heap_block_t *)((uint64_t)block + hsize + size);
    split->size = remaining - hsize;
    split->next = block->next;
    split->prev = block;
    split->magic = KHEAP_MAGIC;
    split->free = 1;

    if (split->next != 0)
        split->next->prev = split;

    block->size = size;
    block->next = split;
}

static void coalesce_next(heap_block_t *block) {
    heap_block_t *next = block->next;

    if (next == 0 || !next->free)
        return;

    if (next_adjacent(block) != next)
        return;

    if (!block_valid(next))
        heap_halt("[KHEAP] corrupt next block\n");

    block->size += header_size() + next->size;
    block->next = next->next;
    if (block->next != 0)
        block->next->prev = block;

    next->magic = KHEAP_FREED_MAGIC;
    next->next = 0;
    next->prev = 0;
}

static int map_heap_pages(uint64_t start, uint64_t bytes) {
    uint64_t mapped = 0;

    while (mapped < bytes) {
        uint64_t virt = start + mapped;
        uint64_t phys = pmm_alloc();

        if (phys == 0) {
            for (uint64_t rollback = 0; rollback < mapped; rollback += PAGE_SIZE) {
                uint64_t rollback_virt = start + rollback;
                uint64_t rollback_phys = vmm_phys(rollback_virt);

                if (rollback_phys != 0) {
                    vmm_unmap(rollback_virt);
                    pmm_free(rollback_phys);
                }
            }
            return 0;
        }

        vmm_map(virt, phys, VMM_WRITABLE);
        mapped += PAGE_SIZE;
    }

    return 1;
}

static int grow_heap(uint64_t min_payload) {
    uint64_t hsize = header_size();
    uint64_t old_brk = heap_brk;
    uint64_t grow_size = align_up(hsize + min_payload, PAGE_SIZE);
    heap_block_t *block;
    heap_block_t *tail;

    if (grow_size < KHEAP_INITIAL_SIZE)
        grow_size = KHEAP_INITIAL_SIZE;

    if (old_brk + grow_size > heap_limit)
        return 0;

    if (!map_heap_pages(old_brk, grow_size))
        return 0;

    heap_brk = old_brk + grow_size;
    block = (heap_block_t *)old_brk;
    block->size = grow_size - hsize;
    block->next = 0;
    block->prev = 0;
    block->magic = KHEAP_MAGIC;
    block->free = 1;

    if (heap_head == 0) {
        heap_head = block;
        return 1;
    }

    tail = last_block();
    if (!block_valid(tail))
        heap_halt("[KHEAP] corrupt tail block\n");

    tail->next = block;
    block->prev = tail;

    if (tail->free)
        coalesce_next(tail);

    return 1;
}

static heap_block_t *find_free_block(uint64_t size) {
    heap_block_t *block = heap_head;

    while (block != 0) {
        if (!block_valid(block))
            heap_halt("[KHEAP] corrupt block header\n");

        if (block->free && block->size >= size)
            return block;

        block = block->next;
    }

    return 0;
}

void kheap_init(void) {
    heap_head = 0;
    heap_brk = KHEAP_BASE;
    heap_limit = KHEAP_BASE + KHEAP_MAX_SIZE;

    if (!grow_heap(KHEAP_INITIAL_SIZE - header_size()))
        heap_halt("[KHEAP] init failed\n");

    serial_print("[KHEAP] initialized\n");
}

void *kmalloc(size_t size) {
    uint64_t flags = irq_save();
    uint64_t aligned_size;
    heap_block_t *block;

    if (size == 0) {
        irq_restore(flags);
        return 0;
    }

    aligned_size = align_up((uint64_t)size, KHEAP_ALIGN);

    block = find_free_block(aligned_size);
    if (block == 0) {
        if (!grow_heap(aligned_size)) {
            irq_restore(flags);
            return 0;
        }
        block = find_free_block(aligned_size);
    }

    if (block == 0) {
        irq_restore(flags);
        return 0;
    }

    split_block(block, aligned_size);
    block->free = 0;

    irq_restore(flags);
    return (void *)((uint64_t)block + header_size());
}

void *kzalloc(size_t size) {
    void *ptr = kmalloc(size);

    if (ptr != 0)
        mem_zero(ptr, size);
    return ptr;
}

void kfree(void *ptr) {
    uint64_t flags;
    heap_block_t *block;

    if (ptr == 0)
        return;

    flags = irq_save();
    block = (heap_block_t *)((uint64_t)ptr - header_size());
    if (!block_valid(block))
        heap_halt("[KHEAP] invalid free\n");

    if (block->free)
        heap_halt("[KHEAP] double free\n");

    block->free = 1;
    coalesce_next(block);

    if (block->prev != 0 && block->prev->free) {
        block = block->prev;
        coalesce_next(block);
    }

    irq_restore(flags);
}

static int heap_check(void) {
    heap_block_t *block = heap_head;
    heap_block_t *prev = 0;

    while (block != 0) {
        if (!block_valid(block))
            return 0;

        if (block->prev != prev)
            return 0;

        if (((uint64_t)block & (KHEAP_ALIGN - 1)) != 0)
            return 0;

        if (block->next != 0 && next_adjacent(block) != block->next)
            return 0;

        prev = block;
        block = block->next;
    }

    return 1;
}

static uint32_t test_rand(uint32_t *state) {
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

int kheap_self_test(void) {
    void *ptrs[256];
    uint32_t rng = 0x12345678U;
    void *large;

    for (uint64_t i = 0; i < 256; i++)
        ptrs[i] = 0;

    for (uint64_t i = 0; i < 4096; i++) {
        uint64_t slot = test_rand(&rng) & 255U;
        uint64_t size = (test_rand(&rng) & 8191U) + 1U;

        if (ptrs[slot] != 0) {
            kfree(ptrs[slot]);
            ptrs[slot] = 0;
        }

        ptrs[slot] = kmalloc(size);
        if (ptrs[slot] == 0 || (((uint64_t)ptrs[slot] & (KHEAP_ALIGN - 1)) != 0))
            return 0;

        if (!heap_check())
            return 0;
    }

    for (uint64_t i = 0; i < 256; i++) {
        if (ptrs[i] != 0) {
            kfree(ptrs[i]);
            ptrs[i] = 0;
        }
    }

    if (!heap_check())
        return 0;

    large = kmalloc(KHEAP_TEST_LARGE_SIZE);
    if (large == 0)
        return 0;

    if (((uint64_t)large & (KHEAP_ALIGN - 1)) != 0)
        return 0;

    kfree(large);

    if (!heap_check())
        return 0;

    if (heap_head == 0 || heap_head->next != 0 || !heap_head->free)
        return 0;

    serial_print("[KHEAP] self-test passed\n");
    return 1;
}
