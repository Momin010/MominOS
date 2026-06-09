#include "vmm.h"
#include "pmm.h"
#include "serial.h"

#define PAGE_SIZE 4096
#define HUGE_PAGE_SIZE (2ULL * 1024 * 1024)
#define ENTRIES_PER_TABLE 512

#define PAGE_ADDR_MASK 0x000FFFFFFFFFF000ULL
#define HUGE_ADDR_MASK 0x000FFFFFFFE00000ULL
#define PTE_PS (1ULL << 7)

static uint64_t *kernel_pml4;

static inline uint64_t read_cr3(void) {
    uint64_t cr3;
    __asm__ volatile ("mov %%cr3, %0" : "=r"(cr3));
    return cr3;
}

static inline void write_cr3(uint64_t cr3) {
    __asm__ volatile ("mov %0, %%cr3" : : "r"(cr3) : "memory");
}

static inline void invlpg(uint64_t virt) {
    __asm__ volatile ("invlpg (%0)" : : "r"(virt) : "memory");
}

static void zero_page(uint64_t phys) {
    uint64_t *page = (uint64_t *)phys;

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++)
        page[i] = 0;
}

static uint64_t *alloc_table(void) {
    uint64_t phys = pmm_alloc();

    if (phys == 0) {
        serial_print("[VMM] out of page-table memory\n");
        while (1)
            __asm__ volatile ("cli; hlt");
    }

    zero_page(phys);
    return (uint64_t *)phys;
}

static uint64_t *next_table(uint64_t *table, uint16_t index, uint64_t flags) {
    uint64_t *next;

    if (!(table[index] & VMM_PRESENT)) {
        next = alloc_table();
        table[index] = ((uint64_t)next & PAGE_ADDR_MASK) | VMM_PRESENT | VMM_WRITABLE | (flags & VMM_USER);
    } else if (flags & VMM_USER) {
        table[index] |= VMM_USER;
    }

    return (uint64_t *)(table[index] & PAGE_ADDR_MASK);
}

uint64_t vmm_kernel_pml4(void) {
    return (uint64_t)kernel_pml4;
}

/* Deep-copy one level of the paging hierarchy. Present, non-huge entries are
   recursively cloned into freshly allocated tables; huge-page and absent
   entries are copied by value. This makes the subtree private to the new
   address space so user mappings (which live in the lower half, sharing the
   same PML4 subtree as the kernel direct map) do not mutate page tables that
   other processes also reference. */
static uint64_t clone_table(uint64_t table_phys, int level) {
    uint64_t *src = (uint64_t *)table_phys;
    uint64_t *dst = alloc_table();

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++) {
        uint64_t entry = src[i];

        if (!(entry & VMM_PRESENT)) {
            dst[i] = entry;
            continue;
        }

        /* Leaf PT entries (level 1) and huge pages map frames directly: copy
           the mapping by value so it points at the same physical frame. */
        if (level == 1 || (entry & PTE_PS)) {
            dst[i] = entry;
            continue;
        }

        {
            uint64_t child = entry & PAGE_ADDR_MASK;
            uint64_t copy = clone_table(child, level - 1);
            dst[i] = (copy & PAGE_ADDR_MASK) | (entry & ~PAGE_ADDR_MASK);
        }
    }

    return (uint64_t)dst;
}

uint64_t vmm_create_address_space(void) {
    uint64_t *pml4 = alloc_table();

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++) {
        uint64_t entry = kernel_pml4[i];

        /* Lower half (indices 0..255) holds user mappings and the kernel
           direct map; give each process a private copy of that subtree so
           per-process user mappings stay isolated. The higher half (kernel
           heap and other dynamic kernel mappings) is shared by value so it
           stays visible across all address spaces. */
        if (i < 256 && (entry & VMM_PRESENT) && !(entry & PTE_PS)) {
            uint64_t copy = clone_table(entry & PAGE_ADDR_MASK, 3);
            pml4[i] = (copy & PAGE_ADDR_MASK) | (entry & ~PAGE_ADDR_MASK);
        } else {
            pml4[i] = entry;
        }
    }

    return (uint64_t)pml4;
}

void vmm_switch_pml4(uint64_t pml4) {
    write_cr3(pml4);
}

static void map_2m(uint64_t virt, uint64_t phys, uint64_t flags) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;

    uint64_t *pdpt = next_table(kernel_pml4, pml4_i, flags);
    uint64_t *pd = next_table(pdpt, pdpt_i, flags);

    pd[pd_i] = (phys & HUGE_ADDR_MASK) | flags | VMM_PRESENT | PTE_PS;
}

static uint64_t *split_2m_page(uint64_t *pd, uint16_t index, uint64_t flags) {
    uint64_t old = pd[index];
    uint64_t base = old & HUGE_ADDR_MASK;
    uint64_t entry_flags = old & ~HUGE_ADDR_MASK;
    uint64_t *pt = alloc_table();

    entry_flags = (entry_flags & ~PTE_PS) | (flags & VMM_USER);
    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++)
        pt[i] = (base + i * PAGE_SIZE) | entry_flags;

    pd[index] = ((uint64_t)pt & PAGE_ADDR_MASK) | (entry_flags & (VMM_PRESENT | VMM_WRITABLE | VMM_USER));
    return pt;
}

static void direct_map_region(uint64_t base, uint64_t length) {
    uint64_t start = base & ~(HUGE_PAGE_SIZE - 1);
    uint64_t end = (base + length + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    for (uint64_t addr = start; addr < end; addr += HUGE_PAGE_SIZE)
        map_2m(addr, addr, VMM_PRESENT | VMM_WRITABLE);
}

void vmm_init(void) {
    uint16_t *count_ptr = (uint16_t *)0x5FF8;
    uint16_t entry_count = *count_ptr;
    uint8_t *entries = (uint8_t *)0x6000;

    kernel_pml4 = alloc_table();

    for (uint16_t i = 0; i < entry_count; i++) {
        uint64_t *entry = (uint64_t *)(entries + i * 24);
        uint64_t base = entry[0];
        uint64_t length = entry[1];
        uint32_t type = *(uint32_t *)&entry[2];

        if (type == 1)
            direct_map_region(base, length);
    }

    pmm_reserve(0x1000, 0x4000);

    write_cr3((uint64_t)kernel_pml4);
    serial_print("[VMM] direct map initialized\n");
}

void vmm_map_in(uint64_t pml4_phys, uint64_t virt, uint64_t phys, uint64_t flags) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;

    uint64_t *pml4 = (uint64_t *)pml4_phys;
    uint64_t *pdpt = next_table(pml4, pml4_i, flags);
    uint64_t *pd = next_table(pdpt, pdpt_i, flags);
    uint64_t *pt;

    if ((pd[pd_i] & (VMM_PRESENT | PTE_PS)) == (VMM_PRESENT | PTE_PS))
        pt = split_2m_page(pd, pd_i, flags);
    else
        pt = next_table(pd, pd_i, flags);

    pt[pt_i] = (phys & PAGE_ADDR_MASK) | flags | VMM_PRESENT;
    invlpg(virt);
}

void vmm_map(uint64_t virt, uint64_t phys, uint64_t flags) {
    vmm_map_in((uint64_t)kernel_pml4, virt, phys, flags);
}

void vmm_unmap(uint64_t virt) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;

    uint64_t *pml4 = kernel_pml4;
    uint64_t *pdpt;
    uint64_t *pd;
    uint64_t *pt;

    if (!(pml4[pml4_i] & VMM_PRESENT))
        return;
    pdpt = (uint64_t *)(pml4[pml4_i] & PAGE_ADDR_MASK);

    if (!(pdpt[pdpt_i] & VMM_PRESENT))
        return;
    pd = (uint64_t *)(pdpt[pdpt_i] & PAGE_ADDR_MASK);

    if (!(pd[pd_i] & VMM_PRESENT))
        return;
    if (pd[pd_i] & PTE_PS) {
        pd[pd_i] = 0;
        invlpg(virt);
        return;
    }

    pt = (uint64_t *)(pd[pd_i] & PAGE_ADDR_MASK);
    pt[pt_i] = 0;
    invlpg(virt);
}

uint64_t vmm_phys(uint64_t virt) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;

    uint64_t *pml4;
    uint64_t *pdpt;
    uint64_t *pd;
    uint64_t *pt;

    if (kernel_pml4 == 0)
        kernel_pml4 = (uint64_t *)(read_cr3() & PAGE_ADDR_MASK);
    pml4 = kernel_pml4;

    if (!(pml4[pml4_i] & VMM_PRESENT))
        return 0;
    pdpt = (uint64_t *)(pml4[pml4_i] & PAGE_ADDR_MASK);

    if (!(pdpt[pdpt_i] & VMM_PRESENT))
        return 0;
    pd = (uint64_t *)(pdpt[pdpt_i] & PAGE_ADDR_MASK);

    if (!(pd[pd_i] & VMM_PRESENT))
        return 0;
    if (pd[pd_i] & PTE_PS)
        return (pd[pd_i] & HUGE_ADDR_MASK) | (virt & (HUGE_PAGE_SIZE - 1));

    pt = (uint64_t *)(pd[pd_i] & PAGE_ADDR_MASK);
    if (!(pt[pt_i] & VMM_PRESENT))
        return 0;

    return (pt[pt_i] & PAGE_ADDR_MASK) | (virt & (PAGE_SIZE - 1));
}
