#include "vmm.h"
#include "pmm.h"
#include "memmap.h"
#include "serial.h"

#define PAGE_SIZE 4096
#define HUGE_PAGE_SIZE (2ULL * 1024 * 1024)
#define ENTRIES_PER_TABLE 512

#define PAGE_ADDR_MASK 0x000FFFFFFFFFF000ULL
#define HUGE_ADDR_MASK 0x000FFFFFFFE00000ULL
#define PTE_PS (1ULL << 7)

/* Higher-half layout. Must stay in sync with linker.ld and multiboot_entry.asm.
 *   KERNEL_VMA      = 0xFFFFFFFF80000000  PML4[511]  (kernel image, phys+KERNEL_VMA)
 *   DIRECT_MAP_BASE = 0xFFFF808000000000  PML4[257]  (all physical RAM)
 * Every page-table walk dereferences a physical frame through the direct map:
 * P2V(phys) = phys + DIRECT_MAP_BASE. PTEs always store physical addresses. */
#define KERNEL_VMA      0xFFFFFFFF80000000ULL
#define DIRECT_MAP_BASE 0xFFFF808000000000ULL

/* Boot tables span phys 0x1000..0x8FFF (8 frames); reserve them in the PMM. */
#define BOOT_PT_BASE 0x1000ULL
#define BOOT_PT_SIZE 0x8000ULL

/* How much of low physical RAM the boot tables already map (32MB), and thus how
 * much vmm_init can rely on being reachable via the direct map before it has
 * finished building the full map. Early page-table frames must land below this. */
#define BOOT_MAPPED_LIMIT (32ULL * 1024 * 1024)

/* kernel_pml4 holds the PHYSICAL address of the active kernel PML4. */
static uint64_t kernel_pml4_phys;

static inline uint64_t *p2v(uint64_t phys) {
    return (uint64_t *)(phys + DIRECT_MAP_BASE);
}

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
    uint64_t *page = p2v(phys);

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++)
        page[i] = 0;
}

/* Returns the PHYSICAL address of a fresh, zeroed page-table frame. */
static uint64_t alloc_table(void) {
    uint64_t phys = pmm_alloc();

    if (phys == 0) {
        serial_print("[VMM] out of page-table memory\n");
        while (1)
            __asm__ volatile ("cli; hlt");
    }

    zero_page(phys);
    return phys;
}

/* table is a VIRTUAL (direct-map) pointer to a page-table frame. Returns a
   VIRTUAL pointer to the next-level table, allocating it if absent. PTEs store
   physical addresses. */
static uint64_t *next_table(uint64_t *table, uint16_t index, uint64_t flags) {
    uint64_t next_phys;

    if (!(table[index] & VMM_PRESENT)) {
        next_phys = alloc_table();
        table[index] = (next_phys & PAGE_ADDR_MASK) | VMM_PRESENT | VMM_WRITABLE | (flags & VMM_USER);
    } else if (flags & VMM_USER) {
        table[index] |= VMM_USER;
    }

    return p2v(table[index] & PAGE_ADDR_MASK);
}

uint64_t vmm_kernel_pml4(void) {
    return kernel_pml4_phys;
}

/* Create a new user address space. With the kernel now living entirely in the
   higher half (PML4 entries 256..511), the lower half (0..255) belongs solely
   to user space. We SHARE the kernel's higher-half PML4 entries by value (so
   the kernel image, direct map and kheap stay mapped in every process) and
   leave the lower half EMPTY. User mappings (ELF segments, user stack) are
   added later, exclusively into the now-private lower half, and can therefore
   be freed wholesale on process exit without touching shared kernel pages. */
uint64_t vmm_create_address_space(void) {
    uint64_t pml4_phys = alloc_table();
    uint64_t *pml4 = p2v(pml4_phys);
    uint64_t *kpml4 = p2v(kernel_pml4_phys);

    /* Kernel higher-half PML4 slots are copied BY VALUE here, so every kernel
       PML4 entry (256 kheap, 257 direct map, 511 image) must already exist
       before the first process is created (they do: kheap_init + vmm_init run
       pre-process). Growth under an existing slot's sub-tables stays visible;
       but any future kernel mapping that lands in a NEW PML4 slot after
       processes exist would not propagate to them without an explicit sync. */

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++) {
        if (i < 256)
            pml4[i] = 0;                 /* private, empty user half */
        else
            pml4[i] = kpml4[i];          /* shared kernel half */
    }

    return pml4_phys;
}

/* Recursively free a user-half paging subtree rooted at table_phys. level 4 =
   PML4, 3 = PDPT, 2 = PD, 1 = PT. Every present non-huge entry points at a
   child table that is freed depth-first; leaf frames (level 1 entries and huge
   pages) are physical user frames and are freed too. Frees table_phys last.
   Only ever called on lower-half (user-private) subtrees. */
static void destroy_subtree(uint64_t table_phys, int level) {
    uint64_t *table = p2v(table_phys);

    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++) {
        uint64_t entry = table[i];

        if (!(entry & VMM_PRESENT))
            continue;

        if (level == 1) {
            /* leaf 4KB frame */
            pmm_free(entry & PAGE_ADDR_MASK);
        } else if (entry & PTE_PS) {
            /* leaf huge page (2MB at PD level / 1GB at PDPT level) */
            pmm_free(entry & HUGE_ADDR_MASK);
        } else {
            destroy_subtree(entry & PAGE_ADDR_MASK, level - 1);
        }
    }

    pmm_free(table_phys);
}

/* Tear down a process address space: free every page table and frame in the
   lower half (entries 0..255, exclusively user-private), then free the PML4
   frame itself. Entries 256..511 (shared kernel) are NEVER touched. The caller
   must guarantee pml4_phys is not the currently active CR3 and is not the
   shared kernel PML4. */
void vmm_destroy_address_space(uint64_t pml4_phys) {
    uint64_t *pml4;

    if (pml4_phys == 0 || pml4_phys == kernel_pml4_phys)
        return;

    pml4 = p2v(pml4_phys);
    for (uint64_t i = 0; i < 256; i++) {
        uint64_t entry = pml4[i];

        if (!(entry & VMM_PRESENT))
            continue;
        if (entry & PTE_PS) {
            pmm_free(entry & HUGE_ADDR_MASK);
            continue;
        }
        destroy_subtree(entry & PAGE_ADDR_MASK, 3);
    }

    pmm_free(pml4_phys);
}

void vmm_switch_pml4(uint64_t pml4) {
    write_cr3(pml4);
}

static void map_2m(uint64_t virt, uint64_t phys, uint64_t flags) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;

    uint64_t *pdpt = next_table(p2v(kernel_pml4_phys), pml4_i, flags);
    uint64_t *pd = next_table(pdpt, pdpt_i, flags);

    pd[pd_i] = (phys & HUGE_ADDR_MASK) | flags | VMM_PRESENT | PTE_PS;
}

static uint64_t *split_2m_page(uint64_t *pd, uint16_t index, uint64_t flags) {
    uint64_t old = pd[index];
    uint64_t base = old & HUGE_ADDR_MASK;
    uint64_t entry_flags = old & ~HUGE_ADDR_MASK;
    uint64_t pt_phys = alloc_table();
    uint64_t *pt = p2v(pt_phys);

    entry_flags = (entry_flags & ~PTE_PS) | (flags & VMM_USER);
    for (uint64_t i = 0; i < ENTRIES_PER_TABLE; i++)
        pt[i] = (base + i * PAGE_SIZE) | entry_flags;

    pd[index] = (pt_phys & PAGE_ADDR_MASK) | (entry_flags & (VMM_PRESENT | VMM_WRITABLE | VMM_USER));
    return pt;
}

/* Map [base, base+length) of physical RAM into the direct map at
   DIRECT_MAP_BASE + phys, using 2MB huge pages. */
static void direct_map_region(uint64_t base, uint64_t length) {
    uint64_t start = base & ~(HUGE_PAGE_SIZE - 1);
    uint64_t end = (base + length + HUGE_PAGE_SIZE - 1) & ~(HUGE_PAGE_SIZE - 1);

    for (uint64_t addr = start; addr < end; addr += HUGE_PAGE_SIZE)
        map_2m(DIRECT_MAP_BASE + addr, addr, VMM_PRESENT | VMM_WRITABLE);
}

void vmm_init(const struct mem_region *regions, uint32_t count) {
    uint64_t max_phys = 0;

    kernel_pml4_phys = alloc_table();

    /* Map the kernel image (and low RAM) into the higher half at KERNEL_VMA so
       the kernel keeps running after the CR3 switch. The boot tables flat-map
       0..32MB at KERNEL_VMA; replicate at least that here (the loaded image,
       .bss and boot stack all live below 32MB). */
    for (uint64_t addr = 0; addr < BOOT_MAPPED_LIMIT; addr += HUGE_PAGE_SIZE)
        map_2m(KERNEL_VMA + addr, addr, VMM_PRESENT | VMM_WRITABLE);

    /* Build the full physical direct map at DIRECT_MAP_BASE. */
    for (uint32_t i = 0; i < count; i++) {
        if (regions[i].type == MEM_TYPE_AVAILABLE) {
            direct_map_region(regions[i].base, regions[i].length);
            if (regions[i].base + regions[i].length > max_phys)
                max_phys = regions[i].base + regions[i].length;
        }
    }

    /* Page-table frames allocated above must lie within the boot-mapped window
       so the p2v() dereferences during this build were valid. Warn loudly if
       the assumption ever breaks. */
    if (pmm_alloc_high_water() > BOOT_MAPPED_LIMIT)
        serial_print("[VMM] WARN: early page tables exceeded boot-mapped 32MB\n");

    pmm_reserve(BOOT_PT_BASE, BOOT_PT_SIZE);

    write_cr3(kernel_pml4_phys);
    serial_print("[VMM] direct map initialized\n");
    (void)max_phys;
}

void vmm_map_in(uint64_t pml4_phys, uint64_t virt, uint64_t phys, uint64_t flags) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;

    uint64_t *pml4 = p2v(pml4_phys);
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
    vmm_map_in(kernel_pml4_phys, virt, phys, flags);
}

void vmm_unmap(uint64_t virt) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;

    uint64_t *pml4 = p2v(kernel_pml4_phys);
    uint64_t *pdpt;
    uint64_t *pd;
    uint64_t *pt;

    if (!(pml4[pml4_i] & VMM_PRESENT))
        return;
    pdpt = p2v(pml4[pml4_i] & PAGE_ADDR_MASK);

    if (!(pdpt[pdpt_i] & VMM_PRESENT))
        return;
    pd = p2v(pdpt[pdpt_i] & PAGE_ADDR_MASK);

    if (!(pd[pd_i] & VMM_PRESENT))
        return;
    if (pd[pd_i] & PTE_PS) {
        pd[pd_i] = 0;
        invlpg(virt);
        return;
    }

    pt = p2v(pd[pd_i] & PAGE_ADDR_MASK);
    pt[pt_i] = 0;
    invlpg(virt);
}

/* Walk the CURRENTLY ACTIVE address space (caller's CR3) and return 1 if virt
   maps to a present USER page. Syscalls run on the calling process's CR3 (no
   CR3 switch on kernel entry, the kernel half is shared in), so this sees the
   process's lower-half user mappings, which the kernel-pml4 walk in vmm_phys
   cannot. The VMM_USER bit must be set at every level for a CPL3-accessible
   page. The direct map is shared into every address space, so the P2V
   dereferences below are valid mid-syscall. */
static int user_page_present(uint64_t virt) {
    uint16_t pml4_i = (virt >> 39) & 0x1FF;
    uint16_t pdpt_i = (virt >> 30) & 0x1FF;
    uint16_t pd_i = (virt >> 21) & 0x1FF;
    uint16_t pt_i = (virt >> 12) & 0x1FF;
    uint64_t need = VMM_PRESENT | VMM_USER;
    uint64_t *pml4 = p2v(read_cr3() & PAGE_ADDR_MASK);
    uint64_t *pdpt;
    uint64_t *pd;
    uint64_t *pt;

    if ((pml4[pml4_i] & need) != need)
        return 0;
    pdpt = p2v(pml4[pml4_i] & PAGE_ADDR_MASK);

    if ((pdpt[pdpt_i] & need) != need)
        return 0;
    if (pdpt[pdpt_i] & PTE_PS)           /* 1GB user page */
        return 1;
    pd = p2v(pdpt[pdpt_i] & PAGE_ADDR_MASK);

    if ((pd[pd_i] & need) != need)
        return 0;
    if (pd[pd_i] & PTE_PS)               /* 2MB user page */
        return 1;
    pt = p2v(pd[pd_i] & PAGE_ADDR_MASK);

    return (pt[pt_i] & need) == need;
}

/* Validate a user buffer [ptr, ptr+len): it must lie wholly in the user half
   with no wrap (rejects kernel-half and non-canonical pointers), and every
   page it spans must be a present user page in the caller's address space
   (so the kernel never faults dereferencing an unmapped/kernel pointer). */
int vmm_user_range_ok(uint64_t ptr, uint64_t len) {
    uint64_t end;
    uint64_t page;

    if (len == 0)
        return ptr <= VMM_USER_MAX;     /* empty buffer: pointer just in range */

    end = ptr + len;
    if (end < ptr)                      /* arithmetic wrap */
        return 0;
    if (end > VMM_USER_MAX)             /* spills into / past the user half */
        return 0;

    /* confirm every page in the range is mapped and user-accessible. */
    for (page = ptr & ~(PAGE_SIZE - 1); page < end; page += PAGE_SIZE) {
        if (!user_page_present(page))
            return 0;
    }
    return 1;
}

/* Validate a NUL-terminated user string at ptr: in the user half, terminating
   within max_len bytes, and on present user pages. The page check before each
   read guarantees the kernel never faults scanning it. */
int vmm_user_str_ok(uint64_t ptr, uint64_t max_len) {
    const char *s = (const char *)ptr;
    uint64_t i;

    if (ptr == 0 || ptr >= VMM_USER_MAX)
        return 0;
    if (!user_page_present(ptr & ~(PAGE_SIZE - 1)))
        return 0;

    for (i = 0; i < max_len; i++) {
        uint64_t cur = ptr + i;

        if (cur >= VMM_USER_MAX)         /* byte i would leave the user half */
            return 0;
        /* crossing into a new page: validate it before touching it. */
        if ((cur & (PAGE_SIZE - 1)) == 0 && !user_page_present(cur))
            return 0;
        if (s[i] == 0)
            return 1;
    }
    return 0;                            /* no terminator within max_len */
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

    if (kernel_pml4_phys == 0)
        kernel_pml4_phys = read_cr3() & PAGE_ADDR_MASK;
    pml4 = p2v(kernel_pml4_phys);

    if (!(pml4[pml4_i] & VMM_PRESENT))
        return 0;
    pdpt = p2v(pml4[pml4_i] & PAGE_ADDR_MASK);

    if (!(pdpt[pdpt_i] & VMM_PRESENT))
        return 0;
    pd = p2v(pdpt[pdpt_i] & PAGE_ADDR_MASK);

    if (!(pd[pd_i] & VMM_PRESENT))
        return 0;
    if (pd[pd_i] & PTE_PS)
        return (pd[pd_i] & HUGE_ADDR_MASK) | (virt & (HUGE_PAGE_SIZE - 1));

    pt = p2v(pd[pd_i] & PAGE_ADDR_MASK);
    if (!(pt[pt_i] & VMM_PRESENT))
        return 0;

    return (pt[pt_i] & PAGE_ADDR_MASK) | (virt & (PAGE_SIZE - 1));
}
