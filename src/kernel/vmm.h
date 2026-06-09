#ifndef VMM_H
#define VMM_H

#include <stdint.h>
#include "memmap.h"

#define VMM_PRESENT  (1ULL << 0)
#define VMM_WRITABLE (1ULL << 1)
#define VMM_USER     (1ULL << 2)
#define VMM_NOEXEC   (1ULL << 63)

/* Higher-half layout (kept in sync with linker.ld and multiboot_entry.asm).
   The kernel runs at KERNEL_VMA; all physical RAM is accessible at
   VMM_DIRECT_MAP_BASE + phys. Use VMM_P2V to dereference a physical frame. */
#define VMM_KERNEL_VMA      0xFFFFFFFF80000000ULL
#define VMM_DIRECT_MAP_BASE 0xFFFF808000000000ULL
#define VMM_P2V(phys)       ((void *)((uint64_t)(phys) + VMM_DIRECT_MAP_BASE))

void vmm_init(const struct mem_region *regions, uint32_t count);
uint64_t vmm_kernel_pml4(void);
uint64_t vmm_create_address_space(void);
void vmm_destroy_address_space(uint64_t pml4);
void vmm_switch_pml4(uint64_t pml4);
void vmm_map_in(uint64_t pml4, uint64_t virt, uint64_t phys, uint64_t flags);
void vmm_map(uint64_t virt, uint64_t phys, uint64_t flags);
void vmm_unmap(uint64_t virt);
uint64_t vmm_phys(uint64_t virt);

#endif
