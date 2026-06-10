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

/* Exclusive upper bound of the user half (lower canonical half). Every user
   pointer/buffer must lie strictly below this. Matches USER_STACK_TOP in
   elf.c: the user stack and all ELF segments are mapped below it. */
#define VMM_USER_MAX        0x0000800000000000ULL
/* Largest NUL-terminated user string the kernel will scan (path/arg cap). */
#define VMM_USER_STR_MAX    256ULL

void vmm_init(const struct mem_region *regions, uint32_t count);
uint64_t vmm_kernel_pml4(void);
uint64_t vmm_create_address_space(void);
void vmm_destroy_address_space(uint64_t pml4);
void vmm_switch_pml4(uint64_t pml4);
void vmm_map_in(uint64_t pml4, uint64_t virt, uint64_t phys, uint64_t flags);
void vmm_map(uint64_t virt, uint64_t phys, uint64_t flags);
void vmm_unmap(uint64_t virt);
uint64_t vmm_phys(uint64_t virt);

/* User trust-boundary validators. A syscall runs with the calling process's
   CR3 active (the kernel half is shared into every address space, so there is
   no CR3 switch on kernel entry), so these walk the active CR3 to confirm
   each page a user pointer spans is a present USER page. They reject
   kernel-half and non-canonical pointers (bounds), ranges that wrap, and
   in-range-but-unmapped pointers (page walk), so the kernel never faults
   dereferencing a user-supplied pointer. */

/* True if [ptr, ptr+len) lies entirely below VMM_USER_MAX, does not wrap, and
   every page it spans is a present user page. len == 0 is accepted (empty
   write/read are legitimate). */
int vmm_user_range_ok(uint64_t ptr, uint64_t len);
/* True if a NUL-terminated user string at ptr is in the user half on present
   user pages and terminates within max_len bytes. Returns 0 on a bad/unmapped
   pointer or a string that runs past the user half / the cap without a NUL. */
int vmm_user_str_ok(uint64_t ptr, uint64_t max_len);

#endif
