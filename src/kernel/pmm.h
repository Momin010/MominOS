#ifndef PMM_H
#define PMM_H

#include <stdint.h>

void pmm_init(void);
void pmm_reserve(uint64_t base, uint64_t length);
uint64_t pmm_alloc(void);
void pmm_free(uint64_t addr);
uint64_t pmm_free_pages(void);

#endif
