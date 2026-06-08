#ifndef ARCH_H
#define ARCH_H

#include <stdint.h>

#define KERNEL_CS 0x08
#define KERNEL_DS 0x10
#define USER_DS   0x1B
#define USER_CS   0x23

void arch_init(void);
void arch_set_kernel_stack(uint64_t rsp);
void user_enter(uint64_t rip, uint64_t rsp);

#endif
