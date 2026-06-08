#ifndef KERNEL_SYSCALL_H
#define KERNEL_SYSCALL_H

#include <stdint.h>

void syscall_init(void);
uint64_t syscall_dispatch(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3);

#endif
