#ifndef SYSCALL_H
#define SYSCALL_H

typedef unsigned long uintptr_t;

#define SYS_WRITE 1
#define SYS_READ 2
#define SYS_OPEN 3
#define SYS_CLOSE 4
#define SYS_BRK 5
#define SYS_EXIT 6

static inline long syscall1(long n, long a1) {
    long ret;
    __asm__ volatile ("syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1)
        : "rcx", "r11", "memory");
    return ret;
}

static inline long syscall3(long n, long a1, long a2, long a3) {
    long ret;
    __asm__ volatile ("syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1), "S"(a2), "d"(a3)
        : "rcx", "r11", "memory");
    return ret;
}

#define syscall_write(fd, buf, count) syscall3(SYS_WRITE, fd, (long)buf, count)
#define syscall_read(fd, buf, count) syscall3(SYS_READ, fd, (long)buf, count)
#define syscall_open(path, flags) syscall3(SYS_OPEN, (long)path, flags, 0)
#define syscall_close(fd) syscall1(SYS_CLOSE, fd)
#define syscall_brk(addr) syscall1(SYS_BRK, addr)
#define syscall_exit(code) syscall1(SYS_EXIT, code)

#endif