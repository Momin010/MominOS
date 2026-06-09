#ifndef SYSCALL_H
#define SYSCALL_H

typedef unsigned long uintptr_t;

#define SYS_WRITE   1
#define SYS_READ    2
#define SYS_OPEN    3
#define SYS_CLOSE   4
#define SYS_EXIT    6
#define SYS_SPAWN   7
#define SYS_WAITPID 8
#define SYS_READDIR 9
#define SYS_CHDIR   10
#define SYS_GETCWD  11

#define O_RDONLY 0
#define O_CREAT  0x40

static inline long syscall1(long n, long a1) {
    long ret;
    __asm__ volatile ("syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1)
        : "rcx", "r11", "memory");
    return ret;
}

static inline long syscall2(long n, long a1, long a2) {
    long ret;
    __asm__ volatile ("syscall"
        : "=a"(ret)
        : "a"(n), "D"(a1), "S"(a2)
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
#define syscall_exit(code) syscall1(SYS_EXIT, code)
#define syscall_spawn(path, argv) syscall2(SYS_SPAWN, (long)path, (long)argv)
#define syscall_waitpid(pid) syscall1(SYS_WAITPID, pid)
#define syscall_readdir(path, buf, size) syscall3(SYS_READDIR, (long)path, (long)buf, size)
#define syscall_chdir(path) syscall1(SYS_CHDIR, (long)path)
#define syscall_getcwd(buf, size) syscall2(SYS_GETCWD, (long)buf, size)

#endif
