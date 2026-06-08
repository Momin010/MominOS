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

void _start(void) {
    const char msg[] = "hello from userspace\n";

    syscall3(1, 1, (long)msg, sizeof(msg) - 1);
    syscall1(6, 0);

    while (1) {
    }
}
