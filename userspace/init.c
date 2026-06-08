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
    const char prompt[] = "init: type a line> ";
    char line[128];
    long n;

    syscall3(1, 1, (long)prompt, sizeof(prompt) - 1);

    /* blocking read of one line from stdin */
    n = syscall3(2, 0, (long)line, sizeof(line));
    if (n > 0) {
        const char pre[] = "init: you typed: ";
        syscall3(1, 1, (long)pre, sizeof(pre) - 1);
        syscall3(1, 1, (long)line, n);
    }

    syscall1(6, 0);

    while (1) {
    }
}
