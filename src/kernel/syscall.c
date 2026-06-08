#include "syscall.h"
#include "sched.h"
#include "serial.h"
#include "tty.h"

#define SYS_WRITE 1
#define SYS_READ  2
#define SYS_EXIT  6

void syscall_init(void) {
    serial_print("[SYSCALL] enabled\n");
}

uint64_t syscall_dispatch(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3) {
    if (n == SYS_WRITE) {
        const char *buf = (const char *)a2;
        (void)a1;
        for (uint64_t i = 0; i < a3; i++)
            serial_putc(buf[i]);
        return a3;
    }

    if (n == SYS_READ) {
        char *buf = (char *)a2;

        if (a1 != 0)
            return (uint64_t)-1;
        return tty_read(buf, a3);
    }

    if (n == SYS_EXIT) {
        serial_print("[USER] exit ");
        serial_print_hex(a1);
        serial_print("\n");
        thread_exit_code((int)a1);
        return 0;
    }

    return (uint64_t)-1;
}
