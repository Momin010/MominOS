#include "syscall.h"
#include "sched.h"
#include "serial.h"

#define SYS_WRITE 1
#define SYS_EXIT  6

void syscall_init(void) {
    serial_print("[SYSCALL] enabled\n");
}

uint64_t syscall_dispatch(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3) {
    const char *buf;

    if (n == SYS_WRITE) {
        buf = (const char *)a2;
        (void)a1;
        for (uint64_t i = 0; i < a3; i++)
            serial_putc(buf[i]);
        return a3;
    }

    if (n == SYS_EXIT) {
        serial_print("[USER] exit ");
        serial_print_hex(a1);
        serial_print("\n");
        thread_exit();
        return 0;
    }

    return (uint64_t)-1;
}
