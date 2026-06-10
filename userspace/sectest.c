#include "stdio.h"
#include "syscall.h"

/* sectest: prove the syscall boundary rejects bad user pointers instead of
   faulting in the kernel. Each call below feeds a kernel-half or wrapping
   pointer/length to a syscall that would otherwise dereference it; every one
   must return -1 and the kernel must keep running (we print after each). */
int main(void) {
    long r;

    /* write(1, <kernel pointer>, 16): kernel would read user buf -> reject. */
    r = syscall_write(1, (void *)0xFFFFFFFF80000000UL, 16);
    printf("sectest: write kptr -> %d (want -1)\n", (int)r);

    /* read(0, <kernel pointer>, 16): kernel would write into user buf -> reject. */
    r = syscall_read(0, (void *)0xFFFFFFFF80000000UL, 16);
    printf("sectest: read kptr  -> %d (want -1)\n", (int)r);

    /* open(<kernel pointer string>): resolve_path would deref it -> reject. */
    r = syscall_open((const char *)0xFFFF808000000000UL, O_RDONLY);
    printf("sectest: open kptr  -> %d (want -1)\n", (int)r);

    /* write with a length that wraps past the address space -> reject. */
    r = syscall_write(1, (void *)0x1000UL, 0xFFFFFFFFFFFFFFFFUL);
    printf("sectest: write wrap -> %d (want -1)\n", (int)r);

    /* in-range but UNMAPPED user pointer: 0x1000 is in the user half but not
       mapped in this process. The page-walk validator must reject it -> -1
       (and the kernel must not fault). */
    r = syscall_write(1, (void *)0x1000UL, 16);
    printf("sectest: write unmap-> %d (want -1)\n", (int)r);
    r = syscall_read(0, (void *)0x1000UL, 16);
    printf("sectest: read unmap -> %d (want -1)\n", (int)r);

    /* path traversal cannot escape root: "/../../etc" normalizes to "/etc",
       which does not exist here, so open fails (no escape, no crash). */
    r = syscall_open("/../../etc/passwd", O_RDONLY);
    printf("sectest: traversal  -> %d (want -1)\n", (int)r);

    printf("sectest: survived, kernel still alive\n");
    return 0;
}
