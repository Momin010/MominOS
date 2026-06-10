#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* cp: copy SRC to DST. */
int main(int argc, char **argv) {
    char buf[512];
    long src, dst, n;

    if (argc < 3) {
        const char *msg = "usage: cp SRC DST\n";
        syscall_write(2, (void *)msg, strlen(msg));
        return 1;
    }

    src = syscall_open(argv[1], O_RDONLY);
    if (src < 0) {
        printf("cp: %s: no such file\n", argv[1]);
        return 1;
    }

    dst = syscall_open(argv[2], O_WRONLY | O_CREAT | O_TRUNC);
    if (dst < 0) {
        printf("cp: %s: cannot create\n", argv[2]);
        syscall_close(src);
        return 1;
    }

    while ((n = syscall_read(src, buf, sizeof(buf))) > 0) {
        if (syscall_write(dst, buf, n) != n) {
            printf("cp: %s: write error\n", argv[2]);
            syscall_close(src);
            syscall_close(dst);
            return 1;
        }
    }

    syscall_close(src);
    syscall_close(dst);
    if (n < 0) {
        printf("cp: %s: read error\n", argv[1]);
        return 1;
    }
    return 0;
}
