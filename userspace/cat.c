#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* cat: print the contents of each file argument to stdout. */
int main(int argc, char **argv) {
    char buf[512];

    if (argc < 2) {
        const char *msg = "usage: cat FILE...\n";
        syscall_write(2, (void *)msg, strlen(msg));
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        long fd = syscall_open(argv[i], 0);
        long n;

        if (fd < 0) {
            printf("cat: %s: no such file\n", argv[i]);
            continue;
        }

        while ((n = syscall_read(fd, buf, sizeof(buf))) > 0)
            syscall_write(1, buf, n);

        syscall_close(fd);
    }

    return 0;
}
