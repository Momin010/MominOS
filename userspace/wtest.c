#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* wtest: end-to-end proof of the ext2 write path.
   1. create /wtest.out, write a known string, close (flushes to disk)
   2. reopen read-only, read it back, print it
   3. verify the bytes match what we wrote */
int main(void) {
    const char *path = "/wtest.out";
    const char *msg = "MominOS write path works!\n";
    char buf[128];
    long fd, n;

    /* create + write */
    fd = syscall_open(path, O_CREAT);
    if (fd < 0) {
        printf("wtest: create failed\n");
        return 1;
    }
    n = syscall_write(fd, (void *)msg, strlen(msg));
    printf("wtest: wrote %d bytes\n", (int)n);
    syscall_close(fd);

    /* reopen and read back */
    fd = syscall_open(path, O_RDONLY);
    if (fd < 0) {
        printf("wtest: reopen failed\n");
        return 1;
    }
    n = syscall_read(fd, buf, sizeof(buf) - 1);
    syscall_close(fd);
    if (n < 0)
        n = 0;
    buf[n] = 0;

    printf("wtest: read back %d bytes: %s", (int)n, buf);

    if (strcmp(buf, msg) == 0)
        printf("wtest: MATCH ok\n");
    else
        printf("wtest: MISMATCH\n");

    return 0;
}
