#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* touch: create each FILE if it does not exist. We open with O_CREAT but
   without O_TRUNC, so an existing file keeps its contents. */
int main(int argc, char **argv) {
    int rc = 0;

    if (argc < 2) {
        const char *msg = "usage: touch FILE...\n";
        syscall_write(2, (void *)msg, strlen(msg));
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        long fd = syscall_open(argv[i], O_WRONLY | O_CREAT);
        if (fd < 0) {
            printf("touch: %s: cannot create\n", argv[i]);
            rc = 1;
            continue;
        }
        syscall_close(fd);
    }

    return rc;
}
