#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* head: print the first N lines (default 10) of a file or stdin.
   Usage: head [-n N] [FILE] */

static int parse_int(const char *s) {
    int v = 0;
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (*s - '0');
        s++;
    }
    return v;
}

static int head_fd(long fd, int limit) {
    char buf[512];
    long n = 0;
    int lines = 0;

    while (lines < limit && (n = syscall_read(fd, buf, sizeof(buf))) > 0) {
        long start = 0;
        for (long i = 0; i < n; i++) {
            if (buf[i] == '\n') {
                syscall_write(1, buf + start, i - start + 1);
                start = i + 1;
                if (++lines >= limit)
                    break;
            }
        }
        if (lines < limit && start < n)
            syscall_write(1, buf + start, n - start);
    }
    return (n < 0) ? -1 : 0;
}

int main(int argc, char **argv) {
    int limit = 10;
    int i = 1;
    const char *file = 0;
    long fd;
    int rc;

    while (i < argc) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            limit = parse_int(argv[i + 1]);
            i += 2;
        } else {
            file = argv[i];
            i++;
        }
    }

    if (file == 0)
        return head_fd(0, limit) < 0 ? 1 : 0;

    fd = syscall_open(file, 0);
    if (fd < 0) {
        printf("head: %s: no such file\n", file);
        return 1;
    }
    rc = head_fd(fd, limit);
    syscall_close(fd);
    if (rc < 0) {
        printf("head: %s: read error\n", file);
        return 1;
    }
    return 0;
}
