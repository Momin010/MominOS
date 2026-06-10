#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* tail: print the last N lines (default 10) of a file or stdin.
   Usage: tail [-n N] [FILE]

   Bounded memory: we retain at most CAP bytes (the most recent ones). If the
   input is larger, older bytes are dropped, which is fine for printing the
   tail. */

#define CAP 8192

static char hold[CAP];

static int parse_int(const char *s) {
    int v = 0;
    while (*s >= '0' && *s <= '9') {
        v = v * 10 + (*s - '0');
        s++;
    }
    return v;
}

static int tail_fd(long fd, int limit) {
    char buf[512];
    long n;
    long len = 0;          /* bytes currently held */

    while ((n = syscall_read(fd, buf, sizeof(buf))) > 0) {
        for (long i = 0; i < n; i++) {
            if (len < CAP) {
                hold[len++] = buf[i];
            } else {
                /* shift down by one to make room (keep most recent CAP) */
                memcpy(hold, hold + 1, CAP - 1);
                hold[CAP - 1] = buf[i];
            }
        }
    }
    if (n < 0)
        return -1;

    if (len == 0)
        return 0;

    /* find the start of the last `limit` lines. Walk back counting newlines;
       a trailing newline at the very end does not begin a new line. */
    long start = len;
    int seen = 0;
    long i = len - 1;
    if (hold[i] == '\n')
        i--;
    for (; i >= 0; i--) {
        if (hold[i] == '\n') {
            seen++;
            if (seen >= limit) {
                start = i + 1;
                break;
            }
        }
        if (i == 0)
            start = 0;
    }

    syscall_write(1, hold + start, len - start);
    return 0;
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
        return tail_fd(0, limit) < 0 ? 1 : 0;

    fd = syscall_open(file, 0);
    if (fd < 0) {
        printf("tail: %s: no such file\n", file);
        return 1;
    }
    rc = tail_fd(fd, limit);
    syscall_close(fd);
    if (rc < 0) {
        printf("tail: %s: read error\n", file);
        return 1;
    }
    return 0;
}
