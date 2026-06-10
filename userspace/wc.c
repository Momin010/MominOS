#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* wc: count lines, words and bytes of a file or stdin.
   Usage: wc [FILE]   (no FILE -> read stdin) */

static int count_fd(long fd, long *lines, long *words, long *bytes) {
    char buf[512];
    long n;
    int in_word = 0;

    *lines = 0;
    *words = 0;
    *bytes = 0;

    while ((n = syscall_read(fd, buf, sizeof(buf))) > 0) {
        for (long i = 0; i < n; i++) {
            char c = buf[i];
            (*bytes)++;
            if (c == '\n')
                (*lines)++;
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                in_word = 0;
            } else if (!in_word) {
                in_word = 1;
                (*words)++;
            }
        }
    }
    return (n < 0) ? -1 : 0;
}

int main(int argc, char **argv) {
    long lines, words, bytes;
    long fd;
    int rc;

    if (argc < 2) {
        if (count_fd(0, &lines, &words, &bytes) < 0) {
            const char *msg = "wc: read error\n";
            syscall_write(2, (void *)msg, strlen(msg));
            return 1;
        }
        printf("%d %d %d\n", (int)lines, (int)words, (int)bytes);
        return 0;
    }

    fd = syscall_open(argv[1], 0);
    if (fd < 0) {
        printf("wc: %s: no such file\n", argv[1]);
        return 1;
    }

    rc = count_fd(fd, &lines, &words, &bytes);
    syscall_close(fd);
    if (rc < 0) {
        printf("wc: %s: read error\n", argv[1]);
        return 1;
    }

    printf("%d %d %d %s\n", (int)lines, (int)words, (int)bytes, argv[1]);
    return 0;
}
