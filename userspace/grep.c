#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* grep: print lines that contain PATTERN (plain substring, no regex).
   Usage: grep PATTERN [FILE]   (no FILE -> read stdin) */

#define LINE_MAX 1024

/* return 1 if needle occurs as a substring of haystack[0..len) */
static int contains(const char *hay, long len, const char *needle) {
    long nlen = (long)strlen(needle);

    if (nlen == 0)
        return 1;
    if (nlen > len)
        return 0;

    for (long i = 0; i + nlen <= len; i++) {
        long j = 0;
        while (j < nlen && hay[i + j] == needle[j])
            j++;
        if (j == nlen)
            return 1;
    }
    return 0;
}

static int grep_fd(long fd, const char *pat) {
    char buf[512];
    char line[LINE_MAX];
    long llen = 0;
    long n;

    while ((n = syscall_read(fd, buf, sizeof(buf))) > 0) {
        for (long i = 0; i < n; i++) {
            char c = buf[i];
            if (c == '\n') {
                if (contains(line, llen, pat)) {
                    syscall_write(1, line, llen);
                    syscall_write(1, "\n", 1);
                }
                llen = 0;
            } else if (llen < LINE_MAX) {
                line[llen++] = c;
            }
            /* lines longer than LINE_MAX have the overflow dropped */
        }
    }
    if (n < 0)
        return -1;

    /* trailing line with no final newline */
    if (llen > 0 && contains(line, llen, pat)) {
        syscall_write(1, line, llen);
        syscall_write(1, "\n", 1);
    }
    return 0;
}

int main(int argc, char **argv) {
    long fd;
    int rc;

    if (argc < 2) {
        const char *msg = "usage: grep PATTERN [FILE]\n";
        syscall_write(2, (void *)msg, strlen(msg));
        return 1;
    }

    if (argc < 3)
        return grep_fd(0, argv[1]) < 0 ? 1 : 0;

    fd = syscall_open(argv[2], 0);
    if (fd < 0) {
        printf("grep: %s: no such file\n", argv[2]);
        return 1;
    }
    rc = grep_fd(fd, argv[1]);
    syscall_close(fd);
    if (rc < 0) {
        printf("grep: %s: read error\n", argv[2]);
        return 1;
    }
    return 0;
}
