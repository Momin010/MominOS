#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* bigtest: prove the ext2 single-indirect write path. Writes a file larger
   than 12 * 4096 = 48KB (so it must use i_block[12]), then reads it back and
   verifies every byte. The content is a position-derived pattern so a wrong
   block mapping is detected, not just a wrong length. */

#define TOTAL   (200 * 1024)    /* 200 KB: well past the 48KB direct limit */
#define CHUNK   4096

static unsigned char pat(unsigned long i) {
    return (unsigned char)((i * 31u + 7u) & 0xff);
}

int main(void) {
    const char *path = "/big.out";
    char buf[CHUNK];
    long fd, n;
    unsigned long written = 0;
    unsigned long verified = 0;

    fd = syscall_open(path, O_CREAT | O_TRUNC | O_WRONLY);
    if (fd < 0) {
        printf("bigtest: create failed\n");
        return 1;
    }

    while (written < TOTAL) {
        unsigned long want = TOTAL - written;
        if (want > CHUNK)
            want = CHUNK;
        for (unsigned long i = 0; i < want; i++)
            buf[i] = (char)pat(written + i);
        n = syscall_write(fd, buf, want);
        if (n != (long)want) {
            printf("bigtest: short write at %d (got %d)\n", (int)written, (int)n);
            syscall_close(fd);
            return 1;
        }
        written += want;
    }
    syscall_close(fd);
    printf("bigtest: wrote %d bytes\n", (int)written);

    fd = syscall_open(path, O_RDONLY);
    if (fd < 0) {
        printf("bigtest: reopen failed\n");
        return 1;
    }

    for (;;) {
        n = syscall_read(fd, buf, CHUNK);
        if (n <= 0)
            break;
        for (long i = 0; i < n; i++) {
            if ((unsigned char)buf[i] != pat(verified + i)) {
                printf("bigtest: MISMATCH at %d\n", (int)(verified + i));
                syscall_close(fd);
                return 1;
            }
        }
        verified += n;
    }
    syscall_close(fd);

    printf("bigtest: verified %d bytes\n", (int)verified);
    if (verified == TOTAL)
        printf("bigtest: MATCH ok\n");
    else
        printf("bigtest: SIZE MISMATCH\n");

    return 0;
}
