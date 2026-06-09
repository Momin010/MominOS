#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* ls: list a directory. The kernel SYS_READDIR packs entries as
   [type u8][name_len u8][name bytes] records into our buffer. */
int main(int argc, char **argv) {
    char buf[4096];
    const char *path = (argc > 1) ? argv[1] : ".";
    long n;
    long off = 0;

    n = syscall_readdir(path, buf, sizeof(buf));
    if (n < 0) {
        printf("ls: %s: cannot read directory\n", path);
        return 1;
    }

    while (off + 2 <= n) {
        unsigned char type = (unsigned char)buf[off];
        unsigned char name_len = (unsigned char)buf[off + 1];
        char *name = buf + off + 2;

        if (off + 2 + name_len > n)
            break;

        syscall_write(1, name, name_len);
        if (type == 2)              /* EXT2_FT_DIR */
            syscall_write(1, "/", 1);
        syscall_write(1, "\n", 1);

        off += 2 + name_len;
    }

    return 0;
}
