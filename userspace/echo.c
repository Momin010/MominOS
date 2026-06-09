#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* echo: print arguments separated by spaces, then a newline. */
int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        syscall_write(1, argv[i], strlen(argv[i]));
        if (i + 1 < argc)
            syscall_write(1, " ", 1);
    }
    syscall_write(1, "\n", 1);
    return 0;
}
