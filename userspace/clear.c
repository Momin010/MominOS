#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* clear: emit the ANSI clear-screen and cursor-home sequence. */
int main(int argc, char **argv) {
    const char *seq = "\033[2J\033[H";

    (void)argc;
    (void)argv;

    syscall_write(1, (void *)seq, strlen(seq));
    return 0;
}
