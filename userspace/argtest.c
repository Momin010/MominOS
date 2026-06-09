#include "stdio.h"
#include "syscall.h"

/* Prints argc and each argv to verify the kernel's user-stack argv
   layout and 16-byte alignment (printf uses SSE under USER_CFLAGS). */
int main(int argc, char **argv) {
    printf("argtest: argc=%d\n", argc);
    for (int i = 0; i < argc; i++)
        printf("argtest: argv[%d]=%s\n", i, argv[i]);
    return argc;
}
