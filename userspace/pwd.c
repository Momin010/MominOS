#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* pwd: print the current working directory. */
int main(int argc, char **argv) {
    char cwd[256];
    long n;

    (void)argc;
    (void)argv;

    n = syscall_getcwd(cwd, sizeof(cwd));
    if (n < 0) {
        const char *msg = "pwd: cannot get current directory\n";
        syscall_write(2, (void *)msg, strlen(msg));
        return 1;
    }

    syscall_write(1, cwd, strlen(cwd));
    syscall_write(1, "\n", 1);
    return 0;
}
