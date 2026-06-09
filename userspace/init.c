#include "stdio.h"
#include "syscall.h"

/* Step 2 driver: read a line (blocking stdin), then spawn a child with
   argv and wait for it, exercising spawn/waitpid/argv end to end. */
int main(int argc, char **argv) {
    char line[128];
    long n;
    long pid;
    long code;
    char *child_argv[4];

    (void)argc;
    (void)argv;

    printf("init: type a line> ");
    n = syscall_read(0, line, sizeof(line) - 1);
    if (n > 0) {
        line[n] = 0;
        printf("init: you typed: %s", line);
    }

    child_argv[0] = "/bin/argtest";
    child_argv[1] = "hello";
    child_argv[2] = "world";
    child_argv[3] = 0;

    printf("init: spawning argtest\n");
    pid = syscall_spawn(child_argv[0], child_argv);
    printf("init: child pid=%d\n", (int)pid);

    code = syscall_waitpid(pid);
    printf("init: child exited code=%d\n", (int)code);

    return 0;
}
