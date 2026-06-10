#include "stdio.h"
#include "string.h"
#include "syscall.h"

/* uname: print a static system identification string. Both the bare form and
   -a print "MominOS x86_64". */
int main(int argc, char **argv) {
    const char *id = "MominOS x86_64";

    (void)argc;
    (void)argv;

    syscall_write(1, (void *)id, strlen(id));
    syscall_write(1, "\n", 1);
    return 0;
}
