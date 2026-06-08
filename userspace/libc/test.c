#include "string.h"
#include "stdlib.h"
#include "stdio.h"

int main(void) {
    printf("Testing string.h functions...\n");

    const char *hello = "Hello, World!";
    printf("strlen(\"%s\") = %d\n", hello, (int)strlen(hello));

    char buf[32];
    strcpy(buf, hello);
    printf("strcpy: %s\n", buf);

    printf("strcmp(\"abc\", \"abc\") = %d\n", strcmp("abc", "abc"));
    printf("strcmp(\"abc\", \"abd\") = %d\n", strcmp("abc", "abd"));
    printf("strcmp(\"abd\", \"abc\") = %d\n", strcmp("abd", "abc"));

    memset(buf, 'X', 5);
    buf[5] = '\0';
    printf("memset: %s\n", buf);

    char dest[32];
    memcpy(dest, hello, strlen(hello) + 1);
    printf("memcpy: %s\n", dest);

    printf("\nTesting stdlib.h functions...\n");
    int *arr = (int *)malloc(5 * sizeof(int));
    if (arr) {
        printf("malloc succeeded\n");
        for (int i = 0; i < 5; i++) {
            arr[i] = i * 10;
        }
        for (int i = 0; i < 5; i++) {
            printf("arr[%d] = %d\n", i, arr[i]);
        }
        free(arr);
        printf("free succeeded\n");
    } else {
        printf("malloc failed\n");
    }

    void *ptr = malloc(0);
    printf("malloc(0) = %p\n", ptr);

    char *big = (char *)malloc(70000);
    printf("malloc(70000) = %p (should be NULL)\n", big);

    printf("\nTesting stdio.h functions...\n");
    putchar('A');
    putchar('\n');

    puts("Testing puts");

    printf("printf: %%s=%s, %%d=%d, %%x=%x, %%c=%c, %%%%=%%\n", "test", -42, 255, 'Z');

    return 0;
}