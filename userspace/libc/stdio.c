#include "stdio.h"
#include "string.h"
#include "syscall.h"

typedef unsigned long uintptr_t;
typedef __builtin_va_list va_list;
#define va_start(v, l) __builtin_va_start(v, l)
#define va_end(v) __builtin_va_end(v)
#define va_arg(v, t) __builtin_va_arg(v, t)

int putchar(int c) {
    char ch = (char)c;
    syscall_write(1, &ch, 1);
    return c;
}

int puts(const char *s) {
    size_t len = strlen(s);
    syscall_write(1, s, len);
    putchar('\n');
    return 0;
}

static void print_int(int n, int base) {
    char buf[32];
    int i = 0;
    int neg = 0;

    if (n == 0) {
        putchar('0');
        return;
    }

    if (n < 0 && base == 10) {
        neg = 1;
        n = -n;
    }

    while (n > 0) {
        int digit = n % base;
        buf[i++] = (digit < 10) ? '0' + digit : 'a' + digit - 10;
        n /= base;
    }

    if (neg) {
        putchar('-');
    }

    while (i--) {
        putchar(buf[i]);
    }
}

static void print_uint(unsigned int n, int base) {
    char buf[32];
    int i = 0;

    if (n == 0) {
        putchar('0');
        return;
    }

    while (n > 0) {
        int digit = n % base;
        buf[i++] = (digit < 10) ? '0' + digit : 'a' + digit - 10;
        n /= base;
    }

    while (i--) {
        putchar(buf[i]);
    }
}

int printf(const char *format, ...) {
    va_list args;
    va_start(args, format);

    while (*format) {
        if (*format == '%') {
            format++;
            switch (*format) {
                case 's': {
                    const char *s = va_arg(args, const char *);
                    if (s) {
                        syscall_write(1, s, strlen(s));
                    } else {
                        puts("(null)");
                    }
                    break;
                }
                case 'd':
                case 'i':
                    print_int(va_arg(args, int), 10);
                    break;
                case 'x':
                    print_uint(va_arg(args, unsigned int), 16);
                    break;
                case 'c':
                    putchar(va_arg(args, int));
                    break;
                case 'p': {
                    uintptr_t p = va_arg(args, uintptr_t);
                    if (p == 0) {
                        puts("(nil)");
                    } else {
                        putchar('0');
                        putchar('x');
                        print_uint(p, 16);
                    }
                    break;
                }
                case '%':
                    putchar('%');
                    break;
                default:
                    putchar('%');
                    putchar(*format);
                    break;
            }
        } else {
            putchar(*format);
        }
        format++;
    }

    va_end(args);
    return 0;
}