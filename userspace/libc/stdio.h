#ifndef STDIO_H
#define STDIO_H

#define NULL ((void*)0)
#define EOF (-1)

typedef unsigned long size_t;

int putchar(int c);
int puts(const char *s);
int printf(const char *format, ...);

#endif