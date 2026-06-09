#ifndef STRING_H
#define STRING_H

#define NULL ((void*)0)

typedef unsigned long size_t;

size_t strlen(const char *str);
void *memcpy(void *dest, const void *src, size_t n);
void *memset(void *s, int c, size_t n);
int strcmp(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
char *strcpy(char *dest, const char *src);

#endif