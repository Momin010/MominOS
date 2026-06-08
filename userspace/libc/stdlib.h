#ifndef STDLIB_H
#define STDLIB_H

#define NULL ((void*)0)

typedef __SIZE_TYPE__ size_t;

void *malloc(size_t size);
void free(void *ptr);

#endif