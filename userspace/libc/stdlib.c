#include "stdlib.h"
#include "syscall.h"

static char heap_buffer[65536];
static char *heap_start = NULL;
static char *heap_end = NULL;
static char *heap_current = NULL;

void *malloc(size_t size) {
    if (size == 0) {
        return NULL;
    }

    if (heap_start == NULL) {
        heap_start = heap_buffer;
        heap_end = heap_buffer + sizeof(heap_buffer);
        heap_current = heap_start;
    }

    size = (size + 7) & ~7;

    if (heap_current + size > heap_end) {
        return NULL;
    }

    void *ptr = heap_current;
    heap_current += size;
    return ptr;
}

void free(void *ptr) {
    (void)ptr;
}