#pragma once
#include <stdint.h>

typedef enum {
    DIAG_EVT_FAULT   = 0,
    DIAG_EVT_SYSCALL = 1,
    DIAG_EVT_LOG     = 2,
} diag_event_type_t;

typedef struct {
    uint64_t vector;
    uint64_t error_code;
    uint64_t rip;
    uint64_t rsp;
    uint64_t rax;
    uint64_t rdi;
    uint64_t rsi;
    uint64_t cr2;
    uint32_t tid;
    char     cwd[64];
} diag_fault_t;

typedef struct {
    uint64_t number;
    uint64_t args[3];
    uint64_t retval;
    uint32_t tid;
} diag_syscall_t;

typedef struct {
    char text[96];
} diag_log_t;

typedef struct {
    diag_event_type_t type;
    uint64_t          tick;
    union {
        diag_fault_t   fault;
        diag_syscall_t syscall;
        diag_log_t     log;
    };
} diag_event_t;

#define DIAG_RING_SIZE 64

struct isr_frame;

void     diag_init(void);
void     diag_capture_fault(struct isr_frame *frame, uint32_t tid, const char *cwd, uint64_t cr2);
void     diag_capture_syscall(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t retval, uint32_t tid);
void     diag_log_write(const char *text);
uint32_t diag_ring_snapshot(diag_event_t *out, uint32_t max_out);
