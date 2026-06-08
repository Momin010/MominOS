#ifndef SCHED_H
#define SCHED_H

#include <stdint.h>

typedef void (*thread_entry_t)(void *arg);

typedef enum thread_state {
    THREAD_READY,
    THREAD_RUNNING,
    THREAD_BLOCKED,
    THREAD_DEAD,
} thread_state_t;

struct thread {
    uint64_t rsp;
    void *stack_base;
    uint64_t kernel_stack_top;
    uint64_t pml4;
    uint32_t id;
    thread_state_t state;
    struct thread *next;
    uint8_t fxsave[512] __attribute__((aligned(16)));
};

void sched_init(void);
struct thread *thread_create(thread_entry_t entry, void *arg);
struct thread *thread_create_process(thread_entry_t entry, void *arg, uint64_t pml4);
void sched_yield(void);
void sched_tick(void);
void sched_after_irq(void);
void thread_exit(void);
uint32_t sched_current_thread_id(void);
uint64_t sched_current_kernel_stack(void);

#endif
