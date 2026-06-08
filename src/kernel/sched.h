#ifndef SCHED_H
#define SCHED_H

#include <stdint.h>

typedef void (*thread_entry_t)(void *arg);

typedef enum thread_state {
    THREAD_READY,
    THREAD_RUNNING,
    THREAD_BLOCKED,
    THREAD_ZOMBIE,
    THREAD_DEAD,
} thread_state_t;

#define MAX_FDS 16

struct vfs_file;

struct thread {
    uint64_t rsp;
    void *stack_base;
    uint64_t kernel_stack_top;
    uint64_t pml4;
    uint32_t id;
    thread_state_t state;
    struct thread *next;
    /* process bits */
    struct vfs_file *fds[MAX_FDS];
    int exit_code;
    int has_exited;
    struct thread *waiter;
    char cwd[128];
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

/* current running thread (for syscalls: fd table, pid, cwd) */
struct thread *sched_current_thread(void);

/* block the current thread; an event source must call sched_wake() on it.
   call with interrupts enabled; it disables them around the block transition. */
void sched_block(void);
/* mark a thread READY if it was BLOCKED. safe from IRQ context. */
void sched_wake(struct thread *thread);

/* exit current thread with a code (for processes). becomes a ZOMBIE until
   reaped by sched_reap(). */
void thread_exit_code(int code);
/* unlink a ZOMBIE thread from the ring and free its kernel resources. */
void sched_reap(struct thread *thread);

#endif
