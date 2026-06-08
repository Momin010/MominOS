#include "sched.h"
#include "kheap.h"
#include "serial.h"

#define THREAD_STACK_SIZE (64ULL * 1024)

extern void switch_context(uint64_t *old_rsp, uint64_t new_rsp);
extern void thread_trampoline(void);

static struct thread *current_thread;
static uint32_t next_thread_id = 1;
static volatile uint32_t need_resched;
static uint32_t scheduler_ready;

static void idle_thread(void *arg) {
    (void)arg;

    while (1)
        __asm__ volatile ("sti; hlt");
}

static inline uint64_t irq_save(void) {
    uint64_t flags;

    __asm__ volatile ("pushfq; pop %0; cli" : "=r"(flags) : : "memory");
    return flags;
}

static inline void irq_restore(uint64_t flags) {
    __asm__ volatile ("push %0; popfq" : : "r"(flags) : "memory");
}

static inline uint64_t align_down(uint64_t value, uint64_t align) {
    return value & ~(align - 1);
}

static void enable_sse(void) {
    uint64_t cr0;
    uint64_t cr4;

    __asm__ volatile ("mov %%cr0, %0" : "=r"(cr0));
    cr0 &= ~(1ULL << 2);
    cr0 |= (1ULL << 1);
    __asm__ volatile ("mov %0, %%cr0" : : "r"(cr0) : "memory");

    __asm__ volatile ("mov %%cr4, %0" : "=r"(cr4));
    cr4 |= (1ULL << 9) | (1ULL << 10);
    __asm__ volatile ("mov %0, %%cr4" : : "r"(cr4) : "memory");

    __asm__ volatile ("fninit");
}

static void fxsave_to(uint8_t *area) {
    __asm__ volatile ("fxsave64 (%0)" : : "r"(area) : "memory");
}

static void fxrstor_from(uint8_t *area) {
    __asm__ volatile ("fxrstor64 (%0)" : : "r"(area) : "memory");
}

static void enqueue_thread(struct thread *thread) {
    struct thread *tail;

    if (current_thread == 0) {
        current_thread = thread;
        thread->next = thread;
        return;
    }

    tail = current_thread;
    while (tail->next != current_thread)
        tail = tail->next;

    tail->next = thread;
    thread->next = current_thread;
}

static struct thread *pick_next(void) {
    struct thread *thread = current_thread->next;

    while (thread != current_thread) {
        if (thread->state == THREAD_READY)
            return thread;
        thread = thread->next;
    }

    if (current_thread->state == THREAD_READY || current_thread->state == THREAD_RUNNING)
        return current_thread;

    return 0;
}

static void schedule_locked(void) {
    struct thread *old = current_thread;
    struct thread *next;

    if (!scheduler_ready || current_thread == 0)
        return;

    next = pick_next();
    if (next == 0 || next == old)
        return;

    if (old->state == THREAD_RUNNING)
        old->state = THREAD_READY;

    next->state = THREAD_RUNNING;
    current_thread = next;
    need_resched = 0;

    fxsave_to(old->fxsave);
    fxrstor_from(next->fxsave);
    switch_context(&old->rsp, next->rsp);
}

void sched_init(void) {
    struct thread *boot;

    enable_sse();

    boot = kzalloc(sizeof(*boot));
    if (boot == 0) {
        serial_print("[SCHED] boot thread alloc failed\n");
        while (1)
            __asm__ volatile ("cli; hlt");
    }

    boot->id = next_thread_id++;
    boot->state = THREAD_RUNNING;
    boot->stack_base = 0;
    fxsave_to(boot->fxsave);
    enqueue_thread(boot);

    scheduler_ready = 1;
    thread_create(idle_thread, 0);
    serial_print("[SCHED] initialized\n");
}

struct thread *thread_create(thread_entry_t entry, void *arg) {
    uint64_t flags = irq_save();
    struct thread *thread;
    uint64_t *stack;
    void *stack_base;
    uint64_t top;
    uint64_t rsp;

    thread = kzalloc(sizeof(*thread));
    if (thread == 0) {
        irq_restore(flags);
        return 0;
    }

    stack_base = kmalloc(THREAD_STACK_SIZE);
    if (stack_base == 0) {
        kfree(thread);
        irq_restore(flags);
        return 0;
    }

    top = align_down((uint64_t)stack_base + THREAD_STACK_SIZE, 16);
    rsp = top - 64;
    stack = (uint64_t *)rsp;

    stack[0] = 0;
    stack[1] = 0;
    stack[2] = 0;
    stack[3] = (uint64_t)arg;
    stack[4] = 0;
    stack[5] = (uint64_t)entry;
    stack[6] = (uint64_t)thread_trampoline;
    stack[7] = 0;

    thread->rsp = rsp;
    thread->stack_base = stack_base;
    thread->id = next_thread_id++;
    thread->state = THREAD_READY;

    fxsave_to(thread->fxsave);
    enqueue_thread(thread);

    irq_restore(flags);
    return thread;
}

void sched_yield(void) {
    uint64_t flags = irq_save();

    need_resched = 1;
    schedule_locked();
    irq_restore(flags);
}

void sched_tick(void) {
    if (scheduler_ready)
        need_resched = 1;
}

void sched_after_irq(void) {
    if (need_resched)
        schedule_locked();
}

void thread_exit(void) {
    uint64_t flags = irq_save();

    current_thread->state = THREAD_DEAD;
    need_resched = 1;

    while (1) {
        schedule_locked();
        irq_restore(flags);
        __asm__ volatile ("cli; hlt");
        flags = irq_save();
    }
}

uint32_t sched_current_thread_id(void) {
    if (current_thread == 0)
        return 0;
    return current_thread->id;
}
