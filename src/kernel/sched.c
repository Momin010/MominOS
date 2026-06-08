#include "sched.h"
#include "arch.h"
#include "kheap.h"
#include "serial.h"
#include "vmm.h"

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

static void map_kernel_range(uint64_t pml4, uint64_t virt, uint64_t size) {
    uint64_t start = virt & ~0xFFFULL;
    uint64_t end = (virt + size + 0xFFFULL) & ~0xFFFULL;

    if (pml4 == vmm_kernel_pml4())
        return;

    for (uint64_t addr = start; addr < end; addr += 4096) {
        uint64_t phys = vmm_phys(addr);

        if (phys != 0)
            vmm_map_in(pml4, addr, phys, VMM_WRITABLE);
    }
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
    arch_set_kernel_stack(next->kernel_stack_top);
    if (old->pml4 != next->pml4)
        vmm_switch_pml4(next->pml4);
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

    boot->stack_base = kmalloc(THREAD_STACK_SIZE);
    if (boot->stack_base == 0) {
        serial_print("[SCHED] boot stack alloc failed\n");
        while (1)
            __asm__ volatile ("cli; hlt");
    }

    boot->id = next_thread_id++;
    boot->state = THREAD_RUNNING;
    boot->cwd[0] = '/';
    boot->cwd[1] = 0;
    boot->kernel_stack_top = (uint64_t)boot->stack_base + THREAD_STACK_SIZE;
    boot->pml4 = vmm_kernel_pml4();
    fxsave_to(boot->fxsave);
    enqueue_thread(boot);
    arch_set_kernel_stack(boot->kernel_stack_top);

    scheduler_ready = 1;
    thread_create(idle_thread, 0);
    serial_print("[SCHED] initialized\n");
}

static struct thread *thread_create_with_pml4(thread_entry_t entry, void *arg, uint64_t pml4) {
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
    thread->kernel_stack_top = (uint64_t)stack_base + THREAD_STACK_SIZE;
    thread->pml4 = pml4;
    thread->id = next_thread_id++;
    thread->state = THREAD_READY;
    thread->cwd[0] = '/';
    thread->cwd[1] = 0;

    fxsave_to(thread->fxsave);
    map_kernel_range(pml4, (uint64_t)thread, sizeof(*thread));
    map_kernel_range(pml4, (uint64_t)stack_base, THREAD_STACK_SIZE);
    enqueue_thread(thread);

    irq_restore(flags);
    return thread;
}

struct thread *thread_create(thread_entry_t entry, void *arg) {
    return thread_create_with_pml4(entry, arg, vmm_kernel_pml4());
}

struct thread *thread_create_process(thread_entry_t entry, void *arg, uint64_t pml4) {
    return thread_create_with_pml4(entry, arg, pml4);
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

uint64_t sched_current_kernel_stack(void) {
    if (current_thread == 0)
        return 0;
    return current_thread->kernel_stack_top;
}

struct thread *sched_current_thread(void) {
    return current_thread;
}

void sched_block(void) {
    uint64_t flags = irq_save();

    current_thread->state = THREAD_BLOCKED;
    need_resched = 1;
    schedule_locked();
    irq_restore(flags);
}

void sched_wake(struct thread *thread) {
    uint64_t flags = irq_save();

    if (thread != 0 && thread->state == THREAD_BLOCKED) {
        thread->state = THREAD_READY;
        need_resched = 1;
    }
    irq_restore(flags);
}

void thread_exit_code(int code) {
    uint64_t flags = irq_save();

    current_thread->exit_code = code;
    current_thread->has_exited = 1;
    current_thread->state = THREAD_ZOMBIE;
    if (current_thread->waiter != 0 && current_thread->waiter->state == THREAD_BLOCKED) {
        current_thread->waiter->state = THREAD_READY;
    }
    need_resched = 1;

    while (1) {
        schedule_locked();
        irq_restore(flags);
        __asm__ volatile ("cli; hlt");
        flags = irq_save();
    }
}

void sched_reap(struct thread *thread) {
    uint64_t flags = irq_save();
    struct thread *pred;

    if (thread == 0 || thread->state != THREAD_ZOMBIE) {
        irq_restore(flags);
        return;
    }

    /* unlink from the circular ring: find predecessor whose next is thread */
    pred = thread;
    while (pred->next != thread)
        pred = pred->next;
    pred->next = thread->next;

    thread->state = THREAD_DEAD;
    irq_restore(flags);

    /* now safe to free; we are running on a different thread's stack */
    kfree(thread->stack_base);
    kfree(thread);
}
