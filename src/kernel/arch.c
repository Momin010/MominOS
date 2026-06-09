#include "arch.h"
#include "serial.h"
#include "syscall.h"

#define IA32_EFER           0xC0000080
#define IA32_STAR           0xC0000081
#define IA32_LSTAR          0xC0000082
#define IA32_FMASK          0xC0000084
#define IA32_GS_BASE        0xC0000101
#define IA32_KERNEL_GS_BASE 0xC0000102

#define EFER_SCE (1ULL << 0)

struct gdt_ptr {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed));

struct tss64 {
    uint32_t reserved0;
    uint64_t rsp0;
    uint64_t rsp1;
    uint64_t rsp2;
    uint64_t reserved1;
    uint64_t ist[7];
    uint64_t reserved2;
    uint16_t reserved3;
    uint16_t iomap_base;
} __attribute__((packed));

struct percpu_data {
    uint64_t kernel_rsp;
    uint64_t user_rsp;
};

extern void syscall_entry(void);

static uint64_t gdt[7];
static struct tss64 tss;
static struct percpu_data percpu;

static inline void wrmsr(uint32_t msr, uint64_t value) {
    uint32_t lo = value & 0xFFFFFFFF;
    uint32_t hi = value >> 32;

    __asm__ volatile ("wrmsr" : : "c"(msr), "a"(lo), "d"(hi));
}

static inline uint64_t rdmsr(uint32_t msr) {
    uint32_t lo;
    uint32_t hi;

    __asm__ volatile ("rdmsr" : "=a"(lo), "=d"(hi) : "c"(msr));
    return ((uint64_t)hi << 32) | lo;
}

static void set_tss_desc(uint16_t selector, uint64_t base, uint32_t limit) {
    uint64_t low = 0;
    uint64_t high = 0;
    uint64_t index = selector / 8;

    low |= limit & 0xFFFFULL;
    low |= (base & 0xFFFFFFULL) << 16;
    low |= 0x89ULL << 40;
    low |= ((uint64_t)(limit >> 16) & 0xFULL) << 48;
    low |= ((base >> 24) & 0xFFULL) << 56;
    high = base >> 32;

    gdt[index] = low;
    gdt[index + 1] = high;
}

static void load_gdt(void) {
    struct gdt_ptr ptr = {
        .limit = sizeof(gdt) - 1,
        .base = (uint64_t)gdt,
    };

    __asm__ volatile (
        "lgdt %0\n"
        "mov %1, %%ax\n"
        "mov %%ax, %%ds\n"
        "mov %%ax, %%es\n"
        "mov %%ax, %%ss\n"
        "pushq %2\n"
        "leaq 1f(%%rip), %%rax\n"
        "pushq %%rax\n"
        "lretq\n"
        "1:\n"
        :
        : "m"(ptr), "i"(KERNEL_DS), "i"(KERNEL_CS)
        : "rax", "memory");
}

void arch_set_kernel_stack(uint64_t rsp) {
    tss.rsp0 = rsp;
    percpu.kernel_rsp = rsp;
}

void arch_init(void) {
    gdt[0] = 0;
    gdt[1] = 0x00AF9A000000FFFFULL;
    gdt[2] = 0x00AF92000000FFFFULL;
    gdt[3] = 0x00AFF2000000FFFFULL;
    gdt[4] = 0x00AFFA000000FFFFULL;
    set_tss_desc(0x28, (uint64_t)&tss, sizeof(tss) - 1);

    tss.iomap_base = sizeof(tss);
    load_gdt();
    __asm__ volatile ("ltr %0" : : "r"((uint16_t)0x28));

    wrmsr(IA32_KERNEL_GS_BASE, (uint64_t)&percpu);
    /* Single-CPU: pin GS base to percpu in both ring 0 and ring 3 and drop
       swapgs entirely. swapgs relies on a strict ring3<->ring0 pairing that
       context-switching mid-syscall (with a second user process) violates,
       desyncing GS_BASE/KERNEL_GS_BASE and corrupting the syscall stack load. */
    wrmsr(IA32_GS_BASE, (uint64_t)&percpu);
    wrmsr(IA32_STAR, ((uint64_t)KERNEL_CS << 32) | ((uint64_t)KERNEL_DS << 48));
    wrmsr(IA32_LSTAR, (uint64_t)syscall_entry);
    wrmsr(IA32_FMASK, 0x200);
    wrmsr(IA32_EFER, rdmsr(IA32_EFER) | EFER_SCE);

    syscall_init();
    serial_print("[ARCH] usermode ready\n");
}
