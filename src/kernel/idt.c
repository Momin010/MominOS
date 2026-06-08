#include "idt.h"
#include "serial.h"

#define IDT_ENTRIES 256
#define KERNEL_CODE_SELECTOR 0x08
#define IDT_INTERRUPT_GATE 0x8E

struct idt_entry {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t ist;
    uint8_t type_attr;
    uint16_t offset_mid;
    uint32_t offset_high;
    uint32_t reserved;
} __attribute__((packed));

struct idt_ptr {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed));

extern void *isr_stub_table[IDT_ENTRIES];

static struct idt_entry idt[IDT_ENTRIES];

static const char *exception_names[32] = {
    "Divide Error",
    "Debug",
    "NMI",
    "Breakpoint",
    "Overflow",
    "Bound Range",
    "Invalid Opcode",
    "Device Not Available",
    "Double Fault",
    "Coprocessor Segment",
    "Invalid TSS",
    "Segment Not Present",
    "Stack Segment",
    "General Protection",
    "Page Fault",
    "Reserved",
    "x87 Floating Point",
    "Alignment Check",
    "Machine Check",
    "SIMD Floating Point",
    "Virtualization",
    "Control Protection",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Reserved",
    "Hypervisor Injection",
    "VMM Communication",
    "Security",
    "Reserved",
};

static void idt_set_gate(uint8_t vector, uint64_t handler) {
    idt[vector].offset_low = handler & 0xFFFF;
    idt[vector].selector = KERNEL_CODE_SELECTOR;
    idt[vector].ist = 0;
    idt[vector].type_attr = IDT_INTERRUPT_GATE;
    idt[vector].offset_mid = (handler >> 16) & 0xFFFF;
    idt[vector].offset_high = (handler >> 32) & 0xFFFFFFFF;
    idt[vector].reserved = 0;
}

void idt_init(void) {
    for (uint16_t i = 0; i < IDT_ENTRIES; i++)
        idt_set_gate((uint8_t)i, (uint64_t)isr_stub_table[i]);

    struct idt_ptr ptr = {
        .limit = sizeof(idt) - 1,
        .base = (uint64_t)idt,
    };

    __asm__ volatile ("lidt %0" : : "m"(ptr));
    serial_print("[IDT] initialized\n");
}

void isr_handler(struct isr_frame *frame) {
    serial_print("\n[ISR] vector ");
    serial_print_hex(frame->vector);

    if (frame->vector < 32) {
        serial_print(" ");
        serial_print(exception_names[frame->vector]);
    }

    serial_print("\n[ISR] rip=");
    serial_print_hex(frame->rip);
    serial_print(" err=");
    serial_print_hex(frame->error_code);

    if (frame->vector == 14) {
        uint64_t cr2;
        __asm__ volatile ("mov %%cr2, %0" : "=r"(cr2));
        serial_print(" cr2=");
        serial_print_hex(cr2);
    }

    serial_print("\n[ISR] halted\n");

    while (1)
        __asm__ volatile ("cli; hlt");
}
