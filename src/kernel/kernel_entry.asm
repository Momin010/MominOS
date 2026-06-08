[BITS 64]

global kernel_start
extern kmain

section .text

kernel_start:
    cld
    call kmain
.halt:
    cli
    hlt
    jmp .halt
