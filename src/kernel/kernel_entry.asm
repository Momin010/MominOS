[BITS 64]

global kernel_start
extern kmain

section .text

kernel_start:
    ; Marker K: did the jump to 0x10000 enter the kernel?
    mov dx, 0x3FD
.waitK:
    in al, dx
    test al, 0x20
    jz .waitK
    mov dx, 0x3F8
    mov al, 'K'
    out dx, al

    ; Mask all PIC IRQs immediately. No IDT yet, any interrupt can triple fault.
    mov al, 0xFF
    out 0xA1, al        ; slave PIC: mask all 8 lines
    out 0x21, al        ; master PIC: mask all 8 lines

    ; Marker 1: survived PIC mask, about to call kmain.
    mov dx, 0x3FD
.wait1:
    in al, dx
    test al, 0x20
    jz .wait1
    mov dx, 0x3F8
    mov al, '1'
    out dx, al

    cld
    call kmain

.halt:
    cli
    hlt
    jmp .halt
