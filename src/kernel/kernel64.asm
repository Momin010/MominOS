[BITS 64]
[ORG 0x10000]
kernel_start:
    call vga_clear_screen
    mov rsi, WELCOME_MSG
    call vga_print_string
    call setup_idt
    mov rsi, IDT_MSG
    call vga_print_string
    jmp shell_start
WELCOME_MSG db 'MominOS 64-bit Pro Active', 13, 10, 0
IDT_MSG db 'IDT and Exceptions Initialized...', 13, 10, 0
%include "../drivers/vga64.asm"
%include "idt64.asm"
%include "shell64.asm"
%include "../drivers/keyboard64.asm"
