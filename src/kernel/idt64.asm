[BITS 64]
align 16
idt_start:
%assign i 0
%rep 32
    dw (isr_%+i - $$) & 0xFFFF
    dw 0x08
    db 0
    db 0x8E
    dw ((isr_%+i - $$) >> 16) & 0xFFFF
    dd ((isr_%+i - $$) >> 32) & 0xFFFFFFFF
    dd 0
%assign i i+1
%endrep
idt_end:
idt_descriptor:
    dw idt_end - idt_start - 1
    dq idt_start
setup_idt:
    lidt [idt_descriptor]
    ret
%assign i 0
%rep 32
isr_%+i:
    cli
    push qword i
    jmp exception_handler
%assign i i+1
%endrep
exception_handler:
    mov rsi, MSG_EXCEPTION
    call vga_print_string
    mov rax, [rsp]
    call print_hex
    jmp $
print_hex:
    push rax
    push rbx
    push rcx
    push rdx
    mov rdx, rax
    mov rcx, 16
.loop:
    rol rdx, 4
    mov rax, rdx
    and rax, 0x0F
    cmp al, 9
    jbe .digit
    add al, 7
.digit:
    add al, '0'
    call vga_put_char
    loop .loop
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
MSG_EXCEPTION db 'EXCEPTION: 0x', 0
