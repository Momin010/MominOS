[BITS 64]

global user_enter
global syscall_entry

extern syscall_dispatch

%define USER_DS_SEL 0x1B
%define USER_CS_SEL 0x23

section .text

user_enter:
    cli
    mov ax, USER_DS_SEL
    mov ds, ax
    mov es, ax
    push qword USER_DS_SEL
    push rsi
    push qword 0x202
    push qword USER_CS_SEL
    push rdi
    iretq

syscall_entry:
    ; GS base is pinned to percpu in both rings (no swapgs needed, single CPU)
    mov gs:[8], rsp
    mov rsp, gs:[0]

    push qword gs:[8]
    push rcx
    push r11
    sub rsp, 8

    mov r8, rdi
    mov r9, rdx
    mov rdi, rax
    mov rdx, rsi
    mov rsi, r8
    mov rcx, r9

    call syscall_dispatch
    add rsp, 8
    pop r11
    pop rcx
    pop rdi

    mov rsp, rdi
    o64 sysret
