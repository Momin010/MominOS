[BITS 64]

global switch_context
global thread_trampoline

extern thread_exit

section .text

switch_context:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    mov [rdi], rsp
    mov rsp, rsi

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

thread_trampoline:
    sti
    mov rdi, r12
    call rbx
    call thread_exit
.halt:
    cli
    hlt
    jmp .halt
