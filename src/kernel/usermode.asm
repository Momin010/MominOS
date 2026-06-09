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

    ; The userspace syscall ABI (see userspace/libc/syscall.h) only lists
    ; rcx/r11/memory as clobbered, so userspace assumes every other register
    ; survives the syscall. The `syscall` instruction itself destroys rcx/r11;
    ; we must preserve the remaining caller-saved registers the kernel touches
    ; (the C dispatch preserves rbx/rbp/r12-r15 for us). Save the user values
    ; before the argument shuffle below scratches them.
    push qword gs:[8]      ; user_rsp (saved on this thread's kernel stack)
    push rcx               ; user return rip
    push r11               ; user rflags
    push rdi
    push rsi
    push rdx
    push r8
    push r9
    push r10
    sub rsp, 8             ; 9 pushes + 8 -> 16-byte aligned for the call

    mov r8, rdi
    mov r9, rdx
    mov rdi, rax
    mov rdx, rsi
    mov rsi, r8
    mov rcx, r9

    call syscall_dispatch
    add rsp, 8

    pop r10
    pop r9
    pop r8
    pop rdx
    pop rsi
    pop rdi
    pop r11
    pop rcx

    mov rsp, [rsp]        ; restore user_rsp from the pushed slot
    o64 sysret
