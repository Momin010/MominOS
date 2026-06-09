[BITS 64]

global _start
extern main

section .text

; Kernel enters here with rsp 16-byte aligned and pointing at:
;   [rsp]      argc   (qword)
;   [rsp+8]    argv[0]
;   ...
;   [rsp+8*argc] NULL
_start:
    mov rdi, [rsp]          ; argc
    lea rsi, [rsp + 8]      ; argv
    ; rsp is 16-byte aligned here. `call main` pushes 8 bytes, so main
    ; sees rsp % 16 == 8, exactly as the System V ABI requires.
    call main

    ; main returned: exit(rax)
    mov rdi, rax
    mov rax, 6             ; SYS_EXIT
    syscall
.hang:
    jmp .hang
