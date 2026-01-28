[BITS 64]
VGA_BUFFER equ 0xB8000
VGA_WIDTH equ 80
VGA_HEIGHT equ 25
DEFAULT_COLOR equ 0x0F
vga_clear_screen:
    push rax
    push rcx
    push rdi
    mov rdi, VGA_BUFFER
    mov rax, 0x0F200F200F200F20
    mov rcx, (VGA_WIDTH * VGA_HEIGHT) / 4
    rep stosq
    mov dword [cursor_x], 0
    mov dword [cursor_y], 0
    pop rdi
    pop rcx
    pop rax
    ret
vga_print_string:
    push rax
    push rsi
.loop:
    lodsb
    or al, al
    jz .done
    call vga_put_char
    jmp .loop
.done:
    pop rsi
    pop rax
    ret
vga_put_char:
    push rax
    push rbx
    push rcx
    push rdx
    cmp al, 10
    je .newline
    cmp al, 13
    je .carriage_return
    mov rbx, rax
    mov eax, [cursor_y]
    mov edx, VGA_WIDTH
    mul edx
    add eax, [cursor_x]
    shl eax, 1
    add rax, VGA_BUFFER
    mov [rax], bl
    mov byte [rax+1], DEFAULT_COLOR
    inc dword [cursor_x]
    cmp dword [cursor_x], VGA_WIDTH
    jge .newline
    jmp .done
.newline:
    mov dword [cursor_x], 0
    inc dword [cursor_y]
    jmp .check_scroll
.carriage_return:
    mov dword [cursor_x], 0
    jmp .done
.check_scroll:
    cmp dword [cursor_y], VGA_HEIGHT
    jl .done
    call vga_scroll
.done:
    pop rdx
    pop rcx
    pop rbx
    pop rax
    ret
vga_scroll:
    push rsi
    push rdi
    push rcx
    mov rdi, VGA_BUFFER
    mov rsi, VGA_BUFFER + (VGA_WIDTH * 2)
    mov rcx, (VGA_WIDTH * (VGA_HEIGHT - 1)) * 2
    rep movsb
    mov rdi, VGA_BUFFER + (VGA_WIDTH * (VGA_HEIGHT - 1) * 2)
    mov rax, 0x0F200F200F200F20
    mov rcx, VGA_WIDTH / 4
    rep stosq
    dec dword [cursor_y]
    pop rcx
    pop rdi
    pop rsi
    ret
cursor_x dd 0
cursor_y dd 0
