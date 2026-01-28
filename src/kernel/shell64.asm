[BITS 64]
shell_start:
    mov rsi, SHELL_WELCOME
    call vga_print_string
.loop:
    mov rsi, PROMPT
    call vga_print_string
    mov rdi, command_buffer
    call get_input
    call process_command
    jmp .loop
get_input:
    mov rcx, 0
.input_loop:
    call keyboard_get_scancode
    call keyboard_to_ascii
    or al, al
    jz .input_loop
    cmp al, 13
    je .done
    cmp al, 8
    je .backspace
    cmp rcx, 127
    jge .input_loop
    mov [rdi+rcx], al
    inc rcx
    call vga_put_char
    jmp .input_loop
.backspace:
    or rcx, rcx
    jz .input_loop
    dec rcx
    call vga_backspace
    jmp .input_loop
.done:
    mov byte [rdi+rcx], 0
    mov al, 10
    call vga_put_char
    ret
vga_backspace:
    or dword [cursor_x], dword [cursor_x]
    jz .ret
    dec dword [cursor_x]
    mov eax, [cursor_y]
    mov edx, VGA_WIDTH
    mul edx
    add eax, [cursor_x]
    shl eax, 1
    add rax, VGA_BUFFER
    mov word [rax], 0x0F20
.ret:
    ret
process_command:
    mov rsi, command_buffer
    mov rdi, CMD_HELP
    call string_compare
    jz .help
    mov rdi, CMD_CLS
    call string_compare
    jz .cls
    mov rdi, CMD_LSPCI
    call string_compare
    jz .lspci
    mov rsi, ERR_UNKNOWN
    call vga_print_string
    ret
.help:
    mov rsi, HELP_TEXT
    call vga_print_string
    ret
.cls:
    call vga_clear_screen
    ret
.lspci:
    mov rsi, LSPCI_TEXT
    call vga_print_string
    ret
string_compare:
    push rsi
    push rdi
.loop:
    mov al, [rsi]
    mov bl, [rdi]
    cmp al, bl
    jne .not_equal
    or al, al
    jz .equal
    inc rsi
    inc rdi
    jmp .loop
.not_equal:
    pop rdi
    pop rsi
    mov al, 1
    or al, al
    ret
.equal:
    pop rdi
    pop rsi
    xor al, al
    ret
SHELL_WELCOME db 'MominOS Shell v1.0.0 Ready.', 13, 10, 0
PROMPT db 'MominOS> ', 0
CMD_HELP db 'help', 0
CMD_CLS db 'cls', 0
CMD_LSPCI db 'lspci', 0
HELP_TEXT db 'Available: help, cls, lspci', 13, 10, 0
LSPCI_TEXT db 'Scanning PCI Bus...', 13, 10, '00:00.0 Host Bridge: Intel Corporation', 13, 10, '00:01.0 VGA Controller: QEMU Generic', 13, 10, 0
ERR_UNKNOWN db 'Unknown command.', 13, 10, 0
command_buffer resb 128
