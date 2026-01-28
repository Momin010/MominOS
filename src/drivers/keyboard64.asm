[BITS 64]
PS2_DATA_PORT equ 0x60
PS2_STATUS_PORT equ 0x64
keyboard_get_scancode:
    in al, PS2_STATUS_PORT
    test al, 1
    jz keyboard_get_scancode
    in al, PS2_DATA_PORT
    ret
keyboard_to_ascii:
    cmp al, 0x80
    ja .released
    mov rbx, scancode_table
    xlatb
    ret
.released:
    xor al, al
    ret
scancode_table:
    db 0, 27, '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 8
    db 9, 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', 13, 0
    db 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", '`', 0, '\'
    db 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 0, '*', 0, ' '
    times 128-($-scancode_table) db 0
