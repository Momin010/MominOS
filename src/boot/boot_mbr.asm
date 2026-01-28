[BITS 16]
[ORG 0x7C00]
jmp short start
nop
start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti
    mov [BOOT_DRIVE], dl
    mov si, LOADING_MSG
    call print_string
    mov bx, 0x8000
    mov dh, 15
    mov dl, [BOOT_DRIVE]
    call load_disk
    jmp 0x0000:0x8000
load_disk:
    push dx
    mov ah, 0x02
    mov al, dh
    mov ch, 0x00
    mov dh, 0x00
    mov cl, 0x02
    int 0x13
    jc disk_error
    pop dx
    cmp dh, al
    jne disk_error
    ret
disk_error:
    mov si, DISK_ERR_MSG
    call print_string
    jmp $
print_string:
    pusha
.loop:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp .loop
.done:
    popa
    ret
BOOT_DRIVE db 0
LOADING_MSG db 'Loading MominOS...', 13, 10, 0
DISK_ERR_MSG db 'Disk read error!', 0
times 510-($-$$) db 0
dw 0xAA55
