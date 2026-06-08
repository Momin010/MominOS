[BITS 16]
[ORG 0x7C00]

start:
    cli
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00
    sti

    mov [0x500], dl         ; save boot drive for stage 2

    mov si, MSG_LOAD
    call print

    ; Read using CHS (floppy compatible) - 128 sectors from track 0 sector 2
    ; 128 sectors = 8 tracks (16 sectors/track * 2 sides)
    ; We'll read in chunks since BIOS may not read all at once
    mov word [dap.count], 128
    mov word [dap.offset], 0x0000
    mov word [dap.segment], 0x0800

    ; Try extended read first (for hard disk / USB)
    mov ah, 0x41
    mov bx, 0x55AA
    int 0x13
    jnc .check_ext
    jmp .use_chs

.check_ext:
    cmp bx, 0xAA55
    jne .use_chs
    test cl, 1
    jz .use_chs

    ; Extended read supported
    mov dword [dap.lba_lo], 1
    mov dword [dap.lba_hi], 0
    mov dl, [0x500]
    mov si, dap
    mov ah, 0x42
    int 0x13
    jnc .jump_stage2

.use_chs:
    ; CHS fallback for floppy: read LBA 1..128 to 0x8000.
    mov ax, 0x0800
    mov es, ax
    xor bx, bx         ; destination offset
    mov di, 1          ; current LBA
    mov si, 128        ; sectors to read

.chs_loop:
    mov ax, di
    xor dx, dx
    mov cx, 18
    div cx             ; AX = LBA / 18, DX = LBA % 18
    push dx            ; sector index

    xor dx, dx
    mov cx, 2
    div cx             ; AX = cylinder, DX = head
    mov ch, al
    mov dh, dl
    pop ax
    mov cl, al
    inc cl             ; sector number is 1-based

    mov dl, [0x500]
    mov ah, 0x02
    mov al, 1
    int 0x13
    jc .disk_err

    inc di
    add bx, 512
    dec si
    jnz .chs_loop

.jump_stage2:
    jmp 0x0000:0x8000

.disk_err:
    mov si, MSG_ERR
    call print
    cli
    hlt

print:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print
.done:
    ret

MSG_LOAD db 'MominOS loading...', 13, 10, 0
MSG_ERR  db 'Disk read failed!', 0
MSG_NO_EXT db 'No extended read!', 0

; Disk Address Packet
dap:
    db 0x10
    db 0
.count:   dw 128
.offset:  dw 0x0000
.segment: dw 0x0800
.lba_lo:  dd 1
.lba_hi:  dd 0

times 510-($-$$) db 0
dw 0xAA55
