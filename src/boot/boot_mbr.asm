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

    ; LBA read: 64 sectors from LBA 1 to 0x0800:0x0000 (= 0x8000)
    mov word  [dap.count],   64
    mov word  [dap.offset],  0x0000
    mov word  [dap.segment], 0x0800
    mov dword [dap.lba_lo],  1
    mov dword [dap.lba_hi],  0

    mov dl, [0x500]
    mov si, dap
    mov ah, 0x42
    int 0x13
    jc  .disk_err

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

; Disk Address Packet
dap:
    db 0x10
    db 0
.count:   dw 64
.offset:  dw 0
.segment: dw 0x0800
.lba_lo:  dd 1
.lba_hi:  dd 0

times 510-($-$$) db 0
dw 0xAA55
