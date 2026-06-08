[BITS 16]
[ORG 0x8000]

; Stage 2: kernel is already in memory at 0x10000 (loaded by MBR).
; Our job: collect E820 map, enter 64-bit long mode, jump to kernel.

stage2_start:
    mov si, MSG_S2
    call print16

    mov si, MSG_KERNEL
    call print16
    call load_kernel

    call check_cpuid
    call check_long_mode
    call enable_a20
    
    mov si, MSG_E820
    call print16
    call collect_e820
    
    mov si, MSG_E820_DONE
    call print16
    
    mov si, MSG_PAGING
    call print16
    call setup_paging
    
    mov si, MSG_PAGING_DONE
    call print16

    mov si, MSG_GDT
    call print16
    cli
    lgdt [gdt64_ptr]

    ; Marker A: GDT loaded, about to enable PAE
    mov dx, 0x3FD
.waitA: in al, dx
    test al, 0x20
    jz .waitA
    mov dx, 0x3F8
    mov al, 'A'
    out dx, al

    mov eax, cr4
    or  eax, 1 << 5        ; PAE
    mov cr4, eax

    ; Marker B: PAE set, about to set EFER.LME
    mov dx, 0x3FD
.waitB: in al, dx
    test al, 0x20
    jz .waitB
    mov dx, 0x3F8
    mov al, 'B'
    out dx, al

    mov ecx, 0xC0000080    ; EFER MSR
    rdmsr
    or  eax, 1 << 8        ; Long Mode Enable
    wrmsr

    ; Marker C: EFER.LME set, about to enable paging (CR0.PG)
    mov dx, 0x3FD
.waitC: in al, dx
    test al, 0x20
    jz .waitC
    mov dx, 0x3F8
    mov al, 'C'
    out dx, al

    mov eax, cr0
    or  eax, (1 << 31) | (1 << 0)   ; Paging Enable | Protected Mode Enable
    mov cr0, eax

    ; Marker D: paging on, about to far-jump to 64-bit
    mov dx, 0x3FD
.waitD: in al, dx
    test al, 0x20
    jz .waitD
    mov dx, 0x3F8
    mov al, 'D'
    out dx, al

    jmp 0x08:long_mode_entry

; --- 16-bit helpers ---

print16:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print16
.done:
    ret

load_kernel:
    mov ax, 0x1000
    mov es, ax
    xor bx, bx         ; destination 0x1000:0x0000 = 0x10000
    mov di, 65         ; kernel starts after MBR + 64 stage-2 sectors
    mov si, 128        ; load 64KB for now

.next_sector:
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
    jc .fail

    inc di
    add bx, 512
    dec si
    jnz .next_sector
    ret

.fail:
    mov si, ERR_KERNEL_LOAD
    call print16
    cli
    hlt

check_cpuid:
    pushfd
    pop  eax
    mov  ecx, eax
    xor  eax, 1 << 21
    push eax
    popfd
    pushfd
    pop  eax
    push ecx
    popfd
    xor eax, ecx
    jnz .ok
    mov si, ERR_NO_CPUID
    call print16
    hlt
.ok:
    ret

check_long_mode:
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb  .fail
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jnz .ok
.fail:
    mov si, ERR_NO_LM
    call print16
    hlt
.ok:
    ret

enable_a20:
    in  al, 0x92
    or  al, 2
    out 0x92, al
    ret

; Collect BIOS E820 memory map.
; Entries stored at 0x6000 (24 bytes each), count at 0x5FF8.
collect_e820:
    push es
    xor ax, ax
    mov es, ax

    mov word [0x5FF8], 0   ; entry count
    xor ebx, ebx           ; EBX=0 starts the enumeration
    mov di, 0x6000         ; ES:DI points to destination buffer

.next:
    mov eax, 0xE820
    mov edx, 0x534D4150    ; 'SMAP' signature
    mov ecx, 24            ; request 24-byte entries
    int 0x15
    jc  .done              ; carry = no more entries or error

    cmp eax, 0x534D4150    ; BIOS must echo 'SMAP'
    jne .done

    add di, 24
    inc word [0x5FF8]

    test ebx, ebx          ; EBX=0 means this was the last entry
    jz  .done
    jmp .next

.done:
    pop es
    ret

; Identity-map first 2MB with 4-level paging structures at 0x1000-0x4FFF.
setup_paging:
    push es
    xor ax, ax
    mov es, ax

    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 4096
    rep stosd              ; zero 16KB

    mov edi, 0x1000
    mov dword [edi],        0x2003   ; PML4[0] -> PDPT at 0x2000
    mov dword [edi+0x1000], 0x3003   ; PDPT[0] -> PD at 0x3000
    mov dword [edi+0x2000], 0x4003   ; PD[0]   -> PT at 0x4000

    mov edi, 0x4000
    mov ebx, 0x00000003    ; present + writable
    mov ecx, 512
.map:
    mov dword [edi], ebx
    add ebx, 0x1000
    add edi, 8
    loop .map
    pop es
    ret

; --- GDT ---
align 8
gdt64_start:
    dq 0                        ; null descriptor
    dq 0x00AF9A000000FFFF       ; 64-bit code segment
    dq 0x00CF92000000FFFF       ; data segment
gdt64_end:

gdt64_ptr:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; --- 64-bit long mode entry (must be within stage 2 area) ---
[BITS 64]
long_mode_entry:
    ; Marker E: first instruction in 64-bit mode
    mov dx, 0x3FD
.waitE: in al, dx
    test al, 0x20
    jz .waitE
    mov dx, 0x3F8
    mov al, 'E'
    out dx, al

    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, 0x90000

    jmp 0x10000

; --- strings ---
MSG_S2       db 'Stage 2 active', 13, 10, 0
MSG_KERNEL   db 'Loading kernel...', 13, 10, 0
ERR_KERNEL_LOAD db 'Kernel load failed!', 0
ERR_NO_CPUID db 'No CPUID!', 0
ERR_NO_LM    db 'No 64-bit CPU!', 0
MSG_E820     db 'Collecting E820...', 13, 10, 0
MSG_E820_DONE db 'E820 done', 13, 10, 0
MSG_PAGING   db 'Setting up paging...', 13, 10, 0
MSG_PAGING_DONE db 'Paging done', 13, 10, 0
MSG_GDT      db 'Loading GDT...', 13, 10, 0
MSG_LM       db 'Entering long mode...', 13, 10, 0
