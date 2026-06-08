[BITS 16]
[ORG 0x8000]

stage2_start:
    mov si, MSG_S2
    call print16

    ; Load kernel: 256 sectors from LBA 65 to 0x1000:0x0000 (= 0x10000)
    mov word  [kdap.count],   256
    mov word  [kdap.offset],  0x0000
    mov word  [kdap.segment], 0x1000
    mov dword [kdap.lba_lo],  65
    mov dword [kdap.lba_hi],  0

    mov dl, [0x500]
    mov si, kdap
    mov ah, 0x42
    int 0x13
    jc .disk_err

    mov si, MSG_CPUID
    call print16

    call check_cpuid
    call check_long_mode
    call enable_a20
    call setup_paging

    cli
    lgdt [gdt64_ptr]

    mov eax, cr4
    or  eax, 1 << 5
    mov cr4, eax

    mov ecx, 0xC0000080
    rdmsr
    or  eax, 1 << 8
    wrmsr

    mov eax, cr0
    or  eax, 1 << 31
    mov cr0, eax

    jmp 0x08:long_mode_entry

.disk_err:
    mov si, MSG_DERR
    call print16
    hlt

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
    mov si, MSG_NO_CPUID
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
    mov si, MSG_NO_LM
    call print16
    hlt
.ok:
    ret

enable_a20:
    in  al, 0x92
    or  al, 2
    out 0x92, al
    ret

; Identity-map first 2MB using 4-level paging at 0x1000
setup_paging:
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 4096
    rep stosd

    mov edi, 0x1000
    mov dword [edi],        0x2003   ; PML4[0] -> PDPT at 0x2000
    mov dword [edi+0x1000], 0x3003   ; PDPT[0] -> PD at 0x3000
    mov dword [edi+0x2000], 0x4003   ; PD[0]   -> PT at 0x4000

    ; Map 512 pages (2MB) in PT
    mov edi, 0x4000
    mov ebx, 0x00000003
    mov ecx, 512
.map:
    mov dword [edi], ebx
    add ebx, 0x1000
    add edi, 8
    loop .map
    ret

; --- GDT ---
align 8
gdt64_start:
    dq 0                        ; null
    dq 0x00AF9A000000FFFF       ; code segment (64-bit)
    dq 0x00AF92000000FFFF       ; data segment
gdt64_end:

gdt64_ptr:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; --- 64-bit entry ---
[BITS 64]
long_mode_entry:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, 0x90000        ; stack below kernel
    jmp 0x10000             ; jump to kernel

; --- Kernel DAP ---
kdap:
    db 0x10
    db 0
.count:   dw 256
.offset:  dw 0x0000
.segment: dw 0x1000
.lba_lo:  dd 65
.lba_hi:  dd 0

MSG_S2       db 'Stage 2 active', 13, 10, 0
MSG_CPUID    db 'Checking CPU...', 13, 10, 0
MSG_NO_CPUID db 'No CPUID support!', 0
MSG_NO_LM    db 'No 64-bit support!', 0
MSG_DERR     db 'Kernel load failed!', 0
