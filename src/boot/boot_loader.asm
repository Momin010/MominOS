[BITS 16]
[ORG 0x8000]
loader_entry:
    mov si, MSG_STAGE2
    call print_string_16
    call check_cpuid
    call check_long_mode
    call enable_a20
    call setup_paging
    cli
    lgdt [gdt64_descriptor]
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax
    jmp 0x08:kernel_entry_long_mode
[BITS 16]
print_string_16:
    lodsb
    or al, al
    jz .done
    mov ah, 0x0E
    int 0x10
    jmp print_string_16
.done:
    ret
check_cpuid:
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 1 << 21
    push eax
    popfd
    pushfd
    pop eax
    push ecx
    popfd
    xor eax, ecx
    jz .no_cpuid
    ret
.no_cpuid:
    mov si, ERR_CPUID
    call print_string_16
    hlt
check_long_mode:
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode
    ret
.no_long_mode:
    mov si, ERR_LM
    call print_string_16
    hlt
enable_a20:
    in al, 0x92
    or al, 2
    out 0x92, al
    ret
setup_paging:
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 4096
    rep stosd
    mov edi, cr3
    mov dword [edi], 0x2003
    add edi, 0x1000
    mov dword [edi], 0x3003
    add edi, 0x1000
    mov dword [edi], 0x4003
    add edi, 0x1000
    mov ebx, 0x00000003
    mov ecx, 512
.set_pt:
    mov dword [edi], ebx
    add ebx, 0x1000
    add edi, 8
    loop .set_pt
    ret
MSG_STAGE2 db 'Stage 2 Loader Active...', 13, 10, 0
ERR_CPUID db 'CPUID not supported!', 0
ERR_LM db 'Long Mode not supported!', 0
align 16
gdt64_start:
    dq 0
CodeSegment equ $ - gdt64_start
    dq 0x00AF9A000000FFFF
DataSegment equ $ - gdt64_start
    dq 0x00AF92000000FFFF
gdt64_end:
gdt64_descriptor:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start
[BITS 64]
kernel_entry_long_mode:
    mov ax, DataSegment
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    jmp 0x10000
