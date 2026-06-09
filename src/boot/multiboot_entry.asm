; MominOS Multiboot2 entry.
;
; GRUB loads this ELF in 32-bit protected mode (paging off, A20 on) per the
; Multiboot2 spec. We must perform the 32->64-bit transition ourselves:
; set up PAE page tables + a 64-bit GDT, enable EFER.LME, turn on paging,
; far-jump into 64-bit code, then hand the Multiboot2 info pointer to kmain.
;
; All BIOS-call logic from the old real-mode bootloader is gone: GRUB already
; loaded the kernel, enabled A20, and collected the memory map (delivered in
; the Multiboot2 info struct, parsed later in C).

MB2_MAGIC      equ 0xE85250D6      ; multiboot2 header magic
MB2_ARCH       equ 0               ; i386 (32-bit protected mode)
MB2_HDR_LEN    equ mb2_header_end - mb2_header_start
MB2_CHECKSUM   equ -(MB2_MAGIC + MB2_ARCH + MB2_HDR_LEN)

MB2_BOOT_MAGIC equ 0x36D76289      ; value GRUB leaves in EAX at handoff

; ---------------------------------------------------------------------------
; Multiboot2 header. Must be 8-byte aligned and within the first 32KB of the
; kernel image. The linker places .multiboot first (see linker.ld).
; ---------------------------------------------------------------------------
section .multiboot
align 8
mb2_header_start:
    dd MB2_MAGIC
    dd MB2_ARCH
    dd MB2_HDR_LEN
    dd MB2_CHECKSUM

    ; End tag (type 0, size 8).
    align 8
    dw 0
    dw 0
    dd 8
mb2_header_end:

; ---------------------------------------------------------------------------
; 32-bit entry point. GRUB jumps here with:
;   EAX = 0x36D76289 (boot magic), EBX = physical addr of mb2 info struct.
; ESP is undefined, so set our stack before any call.
; ---------------------------------------------------------------------------
section .text
[BITS 32]
global _start
extern kmain

_start:
    cli
    mov esp, boot_stack_top

    ; Preserve the multiboot info pointer (EBX) and magic (EAX) across the
    ; setup code, which clobbers EAX/ECX/EDI. We stash them in memory.
    mov [mb_info_ptr], ebx
    mov [mb_magic], eax

    ; Marker A: 32-bit GRUB entry reached.
    call serial_init32
    mov al, 'A'
    call serial_putc32

    ; --- Build PAE page tables at phys 0x1000..0x4FFF ---
    ; PML4 @0x1000, PDPT @0x2000, PD @0x3000, PT @0x4000.
    ; (vmm.c reserves 0x1000..0x4FFF to match this.)
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 4096
    rep stosd                       ; zero 16KB (0x1000..0x4FFF)

    mov edi, 0x1000
    mov dword [edi],        0x2003   ; PML4[0] -> PDPT | present+writable
    mov dword [edi+0x1000], 0x3003   ; PDPT[0] -> PD
    mov dword [edi+0x2000], 0x4003   ; PD[0]   -> PT (first 2MB via 4KB pages)

    ; PD[1..15] -> 2MB huge pages, identity-mapping 2MB..32MB so the kernel,
    ; .bss (incl. the ~128KB pmm bitmap) and stack are all covered before
    ; vmm_init builds the real direct map.
    mov edi, 0x3000 + 8             ; PD[1]
    mov eax, 0x200000 | 0x83        ; 2MB | present+writable+PS(huge)
    mov ecx, 15                     ; PD[1]..PD[15] -> up to 32MB
.fill_pd:
    mov dword [edi], eax
    mov dword [edi+4], 0
    add eax, 0x200000
    add edi, 8
    loop .fill_pd

    ; PT: identity-map first 2MB with 4KB pages.
    mov edi, 0x4000
    mov eax, 0x00000003             ; present+writable
    mov ecx, 512
.fill_pt:
    mov dword [edi], eax
    mov dword [edi+4], 0
    add eax, 0x1000
    add edi, 8
    loop .fill_pt

    mov al, 'B'
    call serial_putc32

    ; --- Load 64-bit GDT ---
    lgdt [gdt64_ptr]

    ; --- Enable PAE (CR4.PAE) ---
    mov eax, cr4
    or  eax, 1 << 5
    mov cr4, eax

    ; --- Set EFER.LME ---
    mov ecx, 0xC0000080
    rdmsr
    or  eax, 1 << 8
    wrmsr

    mov al, 'C'
    call serial_putc32

    ; --- Enable paging (CR0.PG); PE is already set by GRUB ---
    mov eax, cr0
    or  eax, (1 << 31) | (1 << 0)
    mov cr0, eax

    mov al, 'D'
    call serial_putc32

    ; --- Far-jump into 64-bit code segment ---
    jmp 0x08:long_mode_entry

; --- 32-bit serial helpers (port I/O works in protected mode) ---
serial_init32:
    push eax
    push edx
    mov dx, 0x3F9          ; disable interrupts
    mov al, 0x00
    out dx, al
    mov dx, 0x3FB          ; enable DLAB
    mov al, 0x80
    out dx, al
    mov dx, 0x3F8          ; divisor low (115200/3 -> 38400)
    mov al, 0x03
    out dx, al
    mov dx, 0x3F9          ; divisor high
    mov al, 0x00
    out dx, al
    mov dx, 0x3FB          ; 8N1
    mov al, 0x03
    out dx, al
    mov dx, 0x3FA          ; enable FIFO
    mov al, 0xC7
    out dx, al
    mov dx, 0x3FC          ; DTR/RTS/OUT2
    mov al, 0x0B
    out dx, al
    pop edx
    pop eax
    ret

serial_putc32:
    push eax
    push edx
    mov ah, al             ; save char
    mov dx, 0x3FD
.wait:
    in al, dx
    test al, 0x20
    jz .wait
    mov dx, 0x3F8
    mov al, ah
    out dx, al
    pop edx
    pop eax
    ret

; ---------------------------------------------------------------------------
; 64-bit entry.
; ---------------------------------------------------------------------------
[BITS 64]
long_mode_entry:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov rsp, boot_stack_top

    ; Marker E: 64-bit mode reached.
    mov dx, 0x3FD
.waitE:
    in al, dx
    test al, 0x20
    jz .waitE
    mov dx, 0x3F8
    mov al, 'E'
    out dx, al

    ; Mask all PIC IRQs (no IDT yet).
    mov al, 0xFF
    out 0xA1, al
    out 0x21, al

    ; Pass the multiboot2 info pointer as the first arg (RDI) to kmain.
    xor rdi, rdi
    mov edi, [mb_info_ptr]
    cld
    call kmain

.halt:
    cli
    hlt
    jmp .halt

; ---------------------------------------------------------------------------
; GDT (64-bit)
; ---------------------------------------------------------------------------
section .rodata
align 8
gdt64_start:
    dq 0                        ; null
    dq 0x00AF9A000000FFFF       ; 64-bit code
    dq 0x00CF92000000FFFF       ; data
gdt64_end:
gdt64_ptr:
    dw gdt64_end - gdt64_start - 1
    dq gdt64_start

; ---------------------------------------------------------------------------
; BSS: boot stack + saved handoff registers.
; ---------------------------------------------------------------------------
section .bss
align 16
boot_stack_bottom:
    resb 16384
boot_stack_top:

mb_info_ptr: resd 1
mb_magic:    resd 1
