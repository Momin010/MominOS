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

; Higher-half layout. Must stay in sync with src/kernel/linker.ld and vmm.c.
;   KERNEL_VMA      = 0xFFFFFFFF80000000  (PML4[511], PDPT[510], PD[0])  -2GB
;   DIRECT_MAP_BASE = 0xFFFF808000000000  (PML4[257], PDPT[0],   PD[0])
; The kernel is linked at KERNEL_VMA but loaded (LMA) at phys 1MB, so the
; phys->virt rule is virt = phys + KERNEL_VMA for image symbols. While running
; 32-bit (paging just enabled, RIP still at low phys) we reach high-VMA symbols
; through (symbol - KERNEL_VMA), their physical load address.
KERNEL_VMA equ 0xFFFFFFFF80000000
V2P        equ KERNEL_VMA            ; subtract from a kernel VMA to get its LMA

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
    ; RSP must point at the boot stack's physical (load) address while paging
    ; is off / identity-only. boot_stack_top is a high VMA, so subtract V2P.
    mov esp, boot_stack_top - V2P

    ; Preserve the multiboot info pointer (EBX) and magic (EAX) across the
    ; setup code, which clobbers EAX/ECX/EDI. We stash them in memory, again
    ; addressing the (high-VMA) variables through their physical load address.
    mov [mb_info_ptr - V2P], ebx
    mov [mb_magic - V2P], eax

    ; Marker A: 32-bit GRUB entry reached.
    call serial_init32
    mov al, 'A'
    call serial_putc32

    ; --- Build PAE boot page tables at phys 0x1000..0x8FFF ---
    ;   PML4      @ 0x1000
    ;   PDPT_low  @ 0x2000   (PML4[0]   -> temporary low identity map)
    ;   PD_low    @ 0x3000   (covers 0..32MB; PD[0] -> PT_low)
    ;   PT_low    @ 0x4000   (first 2MB, 4KB pages)
    ;   PDPT_kern @ 0x5000   (PML4[511] -> higher-half kernel, KERNEL_VMA)
    ;   PD_kern   @ 0x6000   (PDPT_kern[510] -> 0..32MB via 2MB pages)
    ;   PDPT_dmap @ 0x7000   (PML4[257] -> direct map, DIRECT_MAP_BASE)
    ;   PD_dmap   @ 0x8000   (PDPT_dmap[0] -> 0..32MB via 2MB pages)
    ; (vmm.c reserves 0x1000..0x8FFF to match this.)
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 0x8000 / 4             ; zero 32KB (0x1000..0x8FFF)
    rep stosd

    ; PML4 entries.
    mov edi, 0x1000
    mov dword [edi + 0*8],   0x2003 ; PML4[0]   -> PDPT_low (temp identity)
    mov dword [edi + 257*8], 0x7003 ; PML4[257] -> PDPT_dmap (direct map)
    mov dword [edi + 511*8], 0x5003 ; PML4[511] -> PDPT_kern (high kernel)

    ; PDPT_low[0] -> PD_low ; PDPT_kern[510] -> PD_kern ; PDPT_dmap[0] -> PD_dmap
    mov dword [0x2000 + 0*8],   0x3003
    mov dword [0x5000 + 510*8], 0x6003
    mov dword [0x7000 + 0*8],   0x8003

    ; PD_low[0] -> PT_low (first 2MB via 4KB pages so phys 0 page exists).
    mov dword [0x3000 + 0*8], 0x4003

    ; PD_low[1..15] -> 2MB huge pages, identity 2MB..32MB.
    mov edi, 0x3000 + 8             ; PD_low[1]
    mov eax, 0x200000 | 0x83        ; 2MB | present+writable+PS(huge)
    mov ecx, 15
.fill_pd_low:
    mov dword [edi], eax
    mov dword [edi+4], 0
    add eax, 0x200000
    add edi, 8
    loop .fill_pd_low

    ; PT_low: identity-map first 2MB with 4KB pages.
    mov edi, 0x4000
    mov eax, 0x00000003             ; present+writable
    mov ecx, 512
.fill_pt_low:
    mov dword [edi], eax
    mov dword [edi+4], 0
    add eax, 0x1000
    add edi, 8
    loop .fill_pt_low

    ; PD_kern[0..15] and PD_dmap[0..15] -> 2MB huge pages, phys 0..32MB.
    ; These map the higher-half kernel and the direct-map window so that
    ; execution can continue at a high RIP and vmm_init can dereference
    ; (phys + DIRECT_MAP_BASE) from its very first call.
    mov ebx, 0x6000                 ; PD_kern
    mov edx, 0x8000                 ; PD_dmap
    mov eax, 0x00000083             ; phys 0 | present+writable+PS(huge)
    mov ecx, 16
.fill_pd_high:
    mov [ebx], eax
    mov dword [ebx+4], 0
    mov [edx], eax
    mov dword [edx+4], 0
    add eax, 0x200000
    add ebx, 8
    add edx, 8
    loop .fill_pd_high

    mov al, 'B'
    call serial_putc32

    ; --- Load 64-bit GDT (address through its physical load address) ---
    lgdt [gdt64_ptr - V2P]

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

    ; --- Far-jump into 64-bit code at its LOW (identity) address first. The
    ; high-VMA mapping is live, but RIP is still low; jump to the low label,
    ; then in 64-bit reload RIP/RSP to the high VMA. ---
    jmp 0x08:(long_mode_low - V2P)

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
; Running here at the LOW (identity) RIP right after the far jump. Immediately
; reload segment registers, then jump to the HIGH virtual address of the
; kernel so all subsequent RIP-relative accesses use the higher-half mapping.
long_mode_low:
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Absolute jump to the high VMA of long_mode_high.
    mov rax, long_mode_high
    jmp rax

long_mode_high:
    ; Now executing at the higher-half VMA. Load the high virtual stack.
    mov rsp, boot_stack_top

    ; Marker E: 64-bit higher-half reached.
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

    ; Pass the multiboot2 info pointer (low phys address) as the first arg
    ; (RDI) to kmain. memmap parses it under the still-live low identity map.
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
    ; 32-bit lgdt loads a 32-bit base; store the PHYSICAL (low) address of the
    ; GDT so it is reachable via the identity map at the far jump. gdt64_start
    ; is a high VMA after the higher-half relink, whose truncated low 32 bits
    ; would point into an unmapped region and triple-fault the far jump.
    dq gdt64_start - V2P

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
