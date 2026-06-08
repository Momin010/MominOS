ENTRY(kernel_start)

SECTIONS {
    . = 0x10000;

    .text : {
        *(.text)
    }

    .rodata ALIGN(16) : {
        *(.rodata)
    }

    .data ALIGN(16) : {
        *(.data)
    }

    .bss ALIGN(16) : {
        *(COMMON)
        *(.bss)
    }
}
