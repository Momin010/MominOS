CC     = gcc
CFLAGS = -std=gnu99 -ffreestanding -fno-pic -fno-stack-protector -m64 \
         -mno-red-zone -mno-mmx -mno-sse -mno-sse2 \
         -Wall -Wextra -O2 -Isrc/kernel -Isrc/drivers

AS      = nasm
ASFLAGS = -f elf64

LD      = ld
LDFLAGS = -nostdlib -z noexecstack -T src/kernel/linker.ld

BIN = bin

.PHONY: all clean run

all: $(BIN)/mominos.img

$(BIN):
	mkdir -p $(BIN)

# --- Bootloader ---

$(BIN)/boot_mbr.bin: src/boot/boot_mbr.asm | $(BIN)
	$(AS) -f bin $< -o $@

$(BIN)/boot_loader.bin: src/boot/boot_loader.asm | $(BIN)
	$(AS) -f bin $< -o $@

# Pad stage 2 to exactly 32KB so the kernel lands at 0x10000 in memory.
# MBR loads 128 sectors (64KB) from LBA 1 to 0x8000:
#   0x8000-0xFFFF = stage 2 (32KB padded)
#   0x10000+      = kernel binary
$(BIN)/boot_loader_padded.bin: $(BIN)/boot_loader.bin
	cp $< $@
	truncate -s 32768 $@

# --- Kernel ---

C_SRCS   = src/kernel/kmain.c src/kernel/pmm.c src/kernel/idt.c \
            src/drivers/serial.c src/drivers/vga.c src/drivers/pic.c \
            src/drivers/timer.c src/drivers/keyboard.c
ASM_SRCS = src/kernel/kernel_entry.asm src/kernel/isr.asm

C_OBJS   = $(C_SRCS:.c=.o)
ASM_OBJS = $(ASM_SRCS:.asm=.o)
OBJS     = $(ASM_OBJS) $(C_OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.asm
	$(AS) $(ASFLAGS) $< -o $@

$(BIN)/kernel.elf: $(OBJS) src/kernel/linker.ld | $(BIN)
	$(LD) $(LDFLAGS) $(OBJS) -o $@

$(BIN)/kernel.bin: $(BIN)/kernel.elf
	objcopy -O binary $< $@

# --- Disk image ---
# Pad to exactly 1.44MB (floppy geometry: 80 cylinders x 2 heads x 18 sectors x 512B).
# Floppy BIOS rejects reads to sectors beyond the image size.

$(BIN)/mominos.img: $(BIN)/boot_mbr.bin $(BIN)/boot_loader_padded.bin $(BIN)/kernel.bin
	cat $^ > $@
	truncate -s 1474560 $@

# --- Run ---

run: $(BIN)/mominos.img
	qemu-system-x86_64 \
		-fda $< \
		-serial stdio \
		-display none \
		-no-reboot -no-shutdown

clean:
	rm -f $(OBJS) $(BIN)/*.bin $(BIN)/*.img $(BIN)/*.elf
