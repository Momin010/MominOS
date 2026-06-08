CC      = gcc
CFLAGS  = -m64 -ffreestanding -fno-stack-protector -fno-pic -fno-pie \
           -mno-red-zone -nostdlib -nostdinc -O2 -Wall -Wextra \
           -Isrc/drivers -Isrc/kernel

NASM    = nasm
AS_BIN  = -f bin
AS_ELF  = -f elf64

LD      = ld
LDFLAGS = -T linker.ld -nostdlib -z noexecstack

BIN = bin

.PHONY: all clean run

all: $(BIN)/mominos.img

$(BIN):
	mkdir -p $(BIN)

# Stage 1 MBR
$(BIN)/boot_mbr.bin: src/boot/boot_mbr.asm | $(BIN)
	$(NASM) $(AS_BIN) $< -o $@

# Stage 2 loader (padded to 32KB so kernel lands at 0x10000)
$(BIN)/boot_loader.bin: src/boot/boot_loader.asm | $(BIN)
	$(NASM) $(AS_BIN) $< -o $@

$(BIN)/boot_loader_padded.bin: $(BIN)/boot_loader.bin
	cp $< $@
	truncate -s 32768 $@

# Kernel assembly entry stub
$(BIN)/kernel_entry.o: src/kernel/kernel_entry.asm | $(BIN)
	$(NASM) $(AS_ELF) $< -o $@

# Kernel C files
$(BIN)/kmain.o: src/kernel/kmain.c | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/serial.o: src/drivers/serial.c | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/vga.o: src/drivers/vga.c | $(BIN)
	$(CC) $(CFLAGS) -c $< -o $@

# Link kernel ELF then flatten to binary
KERNEL_OBJS = $(BIN)/kernel_entry.o $(BIN)/kmain.o $(BIN)/serial.o $(BIN)/vga.o

$(BIN)/kernel.elf: $(KERNEL_OBJS) linker.ld
	$(LD) $(LDFLAGS) $(KERNEL_OBJS) -o $@

$(BIN)/kernel.bin: $(BIN)/kernel.elf
	objcopy -O binary $< $@

# Final disk image: MBR + stage2 (padded to 32KB) + kernel
$(BIN)/mominos.img: $(BIN)/boot_mbr.bin $(BIN)/boot_loader_padded.bin $(BIN)/kernel.bin
	cat $^ > $@

run: $(BIN)/mominos.img
	qemu-system-x86_64 \
		-drive file=$<,format=raw,index=0,media=disk \
		-serial stdio \
		-no-reboot \
		-no-shutdown \
		-d int,cpu_reset 2>/dev/null

clean:
	rm -rf $(BIN)
