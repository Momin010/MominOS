CC     = gcc
CFLAGS = -std=gnu99 -ffreestanding -fno-pic -fno-stack-protector -m64 \
         -mno-red-zone -mno-mmx -mno-sse -mno-sse2 \
         -Wall -Wextra -O2 -Isrc/kernel -Isrc/drivers -Isrc/fs

AS      = nasm
ASFLAGS = -f elf64

LD      = ld
LDFLAGS = -nostdlib -z noexecstack -T src/kernel/linker.ld
USER_CFLAGS = -nostdlib -ffreestanding -fno-pic -no-pie -m64 \
              -mno-red-zone -Wall -Wextra -O2

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

C_SRCS   = src/kernel/kmain.c src/kernel/pmm.c src/kernel/vmm.c src/kernel/kheap.c \
            src/kernel/sched.c src/kernel/arch.c src/kernel/syscall.c \
            src/kernel/elf.c src/kernel/idt.c \
            src/fs/ext2.c src/fs/vfs.c \
            src/drivers/serial.c src/drivers/vga.c src/drivers/pic.c \
            src/drivers/timer.c src/drivers/keyboard.c src/drivers/ata.c \
            src/drivers/tty.c
ASM_SRCS = src/kernel/kernel_entry.asm src/kernel/isr.asm src/kernel/switch.asm \
           src/kernel/usermode.asm

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

# --- Userspace ---
# crt0 + libc, then link each program statically at 0x400000 with no PIE.

LIBC_C_OBJS = userspace/libc/string.o userspace/libc/stdlib.o userspace/libc/stdio.o
LIBC_CRT0   = userspace/libc/crt0.o
USER_LDFLAGS = -nostdlib -no-pie -Wl,-z,noexecstack -Wl,--build-id=none -Wl,-Ttext=0x400000 -Wl,-e,_start

userspace/libc/%.o: userspace/libc/%.c
	$(CC) $(USER_CFLAGS) -Iuserspace/libc -c $< -o $@

userspace/libc/crt0.o: userspace/libc/crt0.asm
	$(AS) $(ASFLAGS) $< -o $@

# libc-linked programs: crt0 + program + libc objects.
USER_PROGS = init sh ls cat echo argtest
USER_BINS  = $(addprefix $(BIN)/,$(USER_PROGS))

$(BIN)/%: userspace/%.c $(LIBC_CRT0) $(LIBC_C_OBJS) | $(BIN)
	$(CC) $(USER_CFLAGS) -Iuserspace/libc $(USER_LDFLAGS) \
		$(LIBC_CRT0) $< $(LIBC_C_OBJS) -o $@

$(BIN)/disk.img: Makefile $(USER_BINS) | $(BIN)
	rm -rf $(BIN)/fsroot
	mkdir -p $(BIN)/fsroot/bin
	printf 'hello from MominOS ext2\n' > $(BIN)/fsroot/hello.txt
	dd if=/dev/zero of=$(BIN)/fsroot/big.bin bs=1M count=5 status=none
	cp $(BIN)/init $(BIN)/fsroot/init
	cp $(USER_BINS) $(BIN)/fsroot/bin/
	rm -f $@
	mke2fs -q -F -t ext2 -b 4096 -d $(BIN)/fsroot $@ 64M

# --- Run ---

run: $(BIN)/mominos.img $(BIN)/disk.img
	qemu-system-x86_64 \
		-drive file=$<,format=raw,if=floppy \
		-drive file=$(BIN)/disk.img,format=raw,if=ide \
		-m 512M \
		-serial stdio \
		-display none \
		-no-reboot -no-shutdown

clean:
	rm -f $(OBJS) $(BIN)/*.bin $(BIN)/*.img $(BIN)/*.elf
