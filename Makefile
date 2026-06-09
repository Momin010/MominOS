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

.PHONY: all clean run run-iso iso

all: $(BIN)/kernel.elf

$(BIN):
	mkdir -p $(BIN)

# --- Kernel ---

C_SRCS   = src/kernel/kmain.c src/kernel/pmm.c src/kernel/vmm.c src/kernel/kheap.c \
            src/kernel/sched.c src/kernel/arch.c src/kernel/syscall.c \
            src/kernel/elf.c src/kernel/idt.c src/kernel/memmap.c \
            src/fs/ext2.c src/fs/vfs.c \
            src/drivers/serial.c src/drivers/vga.c src/drivers/pic.c \
            src/drivers/timer.c src/drivers/keyboard.c src/drivers/ata.c \
            src/drivers/tty.c
ASM_SRCS = src/boot/multiboot_entry.asm src/kernel/isr.asm src/kernel/switch.asm \
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

# --- Bootable ISO (GRUB + Multiboot2, BIOS) ---
# The kernel ELF is the boot artifact: GRUB loads it via the multiboot2 command.

iso: $(BIN)/mominos.iso

$(BIN)/mominos.iso: $(BIN)/kernel.elf src/boot/grub.cfg | $(BIN)
	grub-file --is-x86-multiboot2 $(BIN)/kernel.elf
	rm -rf $(BIN)/isodir
	mkdir -p $(BIN)/isodir/boot/grub
	cp $(BIN)/kernel.elf $(BIN)/isodir/boot/mominos.elf
	cp src/boot/grub.cfg $(BIN)/isodir/boot/grub/grub.cfg
	grub-mkrescue -o $@ $(BIN)/isodir

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

# qcow2 root disk, converted from the raw ext2 image. Presented to the guest
# as a plain IDE disk, so the ATA driver mounts it unchanged.
$(BIN)/disk.qcow2: $(BIN)/disk.img | $(BIN)
	qemu-img convert -f raw -O qcow2 $< $@

# --- Run ---
# One command boots the GRUB ISO with the qcow2 hard drive attached.

run: run-iso

run-iso: $(BIN)/mominos.iso $(BIN)/disk.qcow2
	qemu-system-x86_64 \
		-cdrom $(BIN)/mominos.iso \
		-drive file=$(BIN)/disk.qcow2,format=qcow2,if=ide \
		-m 512M \
		-serial stdio \
		-display none \
		-no-reboot -no-shutdown

clean:
	rm -f $(OBJS) $(BIN)/*.bin $(BIN)/*.img $(BIN)/*.iso $(BIN)/*.qcow2 $(BIN)/*.elf
	rm -rf $(BIN)/isodir $(BIN)/fsroot
