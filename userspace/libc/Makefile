CC = gcc
CFLAGS = -nostdlib -ffreestanding -m64 -Wall -Wextra -I.
AS = nasm
ASFLAGS = -f elf64

OBJS = string.o stdlib.o stdio.o
LIB = libc.a

all: $(LIB) test

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: test.c $(LIB)
	$(CC) $(CFLAGS) -o test test.c -L. -lc

clean:
	rm -f $(OBJS) $(LIB) test

.PHONY: all clean test