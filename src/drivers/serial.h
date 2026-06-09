#pragma once

void serial_init(void);
void serial_irq(void);
void serial_putc(char c);
void serial_print(const char *s);
void serial_print_hex(unsigned long val);
