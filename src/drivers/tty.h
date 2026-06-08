#ifndef TTY_H
#define TTY_H

#include <stddef.h>
#include <stdint.h>

void tty_init(void);
/* feed one input character into the line discipline (called from IRQ
   or a test thread). Echoes, handles backspace, and wakes a blocked
   reader when a full line (ending in '\n') is available. */
void tty_feed(char c);
/* blocking read of up to size bytes from the line buffer. parks the
   calling thread until input is available. returns bytes copied. */
size_t tty_read(char *buf, size_t size);

#endif
