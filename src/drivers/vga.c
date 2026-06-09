#include "vga.h"

/* The kernel runs higher-half: physical 0xB8000 is reachable through the
   kernel's flat map at KERNEL_VMA + 0xB8000 (PML4[511], shared into every
   address space), not via an identity map (which no longer exists). */
#define VGA_BUF  ((volatile unsigned short *)(0xFFFFFFFF80000000ULL + 0xB8000))
#define WIDTH    80
#define HEIGHT   25

static int cur_x = 0;
static int cur_y = 0;
static unsigned char cur_color = 0x0F;  /* white on black */

void vga_set_color(unsigned char color) {
    cur_color = color;
}

void vga_clear(void) {
    unsigned short blank = (unsigned short)(cur_color << 8) | ' ';
    for (int i = 0; i < WIDTH * HEIGHT; i++)
        VGA_BUF[i] = blank;
    cur_x = cur_y = 0;
}

static void scroll(void) {
    for (int y = 0; y < HEIGHT - 1; y++)
        for (int x = 0; x < WIDTH; x++)
            VGA_BUF[y * WIDTH + x] = VGA_BUF[(y + 1) * WIDTH + x];
    unsigned short blank = (unsigned short)(cur_color << 8) | ' ';
    for (int x = 0; x < WIDTH; x++)
        VGA_BUF[(HEIGHT - 1) * WIDTH + x] = blank;
    cur_y--;
}

void vga_putc(char c) {
    if (c == '\n') {
        cur_x = 0;
        cur_y++;
    } else if (c == '\r') {
        cur_x = 0;
    } else if (c == '\b') {
        if (cur_x > 0) {
            cur_x--;
            VGA_BUF[cur_y * WIDTH + cur_x] = (unsigned short)(cur_color << 8) | ' ';
        }
    } else {
        VGA_BUF[cur_y * WIDTH + cur_x] = (unsigned short)(cur_color << 8) | (unsigned char)c;
        if (++cur_x >= WIDTH) {
            cur_x = 0;
            cur_y++;
        }
    }
    if (cur_y >= HEIGHT)
        scroll();
}

void vga_print(const char *s) {
    while (*s)
        vga_putc(*s++);
}
