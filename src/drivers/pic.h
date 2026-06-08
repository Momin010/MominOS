#pragma once

#include <stdint.h>

#define PIC_MASTER_OFFSET 0x20
#define PIC_SLAVE_OFFSET  0x28

void pic_remap(void);
void pic_send_eoi(uint8_t irq);
void pic_mask_all(void);
void pic_set_mask(uint8_t irq);
void pic_clear_mask(uint8_t irq);

