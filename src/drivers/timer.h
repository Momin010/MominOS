#pragma once

#include <stdint.h>

void timer_init(uint32_t hz);
void timer_irq(void);
uint64_t timer_ticks(void);

