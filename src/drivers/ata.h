#ifndef ATA_H
#define ATA_H

#include <stdint.h>

#define ATA_SECTOR_SIZE 512

int ata_init(void);
uint32_t ata_sector_count(void);
int ata_read(uint32_t lba, uint32_t sectors, void *buffer);
int ata_write(uint32_t lba, uint32_t sectors, const void *buffer);
int ata_self_test(void);

#endif
