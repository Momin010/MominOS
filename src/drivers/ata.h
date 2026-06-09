#ifndef ATA_H
#define ATA_H

#include <stdint.h>

#define ATA_SECTOR_SIZE 512

/* Standard ATA positions: 0 primary master, 1 primary slave,
   2 secondary master, 3 secondary slave. */
#define ATA_DRIVE_COUNT 4

int ata_init(void);

/* Backward-compatible primary-master (drive 0) API used by the VFS. */
uint32_t ata_sector_count(void);
int ata_read(uint32_t lba, uint32_t sectors, void *buffer);
int ata_write(uint32_t lba, uint32_t sectors, const void *buffer);

/* Multi-drive API. */
int ata_drive_present(int drive);
uint32_t ata_drive_sector_count(int drive);
int ata_read_drive(int drive, uint32_t lba, uint32_t sectors, void *buffer);
int ata_write_drive(int drive, uint32_t lba, uint32_t sectors,
                    const void *buffer);

int ata_self_test(void);
void ata_probe_extra_drives(void);

#endif
