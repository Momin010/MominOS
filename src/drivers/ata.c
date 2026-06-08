#include "ata.h"
#include "kheap.h"
#include "sched.h"
#include "serial.h"

#define ATA_DATA       0x1F0
#define ATA_ERROR      0x1F1
#define ATA_FEATURES   0x1F1
#define ATA_SECCOUNT   0x1F2
#define ATA_LBA0       0x1F3
#define ATA_LBA1       0x1F4
#define ATA_LBA2       0x1F5
#define ATA_DRIVE      0x1F6
#define ATA_STATUS     0x1F7
#define ATA_COMMAND    0x1F7
#define ATA_ALT_STATUS 0x3F6
#define ATA_CONTROL    0x3F6

#define ATA_SR_ERR 0x01
#define ATA_SR_DRQ 0x08
#define ATA_SR_DF  0x20
#define ATA_SR_BSY 0x80

#define ATA_CMD_READ     0x20
#define ATA_CMD_WRITE    0x30
#define ATA_CMD_IDENTIFY 0xEC
#define ATA_CMD_FLUSH    0xE7

#define ATA_POLL_LIMIT 100000ULL
#define ATA_YIELD_EVERY 1024ULL
#define ATA_TEST_LBA 2048U
#define ATA_STREAM_SECTORS 4096U

static uint32_t detected_sectors;
static int disk_present;

static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;

    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

static inline void insw(uint16_t port, void *buffer, uint64_t words) {
    __asm__ volatile ("rep insw"
                      : "+D"(buffer), "+c"(words)
                      : "d"(port)
                      : "memory");
}

static inline void outsw(uint16_t port, const void *buffer, uint64_t words) {
    __asm__ volatile ("rep outsw"
                      : "+S"(buffer), "+c"(words)
                      : "d"(port)
                      : "memory");
}

static void ata_delay_400ns(void) {
    inb(ATA_ALT_STATUS);
    inb(ATA_ALT_STATUS);
    inb(ATA_ALT_STATUS);
    inb(ATA_ALT_STATUS);
}

static int wait_not_busy(void) {
    for (uint64_t i = 0; i < ATA_POLL_LIMIT; i++) {
        uint8_t status = inb(ATA_STATUS);

        if (status == 0xFF)
            return 0;

        if ((status & ATA_SR_BSY) == 0)
            return 1;

        if ((i % ATA_YIELD_EVERY) == 0)
            sched_yield();
    }

    return 0;
}

static int wait_drq(void) {
    for (uint64_t i = 0; i < ATA_POLL_LIMIT; i++) {
        uint8_t status = inb(ATA_STATUS);

        if (status == 0xFF)
            return 0;

        if (status & (ATA_SR_ERR | ATA_SR_DF))
            return 0;

        if ((status & ATA_SR_BSY) == 0 && (status & ATA_SR_DRQ))
            return 1;

        if ((i % ATA_YIELD_EVERY) == 0)
            sched_yield();
    }

    return 0;
}

static void select_lba(uint32_t lba) {
    outb(ATA_DRIVE, 0xE0 | ((lba >> 24) & 0x0F));
    ata_delay_400ns();
}

static int issue_lba_command(uint32_t lba, uint16_t sectors, uint8_t command) {
    uint8_t count = sectors == 256 ? 0 : (uint8_t)sectors;

    if (!wait_not_busy())
        return 0;

    select_lba(lba);
    outb(ATA_FEATURES, 0);
    outb(ATA_SECCOUNT, count);
    outb(ATA_LBA0, lba & 0xFF);
    outb(ATA_LBA1, (lba >> 8) & 0xFF);
    outb(ATA_LBA2, (lba >> 16) & 0xFF);
    outb(ATA_COMMAND, command);
    ata_delay_400ns();

    return 1;
}

static uint16_t chunk_sectors(uint32_t sectors) {
    if (sectors > 128)
        return 128;
    return (uint16_t)sectors;
}

int ata_init(void) {
    uint16_t identify[256];
    uint8_t status;

    detected_sectors = 0;
    disk_present = 0;

    outb(ATA_CONTROL, 0);
    if (!wait_not_busy()) {
        serial_print("[ATA] no primary disk\n");
        return 0;
    }

    select_lba(0);
    outb(ATA_SECCOUNT, 0);
    outb(ATA_LBA0, 0);
    outb(ATA_LBA1, 0);
    outb(ATA_LBA2, 0);
    outb(ATA_COMMAND, ATA_CMD_IDENTIFY);
    ata_delay_400ns();

    status = inb(ATA_STATUS);
    if (status == 0 || status == 0xFF) {
        serial_print("[ATA] no primary disk\n");
        return 0;
    }

    if (!wait_drq()) {
        serial_print("[ATA] identify failed err=");
        serial_print_hex(inb(ATA_ERROR));
        serial_print("\n");
        return 0;
    }

    insw(ATA_DATA, identify, 256);

    detected_sectors = ((uint32_t)identify[61] << 16) | identify[60];
    disk_present = detected_sectors != 0;

    serial_print("[ATA] sectors=");
    serial_print_hex(detected_sectors);
    serial_print("\n");

    return disk_present;
}

uint32_t ata_sector_count(void) {
    return detected_sectors;
}

int ata_read(uint32_t lba, uint32_t sectors, void *buffer) {
    uint8_t *out = buffer;

    if (!disk_present || sectors == 0)
        return 0;

    if (lba + sectors < lba || lba + sectors > detected_sectors)
        return 0;

    while (sectors > 0) {
        uint16_t chunk = chunk_sectors(sectors);

        if (!issue_lba_command(lba, chunk, ATA_CMD_READ))
            return 0;

        for (uint16_t i = 0; i < chunk; i++) {
            if (!wait_drq())
                return 0;
            insw(ATA_DATA, out, ATA_SECTOR_SIZE / 2);
            out += ATA_SECTOR_SIZE;
        }

        lba += chunk;
        sectors -= chunk;
    }

    return 1;
}

int ata_write(uint32_t lba, uint32_t sectors, const void *buffer) {
    const uint8_t *in = buffer;

    if (!disk_present || sectors == 0)
        return 0;

    if (lba + sectors < lba || lba + sectors > detected_sectors)
        return 0;

    while (sectors > 0) {
        uint16_t chunk = chunk_sectors(sectors);

        if (!issue_lba_command(lba, chunk, ATA_CMD_WRITE))
            return 0;

        for (uint16_t i = 0; i < chunk; i++) {
            if (!wait_drq())
                return 0;
            outsw(ATA_DATA, in, ATA_SECTOR_SIZE / 2);
            in += ATA_SECTOR_SIZE;
        }

        if (!wait_not_busy())
            return 0;
        outb(ATA_COMMAND, ATA_CMD_FLUSH);
        ata_delay_400ns();
        if (!wait_not_busy())
            return 0;

        lba += chunk;
        sectors -= chunk;
    }

    return 1;
}

static void fill_pattern(uint8_t *buffer) {
    for (uint32_t i = 0; i < ATA_SECTOR_SIZE; i++)
        buffer[i] = (uint8_t)(0xA5U ^ (i * 37U));
}

static int same_sector(const uint8_t *a, const uint8_t *b) {
    for (uint32_t i = 0; i < ATA_SECTOR_SIZE; i++) {
        if (a[i] != b[i])
            return 0;
    }

    return 1;
}

int ata_self_test(void) {
    uint8_t *write_buf;
    uint8_t *read_buf;
    uint8_t *stream_buf;
    int ok = 0;

    if (!disk_present || detected_sectors <= ATA_TEST_LBA + ATA_STREAM_SECTORS)
        return 0;

    write_buf = kmalloc(ATA_SECTOR_SIZE);
    read_buf = kmalloc(ATA_SECTOR_SIZE);
    stream_buf = kmalloc(ATA_STREAM_SECTORS * ATA_SECTOR_SIZE);

    if (write_buf == 0 || read_buf == 0 || stream_buf == 0)
        goto out;

    fill_pattern(write_buf);

    serial_print("[ATA] write test\n");
    if (!ata_write(ATA_TEST_LBA, 1, write_buf))
        goto out;

    serial_print("[ATA] readback test\n");
    if (!ata_read(ATA_TEST_LBA, 1, read_buf))
        goto out;

    if (!same_sector(write_buf, read_buf))
        goto out;

    serial_print("[ATA] stream test\n");
    if (!ata_read(0, ATA_STREAM_SECTORS, stream_buf))
        goto out;

    serial_print("[ATA] self-test passed\n");
    ok = 1;

out:
    kfree(stream_buf);
    kfree(read_buf);
    kfree(write_buf);
    return ok;
}
