#include "ata.h"
#include "kheap.h"
#include "sched.h"
#include "serial.h"

/* Register offsets relative to a channel's I/O base (0x1F0 primary,
   0x170 secondary). The control/alt-status register lives on a separate
   control base (0x3F6 primary, 0x376 secondary). */
#define ATA_REG_DATA     0
#define ATA_REG_ERROR    1
#define ATA_REG_FEATURES 1
#define ATA_REG_SECCOUNT 2
#define ATA_REG_LBA0     3
#define ATA_REG_LBA1     4
#define ATA_REG_LBA2     5
#define ATA_REG_DRIVE    6
#define ATA_REG_STATUS   7
#define ATA_REG_COMMAND  7

#define ATA_PRIMARY_IO    0x1F0
#define ATA_PRIMARY_CTRL  0x3F6
#define ATA_SECONDARY_IO  0x170
#define ATA_SECONDARY_CTRL 0x376

#define ATA_SELECT_MASTER 0xE0
#define ATA_SELECT_SLAVE  0xF0

#define ATA_MAX_DRIVES 4

#define ATA_SR_ERR 0x01
#define ATA_SR_DRQ 0x08
#define ATA_SR_DF  0x20
#define ATA_SR_BSY 0x80

#define ATA_CMD_READ     0x20
#define ATA_CMD_WRITE    0x30
#define ATA_CMD_IDENTIFY 0xEC
#define ATA_CMD_FLUSH    0xE7

/* Spin-poll the status register tightly: on emulated (and real) hardware the
   drive clears BSY / raises DRQ within microseconds, so a tight inb() loop
   returns almost instantly. Only after an enormous spin count -- a sign the
   device is genuinely stalled -- do we start yielding the CPU. Yielding early
   was catastrophic: syscalls run with interrupts off, so a yield handed the
   CPU to the idle thread which slept until the next 10ms timer tick, turning
   every sector wait into a 10ms stall (a single binary load cost ~1.8s). */
#define ATA_POLL_LIMIT        10000000ULL
#define ATA_SPIN_BEFORE_YIELD 100000ULL
#define ATA_YIELD_EVERY       100000ULL
#define ATA_TEST_LBA 2048U
#define ATA_STREAM_SECTORS 4096U

struct ata_drive {
    uint16_t io_base;    /* command-block base (0 = absent) */
    uint16_t ctrl_base;  /* control-block base (alt status) */
    uint8_t select;      /* 0xE0 master / 0xF0 slave */
    int present;
    uint32_t sectors;
};

static struct ata_drive drives[ATA_MAX_DRIVES];

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

static void ata_delay_400ns(const struct ata_drive *d) {
    inb(d->ctrl_base);
    inb(d->ctrl_base);
    inb(d->ctrl_base);
    inb(d->ctrl_base);
}

static int wait_not_busy(const struct ata_drive *d) {
    uint16_t status_port = d->io_base + ATA_REG_STATUS;

    for (uint64_t i = 0; i < ATA_POLL_LIMIT; i++) {
        uint8_t status = inb(status_port);

        if (status == 0xFF)
            return 0;

        if ((status & ATA_SR_BSY) == 0)
            return 1;

        if (i >= ATA_SPIN_BEFORE_YIELD && (i % ATA_YIELD_EVERY) == 0)
            sched_yield();
    }

    return 0;
}

static int wait_drq(const struct ata_drive *d) {
    uint16_t status_port = d->io_base + ATA_REG_STATUS;

    for (uint64_t i = 0; i < ATA_POLL_LIMIT; i++) {
        uint8_t status = inb(status_port);

        if (status == 0xFF)
            return 0;

        if (status & (ATA_SR_ERR | ATA_SR_DF))
            return 0;

        if ((status & ATA_SR_BSY) == 0 && (status & ATA_SR_DRQ))
            return 1;

        if (i >= ATA_SPIN_BEFORE_YIELD && (i % ATA_YIELD_EVERY) == 0)
            sched_yield();
    }

    return 0;
}

static void select_lba(const struct ata_drive *d, uint32_t lba) {
    outb(d->io_base + ATA_REG_DRIVE, d->select | ((lba >> 24) & 0x0F));
    ata_delay_400ns(d);
}

static int issue_lba_command(const struct ata_drive *d, uint32_t lba,
                             uint16_t sectors, uint8_t command) {
    uint8_t count = sectors == 256 ? 0 : (uint8_t)sectors;

    if (!wait_not_busy(d))
        return 0;

    select_lba(d, lba);
    outb(d->io_base + ATA_REG_FEATURES, 0);
    outb(d->io_base + ATA_REG_SECCOUNT, count);
    outb(d->io_base + ATA_REG_LBA0, lba & 0xFF);
    outb(d->io_base + ATA_REG_LBA1, (lba >> 8) & 0xFF);
    outb(d->io_base + ATA_REG_LBA2, (lba >> 16) & 0xFF);
    outb(d->io_base + ATA_REG_COMMAND, command);
    ata_delay_400ns(d);

    return 1;
}

static uint16_t chunk_sectors(uint32_t sectors) {
    if (sectors > 128)
        return 128;
    return (uint16_t)sectors;
}

/* Probe one of the four standard ATA positions with IDENTIFY. Fills in the
   drive entry (present + sector count) on success. */
static void ata_probe(int index, uint16_t io_base, uint16_t ctrl_base,
                      uint8_t select) {
    struct ata_drive *d = &drives[index];
    uint16_t identify[256];
    uint8_t status;

    d->io_base = io_base;
    d->ctrl_base = ctrl_base;
    d->select = select;
    d->present = 0;
    d->sectors = 0;

    /* Disable interrupts on this channel and pick the drive. */
    outb(ctrl_base, 0);
    outb(io_base + ATA_REG_DRIVE, select);
    ata_delay_400ns(d);

    if (!wait_not_busy(d))
        return;

    /* Issue IDENTIFY with the LBA registers zeroed. */
    outb(io_base + ATA_REG_SECCOUNT, 0);
    outb(io_base + ATA_REG_LBA0, 0);
    outb(io_base + ATA_REG_LBA1, 0);
    outb(io_base + ATA_REG_LBA2, 0);
    outb(io_base + ATA_REG_COMMAND, ATA_CMD_IDENTIFY);
    ata_delay_400ns(d);

    status = inb(io_base + ATA_REG_STATUS);
    if (status == 0 || status == 0xFF)
        return; /* floating bus / no drive */

    if (!wait_drq(d))
        return;

    /* Reject non-ATA (e.g. ATAPI) devices: a real ATA disk leaves the
       LBA-mid/high signature registers at zero after IDENTIFY. */
    if (inb(io_base + ATA_REG_LBA1) != 0 || inb(io_base + ATA_REG_LBA2) != 0)
        return;

    insw(io_base + ATA_REG_DATA, identify, 256);

    d->sectors = ((uint32_t)identify[61] << 16) | identify[60];
    d->present = d->sectors != 0;

    if (d->present) {
        serial_print("[ATA] drive ");
        serial_print_hex((uint32_t)index);
        serial_print(" sectors=");
        serial_print_hex(d->sectors);
        serial_print("\n");
    }
}

int ata_init(void) {
    for (int i = 0; i < ATA_MAX_DRIVES; i++) {
        drives[i].present = 0;
        drives[i].sectors = 0;
    }

    ata_probe(0, ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, ATA_SELECT_MASTER);
    ata_probe(1, ATA_PRIMARY_IO, ATA_PRIMARY_CTRL, ATA_SELECT_SLAVE);
    ata_probe(2, ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, ATA_SELECT_MASTER);
    ata_probe(3, ATA_SECONDARY_IO, ATA_SECONDARY_CTRL, ATA_SELECT_SLAVE);

    if (!drives[0].present)
        serial_print("[ATA] no primary disk\n");

    /* Non-destructive read of any additional drives so a second attached
       disk is shown to be usable at boot. */
    ata_probe_extra_drives();

    /* Contract preserved: nonzero iff the primary master is usable (kmain
       gates the ext2 root mount on this). */
    return drives[0].present;
}

int ata_drive_present(int drive) {
    if (drive < 0 || drive >= ATA_MAX_DRIVES)
        return 0;
    return drives[drive].present;
}

uint32_t ata_drive_sector_count(int drive) {
    if (drive < 0 || drive >= ATA_MAX_DRIVES)
        return 0;
    return drives[drive].sectors;
}

uint32_t ata_sector_count(void) {
    return drives[0].sectors;
}

int ata_read_drive(int drive, uint32_t lba, uint32_t sectors, void *buffer) {
    struct ata_drive *d;
    uint8_t *out = buffer;

    if (drive < 0 || drive >= ATA_MAX_DRIVES)
        return 0;

    d = &drives[drive];
    if (!d->present || sectors == 0)
        return 0;

    if (lba + sectors < lba || lba + sectors > d->sectors)
        return 0;

    while (sectors > 0) {
        uint16_t chunk = chunk_sectors(sectors);

        if (!issue_lba_command(d, lba, chunk, ATA_CMD_READ))
            return 0;

        for (uint16_t i = 0; i < chunk; i++) {
            if (!wait_drq(d))
                return 0;
            insw(d->io_base + ATA_REG_DATA, out, ATA_SECTOR_SIZE / 2);
            out += ATA_SECTOR_SIZE;
        }

        lba += chunk;
        sectors -= chunk;
    }

    return 1;
}

int ata_write_drive(int drive, uint32_t lba, uint32_t sectors,
                     const void *buffer) {
    struct ata_drive *d;
    const uint8_t *in = buffer;

    if (drive < 0 || drive >= ATA_MAX_DRIVES)
        return 0;

    d = &drives[drive];
    if (!d->present || sectors == 0)
        return 0;

    if (lba + sectors < lba || lba + sectors > d->sectors)
        return 0;

    while (sectors > 0) {
        uint16_t chunk = chunk_sectors(sectors);

        if (!issue_lba_command(d, lba, chunk, ATA_CMD_WRITE))
            return 0;

        for (uint16_t i = 0; i < chunk; i++) {
            if (!wait_drq(d))
                return 0;
            outsw(d->io_base + ATA_REG_DATA, in, ATA_SECTOR_SIZE / 2);
            in += ATA_SECTOR_SIZE;
        }

        if (!wait_not_busy(d))
            return 0;
        outb(d->io_base + ATA_REG_COMMAND, ATA_CMD_FLUSH);
        ata_delay_400ns(d);
        if (!wait_not_busy(d))
            return 0;

        lba += chunk;
        sectors -= chunk;
    }

    return 1;
}

/* Backward-compatible primary-master (drive 0) wrappers used by the VFS. */
int ata_read(uint32_t lba, uint32_t sectors, void *buffer) {
    return ata_read_drive(0, lba, sectors, buffer);
}

int ata_write(uint32_t lba, uint32_t sectors, const void *buffer) {
    return ata_write_drive(0, lba, sectors, buffer);
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

    if (!drives[0].present ||
        drives[0].sectors <= ATA_TEST_LBA + ATA_STREAM_SECTORS)
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

/* Read sector 0 from every detected non-root drive and print the first few
   bytes. Non-destructive: proves additional drives are addressable. Skipped
   silently if no extra drive is present. */
void ata_probe_extra_drives(void) {
    uint8_t *buf;
    int any = 0;

    for (int i = 1; i < ATA_MAX_DRIVES; i++)
        any |= drives[i].present;
    if (!any)
        return;

    buf = kmalloc(ATA_SECTOR_SIZE);
    if (buf == 0)
        return;

    for (int i = 1; i < ATA_MAX_DRIVES; i++) {
        if (!drives[i].present)
            continue;

        if (!ata_read_drive(i, 0, 1, buf)) {
            serial_print("[ATA] drive ");
            serial_print_hex((uint32_t)i);
            serial_print(" sector0 read failed\n");
            continue;
        }

        serial_print("[ATA] drive ");
        serial_print_hex((uint32_t)i);
        serial_print(" sector0 bytes=");
        for (int j = 0; j < 8; j++) {
            serial_print_hex(buf[j]);
            serial_print(" ");
        }
        serial_print("\n");
    }

    kfree(buf);
}
