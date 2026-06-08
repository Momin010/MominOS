#include "ext2.h"
#include "ata.h"
#include "kheap.h"
#include "serial.h"

#define EXT2_SUPER_OFFSET 1024
#define EXT2_SUPER_MAGIC  0xEF53
#define EXT2_ROOT_INO     2
#define EXT2_NAME_LEN     255

struct ext2_superblock {
    uint32_t s_inodes_count;
    uint32_t s_blocks_count;
    uint32_t s_r_blocks_count;
    uint32_t s_free_blocks_count;
    uint32_t s_free_inodes_count;
    uint32_t s_first_data_block;
    uint32_t s_log_block_size;
    uint32_t s_log_frag_size;
    uint32_t s_blocks_per_group;
    uint32_t s_frags_per_group;
    uint32_t s_inodes_per_group;
    uint32_t s_mtime;
    uint32_t s_wtime;
    uint16_t s_mnt_count;
    uint16_t s_max_mnt_count;
    uint16_t s_magic;
    uint16_t s_state;
    uint16_t s_errors;
    uint16_t s_minor_rev_level;
    uint32_t s_lastcheck;
    uint32_t s_checkinterval;
    uint32_t s_creator_os;
    uint32_t s_rev_level;
    uint16_t s_def_resuid;
    uint16_t s_def_resgid;
    uint32_t s_first_ino;
    uint16_t s_inode_size;
} __attribute__((packed));

struct ext2_group_desc {
    uint32_t bg_block_bitmap;
    uint32_t bg_inode_bitmap;
    uint32_t bg_inode_table;
    uint16_t bg_free_blocks_count;
    uint16_t bg_free_inodes_count;
    uint16_t bg_used_dirs_count;
    uint16_t bg_pad;
    uint8_t bg_reserved[12];
} __attribute__((packed));

struct ext2_dir_entry {
    uint32_t inode;
    uint16_t rec_len;
    uint8_t name_len;
    uint8_t file_type;
    char name[];
} __attribute__((packed));

static struct ext2_superblock super;
static uint32_t block_size;
static uint32_t sectors_per_block;
static uint32_t group_count;
static struct ext2_group_desc *groups;
static int mounted;

static int name_eq(const char *a, const char *b, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i])
            return 0;
    }

    return b[len] == 0;
}

static int read_block(uint32_t block, void *buffer) {
    return ata_read(block * sectors_per_block, sectors_per_block, buffer);
}

static int read_bytes(uint64_t offset, void *buffer, size_t size) {
    uint8_t *out = buffer;

    while (size > 0) {
        uint64_t lba = offset / ATA_SECTOR_SIZE;
        uint32_t sector_off = offset % ATA_SECTOR_SIZE;
        uint8_t sector[ATA_SECTOR_SIZE];
        size_t chunk = ATA_SECTOR_SIZE - sector_off;

        if (chunk > size)
            chunk = size;

        if (!ata_read((uint32_t)lba, 1, sector))
            return 0;

        for (size_t i = 0; i < chunk; i++)
            out[i] = sector[sector_off + i];

        out += chunk;
        offset += chunk;
        size -= chunk;
    }

    return 1;
}

int ext2_mount(void) {
    uint32_t bgdt_block;
    uint32_t bgdt_bytes;

    mounted = 0;

    if (!read_bytes(EXT2_SUPER_OFFSET, &super, sizeof(super)))
        return 0;

    if (super.s_magic != EXT2_SUPER_MAGIC) {
        serial_print("[EXT2] bad magic\n");
        return 0;
    }

    block_size = 1024U << super.s_log_block_size;
    sectors_per_block = block_size / ATA_SECTOR_SIZE;
    group_count = (super.s_blocks_count + super.s_blocks_per_group - 1) / super.s_blocks_per_group;

    if (block_size < 1024 || sectors_per_block == 0 || group_count == 0)
        return 0;

    groups = kmalloc(group_count * sizeof(*groups));
    if (groups == 0)
        return 0;

    bgdt_block = block_size == 1024 ? 2 : 1;
    bgdt_bytes = group_count * sizeof(*groups);
    if (!read_bytes((uint64_t)bgdt_block * block_size, groups, bgdt_bytes))
        return 0;

    mounted = 1;
    serial_print("[EXT2] mounted block_size=");
    serial_print_hex(block_size);
    serial_print(" groups=");
    serial_print_hex(group_count);
    serial_print("\n");
    return 1;
}

int ext2_read_inode(uint32_t inode_num, struct ext2_inode *inode) {
    uint32_t group;
    uint32_t index;
    uint32_t block;
    uint32_t block_off;
    uint32_t inode_size = super.s_inode_size ? super.s_inode_size : 128;
    uint8_t *buf;
    int ok;

    if (!mounted || inode_num == 0)
        return 0;

    group = (inode_num - 1) / super.s_inodes_per_group;
    index = (inode_num - 1) % super.s_inodes_per_group;

    if (group >= group_count)
        return 0;

    block = groups[group].bg_inode_table + (index * inode_size) / block_size;
    block_off = (index * inode_size) % block_size;

    buf = kmalloc(block_size);
    if (buf == 0)
        return 0;

    ok = read_block(block, buf);
    if (ok) {
        uint8_t *src = buf + block_off;
        uint8_t *dst = (uint8_t *)inode;

        for (size_t i = 0; i < sizeof(*inode); i++)
            dst[i] = src[i];
    }

    kfree(buf);
    return ok;
}

static uint32_t read_indirect_ptr(uint32_t block, uint32_t index) {
    uint32_t *ptrs;
    uint32_t out = 0;

    if (block == 0)
        return 0;

    ptrs = kmalloc(block_size);
    if (ptrs == 0)
        return 0;

    if (read_block(block, ptrs))
        out = ptrs[index];

    kfree(ptrs);
    return out;
}

static uint32_t inode_block(struct ext2_inode *inode, uint32_t logical) {
    uint32_t ptrs_per_block = block_size / sizeof(uint32_t);

    if (logical < 12)
        return inode->i_block[logical];

    logical -= 12;
    if (logical < ptrs_per_block)
        return read_indirect_ptr(inode->i_block[12], logical);

    logical -= ptrs_per_block;
    if (logical < ptrs_per_block * ptrs_per_block) {
        uint32_t first = logical / ptrs_per_block;
        uint32_t second = logical % ptrs_per_block;
        uint32_t indirect = read_indirect_ptr(inode->i_block[13], first);

        return read_indirect_ptr(indirect, second);
    }

    logical -= ptrs_per_block * ptrs_per_block;
    if (logical < ptrs_per_block * ptrs_per_block * ptrs_per_block) {
        uint32_t first = logical / (ptrs_per_block * ptrs_per_block);
        uint32_t rem = logical % (ptrs_per_block * ptrs_per_block);
        uint32_t second = rem / ptrs_per_block;
        uint32_t third = rem % ptrs_per_block;
        uint32_t dbl;
        uint32_t indirect;

        dbl = read_indirect_ptr(inode->i_block[14], first);
        indirect = read_indirect_ptr(dbl, second);
        return read_indirect_ptr(indirect, third);
    }

    return 0;
}

uint64_t ext2_inode_size(const struct ext2_inode *inode) {
    return inode->i_size;
}

uint16_t ext2_inode_mode(const struct ext2_inode *inode) {
    return inode->i_mode;
}

size_t ext2_read(uint32_t inode_num, uint64_t offset, void *buffer, size_t size) {
    struct ext2_inode inode;
    uint8_t *out = buffer;
    uint8_t *block_buf;
    uint64_t file_size;
    size_t done = 0;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    file_size = ext2_inode_size(&inode);
    if (offset >= file_size)
        return 0;

    if (offset + size > file_size)
        size = file_size - offset;

    block_buf = kmalloc(block_size);
    if (block_buf == 0)
        return 0;

    while (done < size) {
        uint32_t logical = (offset + done) / block_size;
        uint32_t block_off = (offset + done) % block_size;
        uint32_t phys_block = inode_block(&inode, logical);
        size_t chunk = block_size - block_off;

        if (chunk > size - done)
            chunk = size - done;

        if (phys_block != 0) {
            if (!read_block(phys_block, block_buf))
                break;
        } else {
            for (uint32_t i = 0; i < block_size; i++)
                block_buf[i] = 0;
        }

        for (size_t i = 0; i < chunk; i++)
            out[done + i] = block_buf[block_off + i];

        done += chunk;
    }

    kfree(block_buf);
    return done;
}

int ext2_readdir(uint32_t inode_num, ext2_dir_cb_t cb, void *ctx) {
    struct ext2_inode inode;
    uint8_t *block_buf;
    uint64_t offset = 0;
    uint64_t dir_size;
    int ok = 1;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    dir_size = ext2_inode_size(&inode);
    block_buf = kmalloc(block_size);
    if (block_buf == 0)
        return 0;

    while (offset < dir_size && ok) {
        uint32_t logical = offset / block_size;
        uint32_t phys_block = inode_block(&inode, logical);
        uint32_t block_off = offset % block_size;

        if (phys_block == 0 || !read_block(phys_block, block_buf)) {
            ok = 0;
            break;
        }

        while (block_off < block_size && offset < dir_size) {
            struct ext2_dir_entry *entry = (struct ext2_dir_entry *)(block_buf + block_off);

            if (entry->rec_len == 0) {
                ok = 0;
                break;
            }

            if (entry->inode != 0) {
                if (!cb(entry->name, entry->name_len, entry->inode, entry->file_type, ctx)) {
                    ok = 0;
                    break;
                }
            }

            block_off += entry->rec_len;
            offset += entry->rec_len;
        }
    }

    kfree(block_buf);
    return ok;
}

struct lookup_ctx {
    const char *name;
    size_t len;
    uint32_t inode;
    uint8_t type;
    int found;
};

static int lookup_cb(const char *name, uint8_t name_len, uint32_t inode, uint8_t type, void *ctx) {
    struct lookup_ctx *lookup = ctx;

    if (name_len == lookup->len && name_eq(name, lookup->name, lookup->len)) {
        lookup->inode = inode;
        lookup->type = type;
        lookup->found = 1;
        return 0;
    }

    return 1;
}

static int lookup_child(uint32_t dir_inode, const char *name, size_t len, uint32_t *inode_out, uint8_t *type_out) {
    struct lookup_ctx ctx = {
        .name = name,
        .len = len,
        .inode = 0,
        .type = 0,
        .found = 0,
    };

    ext2_readdir(dir_inode, lookup_cb, &ctx);
    if (!ctx.found)
        return 0;

    *inode_out = ctx.inode;
    if (type_out != 0)
        *type_out = ctx.type;
    return 1;
}

int ext2_lookup_path(const char *path, uint32_t *inode_out, uint8_t *type_out) {
    uint32_t current = EXT2_ROOT_INO;
    uint8_t type = EXT2_FT_DIR;
    size_t pos = 0;

    if (!mounted || path == 0 || path[0] != '/')
        return 0;

    while (path[pos] == '/')
        pos++;

    if (path[pos] == 0) {
        *inode_out = current;
        if (type_out != 0)
            *type_out = type;
        return 1;
    }

    while (path[pos] != 0) {
        size_t start = pos;
        size_t len;

        while (path[pos] != 0 && path[pos] != '/')
            pos++;
        len = pos - start;

        if (!lookup_child(current, path + start, len, &current, &type))
            return 0;

        while (path[pos] == '/')
            pos++;
    }

    *inode_out = current;
    if (type_out != 0)
        *type_out = type;
    return 1;
}
