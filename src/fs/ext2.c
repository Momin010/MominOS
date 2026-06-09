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
    /* b may point into the middle of a path string (e.g. "bin/sh"), so do not
       test for a trailing NUL here. The caller already compared lengths. */
    for (size_t i = 0; i < len; i++) {
        if (a[i] != b[i])
            return 0;
    }

    return 1;
}

static int read_block(uint32_t block, void *buffer) {
    return ata_read(block * sectors_per_block, sectors_per_block, buffer);
}

static int write_block(uint32_t block, const void *buffer) {
    return ata_write(block * sectors_per_block, sectors_per_block, buffer);
}

/* Read-modify-write at byte granularity: read the affected 512B sector(s),
   overlay the new bytes, and write them back. This avoids clobbering
   surrounding on-disk data when a structure (superblock, inode, group desc)
   is smaller than a sector or our in-memory copy is truncated. */
static int write_bytes(uint64_t offset, const void *buffer, size_t size) {
    const uint8_t *in = buffer;

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
            sector[sector_off + i] = in[i];

        if (!ata_write((uint32_t)lba, 1, sector))
            return 0;

        in += chunk;
        offset += chunk;
        size -= chunk;
    }

    return 1;
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

/* ------------------------------------------------------------------ */
/* Write path                                                          */
/*                                                                     */
/* Supports: create empty file, open-existing, write/append at any     */
/* offset (partial-block read-modify-write), overwrite via truncate,   */
/* and growth through DIRECT blocks (i_block[0..11]) plus the SINGLE-   */
/* INDIRECT block (i_block[12]). Maximum file size is therefore        */
/*   (12 + block_size/4) * block_size                                  */
/* = (12 + 1024) * 4096 = 1036 * 4096 ~= 4.04 MB at 4 KB blocks.       */
/* Double/triple-indirect WRITES are not implemented (reads of them    */
/* already work via inode_block, used by the 5 MB big.bin self-test).  */
/* Directory entries are added into the parent's existing last data    */
/* block (true for a fresh mke2fs root). Allocation helpers scan every */
/* group, so multi-group images also work. i_size and i_blocks (the    */
/* latter in 512-byte sectors, counting indirect blocks too) are kept  */
/* consistent so e2fsck -fn stays clean.                               */
/* ------------------------------------------------------------------ */

#define EXT2_DIRECT_BLOCKS 12

/* On-disk field byte offsets (independent of our possibly-truncated structs) */
#define SB_FREE_BLOCKS_OFF 12
#define SB_FREE_INODES_OFF 16
#define BG_DESC_SIZE       32
#define BG_FREE_BLOCKS_OFF 12
#define BG_FREE_INODES_OFF 14
#define BG_USED_DIRS_OFF   16

static uint64_t inode_byte_offset(uint32_t inode_num) {
    uint32_t inode_size = super.s_inode_size ? super.s_inode_size : 128;
    uint32_t group = (inode_num - 1) / super.s_inodes_per_group;
    uint32_t index = (inode_num - 1) % super.s_inodes_per_group;

    return (uint64_t)groups[group].bg_inode_table * block_size +
           (uint64_t)index * inode_size;
}

static int write_inode(uint32_t inode_num, const struct ext2_inode *inode) {
    /* RMW writes only the 128 bytes of our struct; trailing inode bytes
       (image uses 256-byte inodes) are left untouched. */
    return write_bytes(inode_byte_offset(inode_num), inode, sizeof(*inode));
}

/* Persist the on-disk group descriptor counts for one group. The group
   descriptor table starts at block 1 (block_size != 1024) or block 2. */
static int flush_group_desc(uint32_t group) {
    uint32_t bgdt_block = block_size == 1024 ? 2 : 1;
    uint64_t base = (uint64_t)bgdt_block * block_size + (uint64_t)group * BG_DESC_SIZE;
    struct ext2_group_desc *g = &groups[group];

    if (!write_bytes(base + BG_FREE_BLOCKS_OFF, &g->bg_free_blocks_count, 2))
        return 0;
    if (!write_bytes(base + BG_FREE_INODES_OFF, &g->bg_free_inodes_count, 2))
        return 0;
    if (!write_bytes(base + BG_USED_DIRS_OFF, &g->bg_used_dirs_count, 2))
        return 0;
    return 1;
}

static int flush_superblock_counts(void) {
    if (!write_bytes(EXT2_SUPER_OFFSET + SB_FREE_BLOCKS_OFF, &super.s_free_blocks_count, 4))
        return 0;
    if (!write_bytes(EXT2_SUPER_OFFSET + SB_FREE_INODES_OFF, &super.s_free_inodes_count, 4))
        return 0;
    return 1;
}

/* Scan a bitmap block for a free bit, set it, write it back. Returns the
   bit index within the group (0-based), or 0xFFFFFFFF on failure. */
static uint32_t alloc_from_bitmap(uint32_t bitmap_block, uint32_t max_bits) {
    uint8_t *buf = kmalloc(block_size);
    uint32_t result = 0xFFFFFFFF;

    if (buf == 0)
        return result;

    if (!read_block(bitmap_block, buf))
        goto out;

    for (uint32_t bit = 0; bit < max_bits; bit++) {
        uint32_t byte = bit / 8;
        uint8_t mask = (uint8_t)(1u << (bit % 8));

        if ((buf[byte] & mask) == 0) {
            buf[byte] |= mask;
            if (!write_block(bitmap_block, buf))
                goto out;
            result = bit;
            goto out;
        }
    }

out:
    kfree(buf);
    return result;
}

/* Allocate one data block. Returns absolute block number, or 0 on failure. */
static uint32_t alloc_block(void) {
    for (uint32_t grp = 0; grp < group_count; grp++) {
        uint32_t remaining = super.s_blocks_per_group;
        uint32_t group_base;
        uint32_t bit;

        if (groups[grp].bg_free_blocks_count == 0)
            continue;

        /* clamp to the real number of blocks in the last group */
        group_base = super.s_first_data_block + grp * super.s_blocks_per_group;
        if (group_base + remaining > super.s_blocks_count)
            remaining = super.s_blocks_count - group_base;

        bit = alloc_from_bitmap(groups[grp].bg_block_bitmap, remaining);
        if (bit == 0xFFFFFFFF)
            continue;

        groups[grp].bg_free_blocks_count--;
        super.s_free_blocks_count--;
        flush_group_desc(grp);
        flush_superblock_counts();
        return group_base + bit;
    }

    return 0;
}

/* Allocate one inode. Returns inode number (1-based), or 0 on failure.
   is_dir controls the used_dirs accounting (we only create regular files,
   so it is always 0 here, but kept for correctness). */
static uint32_t alloc_inode(int is_dir) {
    for (uint32_t grp = 0; grp < group_count; grp++) {
        uint32_t bit;

        if (groups[grp].bg_free_inodes_count == 0)
            continue;

        bit = alloc_from_bitmap(groups[grp].bg_inode_bitmap, super.s_inodes_per_group);
        if (bit == 0xFFFFFFFF)
            continue;

        groups[grp].bg_free_inodes_count--;
        super.s_free_inodes_count--;
        if (is_dir)
            groups[grp].bg_used_dirs_count++;
        flush_group_desc(grp);
        flush_superblock_counts();
        return grp * super.s_inodes_per_group + bit + 1;
    }

    return 0;
}

/* Clear one bit in a bitmap block (mark free). Returns 1 on success. */
static int free_in_bitmap(uint32_t bitmap_block, uint32_t bit) {
    uint8_t *buf = kmalloc(block_size);
    int ok = 0;

    if (buf == 0)
        return 0;

    if (!read_block(bitmap_block, buf))
        goto out;

    buf[bit / 8] &= (uint8_t)~(1u << (bit % 8));
    if (write_block(bitmap_block, buf))
        ok = 1;

out:
    kfree(buf);
    return ok;
}

/* Free one data block (absolute number): clear its bitmap bit and bump the
   group + superblock free-block counts. */
static int free_block(uint32_t block) {
    uint32_t grp;
    uint32_t group_base;
    uint32_t bit;

    if (block < super.s_first_data_block || block >= super.s_blocks_count)
        return 0;

    grp = (block - super.s_first_data_block) / super.s_blocks_per_group;
    group_base = super.s_first_data_block + grp * super.s_blocks_per_group;
    bit = block - group_base;

    if (!free_in_bitmap(groups[grp].bg_block_bitmap, bit))
        return 0;

    groups[grp].bg_free_blocks_count++;
    super.s_free_blocks_count++;
    flush_group_desc(grp);
    flush_superblock_counts();
    return 1;
}

/* Allocate one data block and zero its contents on disk. Returns absolute
   block number, or 0 on failure. Used for indirect blocks (must be zeroed so
   unused pointer slots are 0) and for new data blocks. */
static uint32_t alloc_zeroed_block(void) {
    uint32_t blk = alloc_block();
    uint8_t *buf;

    if (blk == 0)
        return 0;

    buf = kmalloc(block_size);
    if (buf == 0) {
        free_block(blk);
        return 0;
    }
    for (uint32_t i = 0; i < block_size; i++)
        buf[i] = 0;
    if (!write_block(blk, buf)) {
        kfree(buf);
        free_block(blk);
        return 0;
    }
    kfree(buf);
    return blk;
}

/* Map a logical block index to a physical block for WRITING, allocating the
   data block (and the single-indirect block if needed) when missing.
   *i_blocks_sectors is incremented (in 512B units) for each newly allocated
   block (data or indirect). Supports direct (0..11) + single-indirect
   (12 .. 11 + ptrs_per_block). Returns physical block, or 0 on failure. */
static uint32_t inode_block_alloc(struct ext2_inode *inode, uint32_t logical,
                                  uint32_t *i_blocks_sectors) {
    uint32_t ptrs_per_block = block_size / sizeof(uint32_t);
    uint32_t sectors_per = block_size / 512;

    if (logical < EXT2_DIRECT_BLOCKS) {
        if (inode->i_block[logical] == 0) {
            uint32_t blk = alloc_block();
            if (blk == 0)
                return 0;
            inode->i_block[logical] = blk;
            *i_blocks_sectors += sectors_per;
        }
        return inode->i_block[logical];
    }

    logical -= EXT2_DIRECT_BLOCKS;
    if (logical < ptrs_per_block) {
        uint32_t *ptrs;
        uint32_t indirect = inode->i_block[12];
        uint32_t result = 0;

        if (indirect == 0) {
            indirect = alloc_zeroed_block();
            if (indirect == 0)
                return 0;
            inode->i_block[12] = indirect;
            *i_blocks_sectors += sectors_per;
        }

        ptrs = kmalloc(block_size);
        if (ptrs == 0)
            return 0;
        if (!read_block(indirect, ptrs)) {
            kfree(ptrs);
            return 0;
        }

        if (ptrs[logical] == 0) {
            uint32_t blk = alloc_block();
            if (blk == 0) {
                kfree(ptrs);
                return 0;
            }
            ptrs[logical] = blk;
            *i_blocks_sectors += sectors_per;
            if (!write_block(indirect, ptrs)) {
                kfree(ptrs);
                return 0;
            }
        }
        result = ptrs[logical];
        kfree(ptrs);
        return result;
    }

    return 0;       /* beyond single-indirect: not supported for writes */
}

/* Write `size` bytes from `data` into inode at byte `offset`, growing the
   file. Read-modify-writes partial blocks so appends mid-block are safe. */
size_t ext2_write(uint32_t inode_num, uint64_t offset, const void *data, size_t size) {
    struct ext2_inode inode;
    const uint8_t *src = data;
    uint8_t *blkbuf;
    uint32_t i_blocks = 0;
    uint64_t end;
    size_t done = 0;

    if (!mounted || data == 0)
        return 0;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    i_blocks = inode.i_blocks;

    blkbuf = kmalloc(block_size);
    if (blkbuf == 0)
        return 0;

    while (done < size) {
        uint32_t logical = (uint32_t)((offset + done) / block_size);
        uint32_t block_off = (uint32_t)((offset + done) % block_size);
        size_t chunk = block_size - block_off;
        uint32_t phys;

        if (chunk > size - done)
            chunk = size - done;

        phys = inode_block_alloc(&inode, logical, &i_blocks);
        if (phys == 0)
            break;

        /* read-modify-write the block to preserve surrounding bytes */
        if (!read_block(phys, blkbuf))
            break;
        for (size_t b = 0; b < chunk; b++)
            blkbuf[block_off + b] = src[done + b];
        if (!write_block(phys, blkbuf))
            break;

        done += chunk;
    }

    kfree(blkbuf);

    end = offset + done;
    if (end > inode.i_size)
        inode.i_size = (uint32_t)end;
    inode.i_blocks = i_blocks;

    if (!write_inode(inode_num, &inode))
        return 0;

    return done;
}

/* Truncate file down to new_size, freeing data + indirect blocks no longer
   needed. Only supports new_size <= current size. */
int ext2_truncate(uint32_t inode_num, uint64_t new_size) {
    struct ext2_inode inode;
    uint32_t ptrs_per_block;
    uint32_t sectors_per;
    uint32_t i_blocks;
    uint32_t keep_blocks;
    uint32_t total_blocks;

    if (!mounted)
        return 0;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    if (new_size > inode.i_size)
        return 0;       /* grow handled by ext2_write */

    ptrs_per_block = block_size / sizeof(uint32_t);
    sectors_per = block_size / 512;
    i_blocks = inode.i_blocks;

    keep_blocks = (uint32_t)((new_size + block_size - 1) / block_size);
    total_blocks = (uint32_t)((inode.i_size + block_size - 1) / block_size);

    /* free direct blocks beyond keep_blocks */
    for (uint32_t l = keep_blocks; l < total_blocks && l < EXT2_DIRECT_BLOCKS; l++) {
        if (inode.i_block[l] != 0) {
            free_block(inode.i_block[l]);
            inode.i_block[l] = 0;
            i_blocks -= sectors_per;
        }
    }

    /* free single-indirect data blocks beyond keep_blocks */
    if (inode.i_block[12] != 0) {
        uint32_t *ptrs = kmalloc(block_size);
        int touched = 0;
        int any_left = 0;

        if (ptrs == 0)
            return 0;
        if (!read_block(inode.i_block[12], ptrs)) {
            kfree(ptrs);
            return 0;
        }

        for (uint32_t i = 0; i < ptrs_per_block; i++) {
            uint32_t logical = EXT2_DIRECT_BLOCKS + i;

            if (ptrs[i] == 0)
                continue;
            if (logical >= keep_blocks) {
                free_block(ptrs[i]);
                ptrs[i] = 0;
                i_blocks -= sectors_per;
                touched = 1;
            } else {
                any_left = 1;
            }
        }

        if (any_left) {
            if (touched && !write_block(inode.i_block[12], ptrs)) {
                kfree(ptrs);
                return 0;
            }
        } else {
            /* the whole indirect block is now empty: free it too */
            free_block(inode.i_block[12]);
            inode.i_block[12] = 0;
            i_blocks -= sectors_per;
        }
        kfree(ptrs);
    }

    inode.i_size = (uint32_t)new_size;
    inode.i_blocks = i_blocks;
    return write_inode(inode_num, &inode);
}

static uint32_t dirent_min_len(uint8_t name_len) {
    /* 8-byte header + name, rounded up to 4 bytes */
    return (8 + name_len + 3) & ~3u;
}

/* Insert a directory entry for (name -> inode) into dir_inode's last data
   block by splitting the trailing entry's slack. Returns 1 on success, 0 if
   there is no room (v1 limit) or on I/O error. */
static int dir_add_entry(uint32_t dir_inode_num, const char *name, uint8_t name_len,
                         uint32_t child_inode, uint8_t file_type) {
    struct ext2_inode dir;
    uint8_t *buf;
    uint32_t logical;
    uint32_t phys_block;
    uint32_t need = dirent_min_len(name_len);
    uint32_t off;
    int ok = 0;

    if (!ext2_read_inode(dir_inode_num, &dir))
        return 0;

    if (dir.i_size == 0 || (dir.i_size % block_size) != 0)
        return 0;

    /* operate on the last data block of the directory */
    logical = (dir.i_size / block_size) - 1;
    phys_block = inode_block(&dir, logical);
    if (phys_block == 0)
        return 0;

    buf = kmalloc(block_size);
    if (buf == 0)
        return 0;

    if (!read_block(phys_block, buf))
        goto out;

    /* walk to the last entry in the block */
    off = 0;
    while (off < block_size) {
        struct ext2_dir_entry *e = (struct ext2_dir_entry *)(buf + off);
        uint32_t rec_len = e->rec_len;
        uint32_t used;
        uint32_t slack;

        if (rec_len == 0)
            break;

        /* slack available after shrinking this entry to its minimum */
        used = (e->inode != 0) ? dirent_min_len(e->name_len) : 0;
        slack = (off + rec_len <= block_size) ? (rec_len - used) : 0;

        if (off + rec_len >= block_size) {
            /* this is the last entry in the block; try to split its slack */
            if (slack >= need) {
                struct ext2_dir_entry *ne;

                if (used == 0) {
                    /* empty slot spanning to block end: reuse in place */
                    e->inode = child_inode;
                    e->name_len = name_len;
                    e->file_type = file_type;
                    for (uint8_t i = 0; i < name_len; i++)
                        e->name[i] = name[i];
                    /* rec_len already runs to block end: keep it */
                } else {
                    e->rec_len = (uint16_t)used;
                    ne = (struct ext2_dir_entry *)(buf + off + used);
                    ne->inode = child_inode;
                    ne->rec_len = (uint16_t)(rec_len - used);
                    ne->name_len = name_len;
                    ne->file_type = file_type;
                    for (uint8_t i = 0; i < name_len; i++)
                        ne->name[i] = name[i];
                }

                if (write_block(phys_block, buf))
                    ok = 1;
            }
            break;
        }

        off += rec_len;
    }

out:
    kfree(buf);
    return ok;
}

/* Create a new empty regular file `name` in directory dir_inode_num. Returns
   the new inode number, or 0. Caller writes content via ext2_write. */
static uint32_t ext2_create_file(uint32_t dir_inode_num, const char *name, uint8_t name_len) {
    struct ext2_inode inode;
    uint32_t inode_num;

    if (!mounted)
        return 0;

    inode_num = alloc_inode(0);
    if (inode_num == 0)
        return 0;

    /* zero the inode, then fill regular-file metadata */
    for (size_t b = 0; b < sizeof(inode); b++)
        ((uint8_t *)&inode)[b] = 0;

    inode.i_mode = 0x8000 | 0644;       /* regular file, rw-r--r-- */
    inode.i_links_count = 1;
    inode.i_size = 0;
    inode.i_blocks = 0;

    if (!write_inode(inode_num, &inode))
        return 0;

    if (!dir_add_entry(dir_inode_num, name, name_len, inode_num, EXT2_FT_REG_FILE))
        return 0;

    return inode_num;
}

/* Split an absolute path into parent directory inode and final component.
   Returns 1 on success. name_out points into `path`. */
static int split_parent(const char *path, uint32_t *parent_out,
                        const char **name_out, size_t *name_len_out) {
    size_t len = 0;
    size_t last_slash = 0;
    int have_slash = 0;
    char parent_path[256];
    uint32_t parent_inode;
    uint8_t ptype;

    while (path[len])
        len++;
    if (len == 0 || path[0] != '/')
        return 0;
    if (path[len - 1] == '/')
        return 0;       /* no trailing-slash directories in v1 */

    for (size_t i = 0; i < len; i++) {
        if (path[i] == '/') {
            last_slash = i;
            have_slash = 1;
        }
    }
    if (!have_slash)
        return 0;

    if (last_slash == 0) {
        parent_path[0] = '/';
        parent_path[1] = 0;
    } else {
        if (last_slash >= sizeof(parent_path))
            return 0;
        for (size_t i = 0; i < last_slash; i++)
            parent_path[i] = path[i];
        parent_path[last_slash] = 0;
    }

    if (!ext2_lookup_path(parent_path, &parent_inode, &ptype))
        return 0;
    if (ptype != EXT2_FT_DIR)
        return 0;

    *parent_out = parent_inode;
    *name_out = path + last_slash + 1;
    *name_len_out = len - last_slash - 1;
    return 1;
}

/* Public: create `path` as a new empty regular file. Fails if it already
   exists. Returns the new inode number, or 0. */
uint32_t ext2_create_empty(const char *path) {
    uint32_t parent;
    const char *name;
    size_t name_len;
    uint32_t existing;
    uint8_t etype;

    if (!mounted)
        return 0;

    if (ext2_lookup_path(path, &existing, &etype))
        return 0;       /* already exists */

    if (!split_parent(path, &parent, &name, &name_len))
        return 0;

    if (name_len == 0 || name_len > 255)
        return 0;

    return ext2_create_file(parent, name, (uint8_t)name_len);
}

/* Public: create `path` as a new regular file containing `size` bytes.
   Fails if the file already exists. Returns the inode number, or 0. */
uint32_t ext2_create(const char *path, const void *data, size_t size) {
    uint32_t inode_num = ext2_create_empty(path);

    if (inode_num == 0)
        return 0;

    if (size > 0 && ext2_write(inode_num, 0, data, size) != size)
        return 0;

    return inode_num;
}
