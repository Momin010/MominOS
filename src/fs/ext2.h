#ifndef EXT2_H
#define EXT2_H

#include <stddef.h>
#include <stdint.h>

#define EXT2_FT_REG_FILE 1
#define EXT2_FT_DIR      2

#define EXT2_N_BLOCKS 15

struct ext2_inode {
    uint16_t i_mode;
    uint16_t i_uid;
    uint32_t i_size;
    uint32_t i_atime;
    uint32_t i_ctime;
    uint32_t i_mtime;
    uint32_t i_dtime;
    uint16_t i_gid;
    uint16_t i_links_count;
    uint32_t i_blocks;
    uint32_t i_flags;
    uint32_t i_osd1;
    uint32_t i_block[EXT2_N_BLOCKS];
    uint32_t i_generation;
    uint32_t i_file_acl;
    uint32_t i_dir_acl;
    uint32_t i_faddr;
    uint8_t i_osd2[12];
} __attribute__((packed));

typedef int (*ext2_dir_cb_t)(const char *name, uint8_t name_len, uint32_t inode, uint8_t type, void *ctx);

int ext2_mount(void);
int ext2_lookup_path(const char *path, uint32_t *inode_out, uint8_t *type_out);
int ext2_read_inode(uint32_t inode_num, struct ext2_inode *inode);
size_t ext2_read(uint32_t inode_num, uint64_t offset, void *buffer, size_t size);
int ext2_readdir(uint32_t inode_num, ext2_dir_cb_t cb, void *ctx);
uint64_t ext2_inode_size(const struct ext2_inode *inode);
uint16_t ext2_inode_mode(const struct ext2_inode *inode);

/* Create `path` as a new regular file containing `size` bytes from `data`.
   Returns inode number, or 0. Supports direct + single-indirect blocks. */
uint32_t ext2_create(const char *path, const void *data, size_t size);

/* Create `path` as an empty regular file (size 0). Fails if it exists.
   Returns the new inode number, or 0. */
uint32_t ext2_create_empty(const char *path);

/* Write `size` bytes from `data` into inode at byte `offset`, growing the
   file (allocating data + single-indirect blocks) as needed and updating
   i_size / i_blocks. Returns bytes written, or 0 on failure. */
size_t ext2_write(uint32_t inode_num, uint64_t offset, const void *data, size_t size);

/* Truncate the file to `new_size`, freeing any now-unused data and indirect
   blocks and updating bitmaps, free counts, i_size and i_blocks. Currently
   only supports truncating down to <= current size. Returns 1 on success. */
int ext2_truncate(uint32_t inode_num, uint64_t new_size);

#endif
