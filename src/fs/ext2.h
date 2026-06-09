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
   Direct blocks only (max 12 * block_size). Returns inode number, or 0. */
uint32_t ext2_create(const char *path, const void *data, size_t size);

#endif
