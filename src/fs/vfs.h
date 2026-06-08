#ifndef VFS_H
#define VFS_H

#include <stddef.h>
#include <stdint.h>

typedef struct vfs_file vfs_file_t;

struct vfs_stat {
    uint64_t size;
    uint16_t mode;
    uint8_t type;
};

typedef int (*vfs_readdir_cb_t)(const char *name, uint8_t name_len, struct vfs_stat *stat, void *ctx);

int vfs_mount_root(void);
vfs_file_t *vfs_open(const char *path);
size_t vfs_read(vfs_file_t *file, void *buffer, size_t size);
int vfs_seek(vfs_file_t *file, uint64_t offset);
void vfs_close(vfs_file_t *file);
int vfs_stat(const char *path, struct vfs_stat *stat);
int vfs_readdir(const char *path, vfs_readdir_cb_t cb, void *ctx);
int vfs_self_test(void);

#endif
