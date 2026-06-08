#include "vfs.h"
#include "ext2.h"
#include "kheap.h"
#include "serial.h"

struct vfs_file {
    uint32_t inode;
    uint64_t offset;
    uint64_t size;
    uint8_t type;
};

static int mounted;

static int str_eq_literal(const char *buf, size_t len, const char *lit) {
    for (size_t i = 0; i < len; i++) {
        if (lit[i] == 0 || buf[i] != lit[i])
            return 0;
    }

    return lit[len] == 0;
}

int vfs_mount_root(void) {
    mounted = ext2_mount();
    return mounted;
}

vfs_file_t *vfs_open(const char *path) {
    uint32_t inode_num;
    uint8_t type;
    struct ext2_inode inode;
    vfs_file_t *file;

    if (!mounted || !ext2_lookup_path(path, &inode_num, &type))
        return 0;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    file = kmalloc(sizeof(*file));
    if (file == 0)
        return 0;

    file->inode = inode_num;
    file->offset = 0;
    file->size = ext2_inode_size(&inode);
    file->type = type;
    return file;
}

size_t vfs_read(vfs_file_t *file, void *buffer, size_t size) {
    size_t read;

    if (file == 0 || buffer == 0)
        return 0;

    read = ext2_read(file->inode, file->offset, buffer, size);
    file->offset += read;
    return read;
}

int vfs_seek(vfs_file_t *file, uint64_t offset) {
    if (file == 0 || offset > file->size)
        return 0;

    file->offset = offset;
    return 1;
}

void vfs_close(vfs_file_t *file) {
    kfree(file);
}

int vfs_stat(const char *path, struct vfs_stat *stat) {
    uint32_t inode_num;
    uint8_t type;
    struct ext2_inode inode;

    if (stat == 0 || !mounted || !ext2_lookup_path(path, &inode_num, &type))
        return 0;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    stat->size = ext2_inode_size(&inode);
    stat->mode = ext2_inode_mode(&inode);
    stat->type = type;
    return 1;
}

struct vfs_readdir_ctx {
    vfs_readdir_cb_t cb;
    void *ctx;
};

static int vfs_dir_cb(const char *name, uint8_t name_len, uint32_t inode_num, uint8_t type, void *ctx) {
    struct vfs_readdir_ctx *vctx = ctx;
    struct ext2_inode inode;
    struct vfs_stat stat;

    if (!ext2_read_inode(inode_num, &inode))
        return 0;

    stat.size = ext2_inode_size(&inode);
    stat.mode = ext2_inode_mode(&inode);
    stat.type = type;

    return vctx->cb(name, name_len, &stat, vctx->ctx);
}

int vfs_readdir(const char *path, vfs_readdir_cb_t cb, void *ctx) {
    uint32_t inode_num;
    uint8_t type;
    struct vfs_readdir_ctx vctx;

    if (!mounted || cb == 0 || !ext2_lookup_path(path, &inode_num, &type))
        return 0;

    if (type != EXT2_FT_DIR)
        return 0;

    vctx.cb = cb;
    vctx.ctx = ctx;
    return ext2_readdir(inode_num, vfs_dir_cb, &vctx);
}

struct readdir_test_ctx {
    int saw_hello;
    int saw_big;
};

static int print_dir_cb(const char *name, uint8_t name_len, struct vfs_stat *stat, void *ctx) {
    struct readdir_test_ctx *test = ctx;

    serial_print("[VFS] / ");
    for (uint8_t i = 0; i < name_len; i++)
        serial_putc(name[i]);
    serial_print(" size=");
    serial_print_hex(stat->size);
    serial_print("\n");

    if (str_eq_literal(name, name_len, "hello.txt"))
        test->saw_hello = 1;
    if (str_eq_literal(name, name_len, "big.bin"))
        test->saw_big = 1;

    return 1;
}

int vfs_self_test(void) {
    vfs_file_t *file;
    char hello[64];
    uint8_t probe[512];
    struct vfs_stat stat;
    struct readdir_test_ctx dir_ctx = {0, 0};
    size_t read;

    if (!vfs_readdir("/", print_dir_cb, &dir_ctx))
        return 0;

    if (!dir_ctx.saw_hello || !dir_ctx.saw_big)
        return 0;

    file = vfs_open("/hello.txt");
    if (file == 0)
        return 0;

    read = vfs_read(file, hello, sizeof(hello) - 1);
    hello[read] = 0;
    vfs_close(file);

    if (!str_eq_literal(hello, read, "hello from MominOS ext2\n"))
        return 0;

    if (!vfs_stat("/big.bin", &stat) || stat.size < 5ULL * 1024 * 1024)
        return 0;

    file = vfs_open("/big.bin");
    if (file == 0)
        return 0;

    if (!vfs_seek(file, 4ULL * 1024 * 1024 + 64ULL * 1024)) {
        vfs_close(file);
        return 0;
    }

    read = vfs_read(file, probe, sizeof(probe));
    vfs_close(file);

    if (read != sizeof(probe))
        return 0;

    serial_print("[VFS] self-test passed\n");
    return 1;
}
