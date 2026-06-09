#include "syscall.h"
#include "sched.h"
#include "serial.h"
#include "tty.h"
#include "vfs.h"
#include "elf.h"

#define SYS_WRITE   1
#define SYS_READ    2
#define SYS_OPEN    3
#define SYS_CLOSE   4
#define SYS_EXIT    6
#define SYS_SPAWN   7
#define SYS_WAITPID 8
#define SYS_READDIR 9
#define SYS_CHDIR   10
#define SYS_GETCWD  11

#define O_RDONLY 0

void syscall_init(void) {
    serial_print("[SYSCALL] enabled\n");
}

static int str_eq(const char *a, const char *b) {
    while (*a && *a == *b) {
        a++;
        b++;
    }
    return *a == *b;
}

static uint64_t str_copy(char *dst, const char *src, uint64_t cap) {
    uint64_t n = 0;
    while (src[n] && n + 1 < cap) {
        dst[n] = src[n];
        n++;
    }
    if (cap > 0)
        dst[n] = 0;
    return n;
}

/* Resolve a user-supplied path against the process cwd into abs[].
   Absolute paths (leading '/') are copied verbatim; relative paths are
   joined onto cwd. */
static void resolve_path(const char *path, char *abs, uint64_t cap) {
    struct thread *cur = sched_current_thread();

    if (path[0] == '/') {
        str_copy(abs, path, cap);
        return;
    }

    uint64_t n = str_copy(abs, cur->cwd, cap);
    if (n == 0 || abs[n - 1] != '/') {
        if (n + 1 < cap) {
            abs[n++] = '/';
            abs[n] = 0;
        }
    }
    str_copy(abs + n, path, cap - n);
}

static int alloc_fd(struct thread *t) {
    for (int i = 3; i < MAX_FDS; i++) {
        if (t->fds[i] == 0)
            return i;
    }
    return -1;
}

/* one packed directory record: [type u8][name_len u8][name bytes] */
struct readdir_pack {
    char *buf;
    uint64_t cap;
    uint64_t used;
};

static int readdir_pack_cb(const char *name, uint8_t name_len, struct vfs_stat *stat, void *ctx) {
    struct readdir_pack *p = ctx;
    uint64_t need = 2 + name_len;

    if (p->used + need > p->cap)
        return 0;       /* stop: buffer full */

    p->buf[p->used++] = (char)stat->type;
    p->buf[p->used++] = (char)name_len;
    for (uint8_t i = 0; i < name_len; i++)
        p->buf[p->used++] = name[i];
    return 1;
}

uint64_t syscall_dispatch(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3) {
    struct thread *cur = sched_current_thread();

    {
        uint64_t rflags;
        __asm__ volatile ("pushfq; pop %0" : "=r"(rflags));
        serial_print("[SYS] tid=");
        serial_print_hex(cur ? cur->id : 0xDEAD);
        serial_print(" n=");
        serial_print_hex(n);
        serial_print(" IF=");
        serial_print_hex((rflags >> 9) & 1);
        serial_print("\n");
    }

    if (n == SYS_WRITE) {
        const char *buf = (const char *)a2;
        (void)a1;
        for (uint64_t i = 0; i < a3; i++)
            serial_putc(buf[i]);
        return a3;
    }

    if (n == SYS_READ) {
        if (a1 == 0)
            return tty_read((char *)a2, a3);
        if (a1 >= 3 && a1 < MAX_FDS && cur->fds[a1] != 0)
            return vfs_read(cur->fds[a1], (void *)a2, a3);
        return (uint64_t)-1;
    }

    if (n == SYS_OPEN) {
        char abs[256];
        vfs_file_t *file;
        int fd;

        (void)a2;       /* flags: read-only for now */
        resolve_path((const char *)a1, abs, sizeof(abs));
        file = vfs_open(abs);
        if (file == 0)
            return (uint64_t)-1;

        fd = alloc_fd(cur);
        if (fd < 0) {
            vfs_close(file);
            return (uint64_t)-1;
        }
        cur->fds[fd] = file;
        return (uint64_t)fd;
    }

    if (n == SYS_CLOSE) {
        if (a1 >= 3 && a1 < MAX_FDS && cur->fds[a1] != 0) {
            vfs_close(cur->fds[a1]);
            cur->fds[a1] = 0;
            return 0;
        }
        return (uint64_t)-1;
    }

    if (n == SYS_EXIT) {
        serial_print("[USER] exit ");
        serial_print_hex(a1);
        serial_print("\n");
        thread_exit_code((int)a1);
        return 0;
    }

    if (n == SYS_SPAWN) {
        char abs[256];
        char *const *argv = (char *const *)a2;
        struct thread *child;

        resolve_path((const char *)a1, abs, sizeof(abs));
        child = elf_load_process(abs, argv, cur);
        if (child == 0)
            return (uint64_t)-1;
        return child->id;
    }

    if (n == SYS_WAITPID) {
        struct thread *child = sched_find_thread((uint32_t)a1);
        int code;
        uint64_t flags;

        if (child == 0)
            return (uint64_t)-1;

        child->waiter = cur;

        /* atomic check-then-block: with IRQs disabled the child (running on
           another thread) cannot be scheduled in to set has_exited and miss
           our wake, because we never yield the CPU between the check and the
           block transition. */
        __asm__ volatile ("pushfq; pop %0; cli" : "=r"(flags) : : "memory");
        while (!child->has_exited)
            sched_block_locked();
        __asm__ volatile ("push %0; popfq" : : "r"(flags) : "memory");

        code = child->exit_code;
        sched_reap(child);
        return (uint64_t)(int64_t)code;
    }

    if (n == SYS_READDIR) {
        char abs[256];
        struct readdir_pack pack;

        resolve_path((const char *)a1, abs, sizeof(abs));
        pack.buf = (char *)a2;
        pack.cap = a3;
        pack.used = 0;
        serial_print("[RDIR] abs='");
        serial_print(abs);
        serial_print("' a1=");
        serial_print_hex(a1);
        serial_print(" buf=");
        serial_print_hex(a2);
        serial_print(" cap=");
        serial_print_hex(a3);
        serial_print("\n");
        if (!vfs_readdir(abs, readdir_pack_cb, &pack)) {
            serial_print("[RDIR] vfs_readdir FAILED\n");
            return (uint64_t)-1;
        }
        serial_print("[RDIR] used=");
        serial_print_hex(pack.used);
        serial_print("\n");
        return pack.used;
    }

    if (n == SYS_CHDIR) {
        char abs[256];
        struct vfs_stat stat;

        resolve_path((const char *)a1, abs, sizeof(abs));
        if (!vfs_stat(abs, &stat) || stat.type != 2)   /* EXT2_FT_DIR */
            return (uint64_t)-1;
        str_copy(cur->cwd, abs, sizeof(cur->cwd));
        return 0;
    }

    if (n == SYS_GETCWD) {
        char *buf = (char *)a1;
        uint64_t cap = a2;
        return str_copy(buf, cur->cwd, cap);
    }

    (void)str_eq;
    return (uint64_t)-1;
}
