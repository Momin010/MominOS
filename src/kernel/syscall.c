#include "syscall.h"
#include "sched.h"
#include "serial.h"
#include "tty.h"
#include "vfs.h"
#include "elf.h"
#include "vmm.h"
#include "../ai/diag/capture.h"

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
#define O_WRONLY 0x01
#define O_CREAT  0x40
#define O_TRUNC  0x200
#define O_APPEND 0x400

/* Must be >= MAX_ARGV in elf.c so the boundary validates every argv slot the
   loader later walks and copies. */
#define SPAWN_MAX_ARGV 32

/* Optional stdout-redirect descriptor passed to SYS_SPAWN in a3. The child's
   fd 1 is opened by the kernel on this path so its stdout lands in a file. */
struct spawn_redirect {
    const char *path;       /* absolute or cwd-relative path, or 0 for none */
    long append;            /* nonzero -> append (>>), zero -> truncate (>) */
};

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

/* Lexically normalize an absolute path in place: collapse "." and empty
   components, and resolve ".." by popping the previous component. ".." at the
   root stays at the root, so a path can never escape above "/". The result is
   always absolute and has no trailing slash (except the bare root "/"). */
static void normalize_path(char *path) {
    char out[256];
    uint64_t o = 0;       /* write cursor in out[]; out is built as components */
    uint64_t i = 0;

    /* every absolute path starts with a single leading '/'. */
    out[o++] = '/';

    while (path[i]) {
        uint64_t start;
        uint64_t len;

        while (path[i] == '/')        /* skip run of slashes */
            i++;
        start = i;
        while (path[i] && path[i] != '/')
            i++;
        len = i - start;

        if (len == 0)
            continue;                 /* trailing slash */
        if (len == 1 && path[start] == '.')
            continue;                 /* "." -> no-op */
        if (len == 2 && path[start] == '.' && path[start + 1] == '.') {
            /* pop the last component, clamping at root */
            while (o > 1 && out[o - 1] != '/')
                o--;
            if (o > 1)
                o--;                  /* drop the separating '/' */
            continue;
        }

        /* append "/component", bounded by out[] */
        if (o > 1) {
            if (o + 1 >= sizeof(out))
                break;
            out[o++] = '/';
        }
        for (uint64_t k = 0; k < len; k++) {
            if (o + 1 >= sizeof(out))
                break;
            out[o++] = path[start + k];
        }
    }

    out[o] = 0;
    str_copy(path, out, 256);
}

/* Resolve a user-supplied path against the process cwd into abs[], then
   normalize so ".." can never escape root. Absolute paths (leading '/') are
   copied verbatim; relative paths are joined onto cwd. */
static void resolve_path(const char *path, char *abs, uint64_t cap) {
    struct thread *cur = sched_current_thread();

    if (path[0] == '/') {
        str_copy(abs, path, cap);
        normalize_path(abs);
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
    normalize_path(abs);
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

static uint64_t syscall_dispatch_impl(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3) {
    struct thread *cur = sched_current_thread();

    if (n == SYS_WRITE) {
        const char *buf = (const char *)a2;

        /* user supplies buf/len: must lie wholly in the user half and not wrap.
           guards both the vfs path and the serial buf[i] loop below. */
        if (!vmm_user_range_ok(a2, a3))
            return (uint64_t)-1;

        /* fd 1 (stdout) and 2 (stderr): write to a redirected file if one is
           installed in the fd table, otherwise to the serial console. */
        if (a1 == 1 || a1 == 2) {
            if (a1 < MAX_FDS && cur->fds[a1] != 0)
                return vfs_write(cur->fds[a1], buf, a3);
            for (uint64_t i = 0; i < a3; i++)
                serial_putc(buf[i]);
            return a3;
        }

        /* fd >= 3: write to an open file via the VFS */
        if (a1 >= 3 && a1 < MAX_FDS && cur->fds[a1] != 0)
            return vfs_write(cur->fds[a1], buf, a3);

        return (uint64_t)-1;
    }

    if (n == SYS_READ) {
        /* kernel writes into the user-supplied buffer: validate before any
           path can touch it (both tty_read and vfs_read store into it). */
        if (!vmm_user_range_ok(a2, a3))
            return (uint64_t)-1;
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

        if (!vmm_user_str_ok(a1, VMM_USER_STR_MAX))
            return (uint64_t)-1;
        resolve_path((const char *)a1, abs, sizeof(abs));
        /* a write intent (O_WRONLY/O_TRUNC/O_APPEND, or bare O_CREAT) opens a
           write-through handle; O_APPEND keeps content, otherwise truncate. */
        if (a2 & (O_WRONLY | O_TRUNC | O_APPEND | O_CREAT))
            file = vfs_open_write(abs, (a2 & O_APPEND) ? 1 : 0);
        else
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
        const struct spawn_redirect *redir = (const struct spawn_redirect *)a3;
        vfs_file_t *out = 0;
        struct thread *child;

        /* program path string */
        if (!vmm_user_str_ok(a1, VMM_USER_STR_MAX))
            return (uint64_t)-1;

        /* argv may be 0 (no args). Otherwise validate each pointer slot before
           dereferencing it, then the string it points at, capped at SPAWN_MAX_ARGV
           (elf.c walks/copies these same pointers, so they must be safe here). */
        if (a2 != 0) {
            uint64_t i;

            for (i = 0; i < SPAWN_MAX_ARGV; i++) {
                uint64_t slot = a2 + i * sizeof(char *);

                if (!vmm_user_range_ok(slot, sizeof(char *)))
                    return (uint64_t)-1;
                if (argv[i] == 0)
                    break;                  /* NULL terminator */
                if (!vmm_user_str_ok((uint64_t)argv[i], VMM_USER_STR_MAX))
                    return (uint64_t)-1;
            }
            if (i == SPAWN_MAX_ARGV)        /* no terminator within the cap */
                return (uint64_t)-1;
        }

        /* redir struct (if given) and its path string */
        if (a3 != 0) {
            if (!vmm_user_range_ok(a3, sizeof(struct spawn_redirect)))
                return (uint64_t)-1;
            if (redir->path != 0 &&
                !vmm_user_str_ok((uint64_t)redir->path, VMM_USER_STR_MAX))
                return (uint64_t)-1;
        }

        /* If the parent requested stdout redirection, open the target now (in
           the parent's cwd context) so failures surface before we spawn. */
        if (redir != 0 && redir->path != 0 && redir->path[0] != 0) {
            char rabs[256];

            resolve_path(redir->path, rabs, sizeof(rabs));
            out = vfs_open_write(rabs, redir->append ? 1 : 0);
            if (out == 0)
                return (uint64_t)-1;
        }

        resolve_path((const char *)a1, abs, sizeof(abs));
        child = elf_load_process(abs, argv, cur);
        if (child == 0) {
            if (out != 0)
                vfs_close(out);
            return (uint64_t)-1;
        }

        /* hand the open file to the child as its fd 1 (stdout). The child
           solely owns it; it is closed when the child is reaped (waitpid). */
        if (out != 0)
            child->fds[1] = out;
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
        /* close any fds the child still holds (e.g. a redirected stdout); the
           reaper frees the thread but not its open files. */
        for (int i = 1; i < MAX_FDS; i++) {
            if (child->fds[i] != 0) {
                vfs_close(child->fds[i]);
                child->fds[i] = 0;
            }
        }
        sched_reap(child);
        return (uint64_t)(int64_t)code;
    }

    if (n == SYS_READDIR) {
        char abs[256];
        struct readdir_pack pack;

        /* path string in, kernel packs records into the user buffer out. */
        if (!vmm_user_str_ok(a1, VMM_USER_STR_MAX))
            return (uint64_t)-1;
        if (!vmm_user_range_ok(a2, a3))
            return (uint64_t)-1;
        resolve_path((const char *)a1, abs, sizeof(abs));
        pack.buf = (char *)a2;
        pack.cap = a3;
        pack.used = 0;
        if (!vfs_readdir(abs, readdir_pack_cb, &pack))
            return (uint64_t)-1;
        return pack.used;
    }

    if (n == SYS_CHDIR) {
        char abs[256];
        struct vfs_stat stat;

        if (!vmm_user_str_ok(a1, VMM_USER_STR_MAX))
            return (uint64_t)-1;
        resolve_path((const char *)a1, abs, sizeof(abs));
        if (!vfs_stat(abs, &stat) || stat.type != 2)   /* EXT2_FT_DIR */
            return (uint64_t)-1;
        str_copy(cur->cwd, abs, sizeof(cur->cwd));
        return 0;
    }

    if (n == SYS_GETCWD) {
        char *buf = (char *)a1;
        uint64_t cap = a2;

        /* kernel writes up to cap bytes into the user buffer. */
        if (!vmm_user_range_ok(a1, cap))
            return (uint64_t)-1;
        return str_copy(buf, cur->cwd, cap);
    }

    (void)str_eq;
    return (uint64_t)-1;
}

uint64_t syscall_dispatch(uint64_t n, uint64_t a1, uint64_t a2, uint64_t a3) {
    uint64_t ret = syscall_dispatch_impl(n, a1, a2, a3);
    if (n != 6) { /* skip SYS_EXIT — thread_exit_code never returns */
        struct thread *cur = sched_current_thread();
        diag_capture_syscall(n, a1, a2, a3, ret, cur ? cur->id : 0);
    }
    return ret;
}
