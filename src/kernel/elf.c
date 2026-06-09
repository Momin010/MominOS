#include "elf.h"
#include "arch.h"
#include "kheap.h"
#include "pmm.h"
#include "sched.h"
#include "serial.h"
#include "vfs.h"
#include "vmm.h"

#define EI_NIDENT 16
#define PT_LOAD 1
#define PF_X 1
#define PF_W 2
#define PF_R 4
#define USER_STACK_TOP 0x0000800000000000ULL
#define USER_STACK_SIZE (64ULL * 1024)
#define PAGE_SIZE 4096ULL
#define MAX_ARGV 32

struct elf64_ehdr {
    uint8_t e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} __attribute__((packed));

struct elf64_phdr {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
} __attribute__((packed));

struct user_start {
    uint64_t entry;
    uint64_t stack;
};

static uint64_t align_down(uint64_t value) {
    return value & ~(PAGE_SIZE - 1);
}

static uint64_t align_up(uint64_t value) {
    return (value + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
}

static void zero_page(uint8_t *page) {
    for (uint64_t i = 0; i < PAGE_SIZE; i++)
        page[i] = 0;
}

static void copy_bytes(uint8_t *dst, const uint8_t *src, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        dst[i] = src[i];
}

static int map_segment(uint64_t pml4, uint8_t *image, struct elf64_phdr *ph) {
    uint64_t seg_start = align_down(ph->p_vaddr);
    uint64_t seg_end = align_up(ph->p_vaddr + ph->p_memsz);
    uint64_t flags = VMM_USER;

    if (ph->p_flags & PF_W)
        flags |= VMM_WRITABLE;

    for (uint64_t virt = seg_start; virt < seg_end; virt += PAGE_SIZE) {
        uint64_t phys = pmm_alloc();
        uint8_t *page = (uint8_t *)phys;
        uint64_t page_file_start;
        uint64_t page_file_end;

        if (phys == 0)
            return 0;

        zero_page(page);
        vmm_map_in(pml4, virt, phys, flags);

        page_file_start = virt > ph->p_vaddr ? virt - ph->p_vaddr : 0;
        page_file_end = page_file_start + PAGE_SIZE;

        if (page_file_start < ph->p_filesz) {
            uint64_t copy_start = ph->p_vaddr + page_file_start;
            uint64_t page_off = copy_start - virt;
            uint64_t copy_size;

            if (page_file_end > ph->p_filesz)
                page_file_end = ph->p_filesz;

            copy_size = page_file_end - page_file_start;
            copy_bytes(page + page_off, image + ph->p_offset + page_file_start, copy_size);
        }
    }

    return 1;
}

/* Map the user stack and return the physical address of the topmost page
   (the page containing USER_STACK_TOP-1) via *top_phys. */
static int map_user_stack(uint64_t pml4, uint64_t *top_phys) {
    uint64_t start = USER_STACK_TOP - USER_STACK_SIZE;

    *top_phys = 0;
    for (uint64_t virt = start; virt < USER_STACK_TOP; virt += PAGE_SIZE) {
        uint64_t phys = pmm_alloc();

        if (phys == 0)
            return 0;

        zero_page((uint8_t *)phys);
        vmm_map_in(pml4, virt, phys, VMM_USER | VMM_WRITABLE);

        if (virt == USER_STACK_TOP - PAGE_SIZE)
            *top_phys = phys;
    }

    return 1;
}

static void map_kernel_object(uint64_t pml4, void *ptr, uint64_t size) {
    uint64_t start = (uint64_t)ptr & ~(PAGE_SIZE - 1);
    uint64_t end = ((uint64_t)ptr + size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

    for (uint64_t virt = start; virt < end; virt += PAGE_SIZE) {
        uint64_t phys = vmm_phys(virt);

        if (phys != 0)
            vmm_map_in(pml4, virt, phys, VMM_WRITABLE);
    }
}

static uint64_t str_len(const char *s) {
    uint64_t n = 0;
    while (s[n])
        n++;
    return n;
}

/* Build argc/argv/strings on the new process's user stack. The top stack
   page is identity-accessible to the kernel via top_phys. Returns the
   user-virtual rsp where argc sits (System V layout):
       [rsp]      argc
       [rsp+8]    argv[0]
       ...
       [rsp+8N]   argv[argc-1]
       [...]      NULL
       (strings live higher up in the page)
   rsp is kept 16-byte aligned so that after the C entry pushes the
   return-address-equivalent the compiler sees a correctly aligned frame. */
static uint64_t build_argv_stack(uint64_t top_phys, char *const argv[], int argc) {
    /* page base virtual address and matching physical base */
    uint64_t page_virt = USER_STACK_TOP - PAGE_SIZE;
    uint8_t *page = (uint8_t *)top_phys;
    uint64_t sp_virt = USER_STACK_TOP;     /* grows downward */
    uint64_t str_virt[MAX_ARGV];
    int i;

    /* 1. copy each string onto the stack, top-down */
    for (i = argc - 1; i >= 0; i--) {
        uint64_t len = str_len(argv[i]) + 1;
        sp_virt -= len;
        copy_bytes(page + (sp_virt - page_virt), (const uint8_t *)argv[i], len);
        str_virt[i] = sp_virt;
    }

    /* 2. align the string area down to 8 */
    sp_virt &= ~0x7ULL;

    /* 3. reserve argv[] (argc pointers + NULL terminator) + argc word.
       Total words = 1 (argc) + argc + 1 (NULL). Align the final rsp to
       16 bytes so userspace SSE (movaps) stays happy. */
    {
        uint64_t words = 1 + (uint64_t)argc + 1;
        uint64_t bytes = words * 8;
        uint64_t rsp = sp_virt - bytes;

        rsp &= ~0xFULL;            /* 16-byte align rsp */
        /* recompute the array base so argc lands exactly at rsp */
        sp_virt = rsp;

        uint64_t off = sp_virt - page_virt;
        uint64_t *slot = (uint64_t *)(page + off);

        slot[0] = (uint64_t)argc;
        for (i = 0; i < argc; i++)
            slot[1 + i] = str_virt[i];
        slot[1 + argc] = 0;        /* argv NULL terminator */

        return sp_virt;
    }
}

static void user_process_entry(void *arg) {
    struct user_start *start = arg;
    uint64_t entry = start->entry;
    uint64_t stack = start->stack;

    user_enter(entry, stack);
}

struct thread *elf_load_process(const char *path, char *const argv[], struct thread *parent) {
    struct vfs_stat stat;
    vfs_file_t *file;
    uint8_t *image;
    struct elf64_ehdr *eh;
    uint64_t pml4;
    struct user_start *start;
    uint64_t top_phys;
    struct thread *thread;
    int argc = 0;

    if (!vfs_stat(path, &stat))
        return 0;

    file = vfs_open(path);
    if (file == 0)
        return 0;

    image = kmalloc(stat.size);
    if (image == 0) {
        vfs_close(file);
        return 0;
    }

    if (vfs_read(file, image, stat.size) != stat.size) {
        vfs_close(file);
        kfree(image);
        return 0;
    }
    vfs_close(file);

    eh = (struct elf64_ehdr *)image;
    if (stat.size < sizeof(*eh) ||
        eh->e_ident[0] != 0x7F || eh->e_ident[1] != 'E' ||
        eh->e_ident[2] != 'L' || eh->e_ident[3] != 'F' ||
        eh->e_ident[4] != 2) {
        serial_print("[ELF] invalid image\n");
        kfree(image);
        return 0;
    }

    pml4 = vmm_create_address_space();
    for (uint16_t i = 0; i < eh->e_phnum; i++) {
        struct elf64_phdr *ph = (struct elf64_phdr *)(image + eh->e_phoff + i * eh->e_phentsize);

        if (ph->p_type == PT_LOAD) {
            if (!map_segment(pml4, image, ph)) {
                kfree(image);
                return 0;
            }
        }
    }

    if (!map_user_stack(pml4, &top_phys)) {
        kfree(image);
        return 0;
    }

    start = kmalloc(sizeof(*start));
    if (start == 0) {
        kfree(image);
        return 0;
    }

    start->entry = eh->e_entry;

    if (argv != 0) {
        while (argv[argc] != 0 && argc < MAX_ARGV)
            argc++;
    }
    /* argc==0 still builds a valid, 16-byte-aligned frame: argc=0 at [rsp]
       and a NULL argv terminator, matching the System V ABI so crt0 reads a
       real argc/argv and userspace SSE on the entry frame stays aligned. */
    start->stack = build_argv_stack(top_phys, argv, argc);

    map_kernel_object(pml4, start, sizeof(*start));

    thread = thread_create_process(user_process_entry, start, pml4);
    if (thread == 0) {
        kfree(start);
        kfree(image);
        return 0;
    }
    /* waiter is set later by waitpid(); parent is unused for now */
    (void)parent;

    kfree(image);
    return thread;
}

int elf_spawn(const char *path) {
    return elf_load_process(path, 0, 0) != 0;
}
