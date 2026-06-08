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

static int map_user_stack(uint64_t pml4) {
    uint64_t start = USER_STACK_TOP - USER_STACK_SIZE;

    for (uint64_t virt = start; virt < USER_STACK_TOP; virt += PAGE_SIZE) {
        uint64_t phys = pmm_alloc();

        if (phys == 0)
            return 0;

        zero_page((uint8_t *)phys);
        vmm_map_in(pml4, virt, phys, VMM_USER | VMM_WRITABLE);
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

static void user_process_entry(void *arg) {
    struct user_start *start = arg;
    uint64_t entry = start->entry;
    uint64_t stack = start->stack;

    serial_print("[ELF] entering usermode\n");
    user_enter(entry, stack);
}

int elf_spawn(const char *path) {
    struct vfs_stat stat;
    vfs_file_t *file;
    uint8_t *image;
    struct elf64_ehdr *eh;
    uint64_t pml4;
    struct user_start *start;

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

    if (!map_user_stack(pml4)) {
        kfree(image);
        return 0;
    }

    start = kmalloc(sizeof(*start));
    if (start == 0) {
        kfree(image);
        return 0;
    }

    start->entry = eh->e_entry;
    start->stack = USER_STACK_TOP - 8;
    map_kernel_object(pml4, start, sizeof(*start));

    if (thread_create_process(user_process_entry, start, pml4) == 0) {
        kfree(start);
        kfree(image);
        return 0;
    }

    serial_print("[ELF] spawned ");
    serial_print(path);
    serial_print("\n");
    kfree(image);
    return 1;
}
