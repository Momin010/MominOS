#ifndef ELF_H
#define ELF_H

#include "sched.h"

/* Load a static ELF at `path` into a fresh address space, lay out argv
   on its user stack, and create a runnable process thread. Returns the
   new thread (pid = thread->id) or 0 on failure. parent may be 0. */
struct thread *elf_load_process(const char *path, char *const argv[], struct thread *parent);

/* Legacy boot helper: load and run a process with no args, no parent. */
int elf_spawn(const char *path);

#endif
