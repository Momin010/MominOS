"""50 Seed Fault Templates for Kernel Fault Diagnosis Synthetic Data.

Each template is a dict with:
  - id: unique template ID
  - fault_type: category (oom, deadlock, panic, corruption, leak, irq, race, module, fs, network, gpu, security)
  - symptom: short description of the visible symptom
  - root_cause: the underlying kernel issue
  - difficulty: easy/medium/hard
  - template_sft: string with {placeholders} for generating SFT training samples
  - template_dpo_chosen: correct response template
  - template_dpo_rejected: plausible but incorrect response template
  - expected_patterns: list of regex patterns that should appear in good diagnosis
"""

from typing import List, Dict

FAULT_TEMPLATES: List[Dict] = [
    # ── OOM / Memory ──────────────────────────────────────────────────────
    {
        "id": "OOM-001",
        "fault_type": "oom",
        "symptom": "system freezes under memory pressure, OOM killer fires",
        "root_cause": "Memory cgroup limit too low for workload; anonymous page reclaim storm",
        "difficulty": "medium",
        "template_sft": (
            "Kernel log shows 'oom-killer' firing for process running in "
            "cgroup with memory.limit_in_bytes={cgroup_limit}. "
            "System has {total_ram} RAM, swap={swap_size}. "
            "Process RSS={rss}, workload={workload_name}. "
            "Diagnose the root cause and suggest a fix."
        ),
        "template_dpo_chosen": (
            "The OOM killer was triggered because the memory cgroup limit "
            "({cgroup_limit}) is below the process's working set ({rss}). "
            "Root cause: cgroup memory limit too restrictive. "
            "Fix: increase memory.limit_in_bytes to at least {rss*1.5} "
            "or move workload to a cgroup with higher limit."
        ),
        "template_dpo_rejected": (
            "The system ran out of RAM. Add more physical memory. "
            "The OOM killer is a normal kernel feature that reclaims memory."
        ),
        "expected_patterns": ["cgroup", "memory.limit", "working set", "RSS"],
    },
    {
        "id": "OOM-002",
        "fault_type": "oom",
        "symptom": "process killed despite plenty of free RAM",
        "root_cause": "Overcommit policy + mmap exhaustion in 32-bit process",
        "difficulty": "hard",
        "template_sft": (
            "A {arch}-bit process was killed by OOM killer despite "
            "{free_ram} free RAM. Process uses {vma_count} VMAs, "
            "vm.overcommit_memory={overcommit}. "
            "Diagnose and suggest fix."
        ),
        "template_dpo_chosen": (
            "Process killed due to virtual address space exhaustion, not "
            "physical RAM. Root cause: {vma_count} memory-mapped regions "
            "exhausted the {arch}-bit process's {max_vmas} VMA limit or "
            "address space. Fix: increase vm.max_map_count, or use 64-bit "
            "build, or reduce mmap fragmentation."
        ),
        "template_dpo_rejected": (
            "Not enough RAM. Add swap or more memory."
        ),
        "expected_patterns": ["VMA", "vm.max_map_count", "address space", "virtual"],
    },
    {
        "id": "OOM-003",
        "fault_type": "oom",
        "symptom": "kworker consuming gigabytes of slab memory",
        "root_cause": "Slab memory leak in kernfs/dentry cache",
        "difficulty": "hard",
        "template_sft": (
            "kworker{worker_id} consuming {slab_mb} MB in slab caches. "
            "dentry slab = {dentry_slab}, inode_cache = {inode_slab}. "
            "Number of dentries: {dentry_count} after {uptime}. "
            "No workload accessing files. Diagnose."
        ),
        "template_dpo_chosen": (
            "Slab memory leak identified: dentry/inode caches not being "
            "reclaimed despite no active file access. Root cause: kernel "
            "bug where {fs_name} filesystem holds extra references preventing "
            "shrinker from freeing dentries. Fix: kernel patch to "
            "{fs_name}.{shrink_fn}, or drop_caches=2 as temporary workaround."
        ),
        "template_dpo_rejected": (
            "Cache memory is normal, Linux uses free RAM for caching. Not a problem."
        ),
        "expected_patterns": ["slab", "dentry", "shrink", "cache"],
    },
    {
        "id": "OOM-004",
        "fault_type": "oom",
        "symptom": "fragmented memory unable to allocate order-4 pages",
        "root_cause": "Fragmentation prevents contiguous allocation",
        "difficulty": "medium",
        "template_sft": (
            "Device driver {driver_name} fails to allocate {order}-order "
            "pages. /proc/buddyinfo shows {buddy_info} at order {order}. "
            "Total free: {free_mb} MB but allocation fails. Diagnose."
        ),
        "template_dpo_chosen": (
            "Memory fragmentation prevents order-{order} allocation despite "
            "{free_mb} MB free. Root cause: high-order page shortage due to "
            "fragmentation. Fix: enable compaction (vm.compact_memory=1), "
            "use GFP_COMPACT or switch driver to use scatter-gather or "
            "vmalloc instead of kmalloc for large buffers."
        ),
        "template_dpo_rejected": (
            "Out of memory. Add more RAM to the system."
        ),
        "expected_patterns": ["fragmentation", "order", "compaction", "buddy"],
    },
    {
        "id": "OOM-005",
        "fault_type": "oom",
        "symptom": "swap thrashing with high IO wait",
        "root_cause": "vm.swappiness too high for workload; anonymous pages swapped aggressively",
        "difficulty": "easy",
        "template_sft": (
            "System with {total_ram} RAM, swap={swap_total}. "
            "Swap usage={swap_used}, si={swap_in}, so={swap_out} pages/s. "
            "CPU iowait={iowait}% for {duration} minutes. "
            "Running {workload}. vm.swappiness={swappiness}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Swap thrashing detected. Root cause: vm.swappiness={swappiness} "
            "causes the kernel to swap out anon pages aggressively even "
            "though there is {free_ram} free RAM. The swap-in/swap-out "
            "storm causes {iowait}% iowait. Fix: reduce vm.swappiness to "
            "10-20. For workloads that benefit from anon pages staying in "
            "RAM, set swappiness=1."
        ),
        "template_dpo_rejected": (
            "IO bottleneck. Upgrade to faster storage."
        ),
        "expected_patterns": ["swappiness", "swap", "thrashing", "iowait"],
    },
    # ── Deadlocks ─────────────────────────────────────────────────────────
    {
        "id": "DEADLOCK-001",
        "fault_type": "deadlock",
        "symptom": "processes stuck in D state with 'INFO: task hung'",
        "root_cause": "Circular lock dependency between {lock_a} and {lock_b}",
        "difficulty": "hard",
        "template_sft": (
            "'INFO: task hung in {function}' in dmesg. "
            "Process {pid_a} ({comm_a}) holds {lock_a} waiting for {lock_b}. "
            "Process {pid_b} ({comm_b}) holds {lock_b} waiting for {lock_a}. "
            "Stack traces attached. Diagnose the deadlock."
        ),
        "template_dpo_chosen": (
            "Circular lock dependency deadlock between {lock_a} and {lock_b}. "
            "Root cause: {comm_a} acquires {lock_a} then {lock_b}, while "
            "{comm_b} acquires {lock_b} then {lock_a}, creating an "
            "AB-BA deadlock. Fix: ensure consistent lock ordering "
            "(always acquire {lock_a} before {lock_b}) or use "
            "mutex_lock_nested() or convert to trylock with backoff."
        ),
        "template_dpo_rejected": (
            "Processes are stuck. Reboot the system to recover."
        ),
        "expected_patterns": ["AB-BA", "lock ordering", "circular", "waiting for"],
    },
    {
        "id": "DEADLOCK-002",
        "fault_type": "deadlock",
        "symptom": "RCU stall detected",
        "root_cause": "Preempt disabled region holding RCU read lock for too long",
        "difficulty": "hard",
        "template_sft": (
            "RCU stall detected on CPU{cpu} for {duration}s. "
            "Backtrace shows {function} in {module} with "
            "preempt_count={preempt_count}. "
            "Call trace: {stack}. Diagnose."
        ),
        "template_dpo_chosen": (
            "RCU stall caused by prolonged preempt-disable or irq-off region "
            "in {function}. Root cause: {module} holds a spinlock or "
            "preempt_disable() while calling synchronize_rcu() or "
            "rcu_read_lock() across a long operation. Fix: review {function} "
            "for blocking calls inside RCU read-side critical sections, "
            "replace spin_lock() with sleepable mutex where possible."
        ),
        "template_dpo_rejected": (
            "CPU is too slow. Upgrade the processor."
        ),
        "expected_patterns": ["RCU", "preempt", "stall", "critical section"],
    },
    {
        "id": "DEADLOCK-003",
        "fault_type": "deadlock",
        "symptom": "mutex_lock() recursion in the same thread",
        "root_cause": "Non-recursive mutex taken twice by same thread",
        "difficulty": "easy",
        "template_sft": (
            "Process {comm} ({pid}) triggered 'BUG: mutex is locked' at "
            "{function}. Already held by {comm}/{pid}. "
            "Call chain: {call_chain}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Recursive mutex deadlock. Root cause: {function} calls itself "
            "or another function that re-acquires the same non-recursive "
            "mutex. Fix: convert to mutex_lock_nested() if recursive "
            "acquisition is intentional, or restructure code to release "
            "mutex before recursive call, or use a trylock pattern."
        ),
        "template_dpo_rejected": (
            "Bug in the kernel. Update to latest version."
        ),
        "expected_patterns": ["recursive", "mutex", "re-entrant", "nested"],
    },
    {
        "id": "DEADLOCK-004",
        "fault_type": "deadlock",
        "symptom": "hung task in filesystem writeback",
        "root_cause": "Journal thread blocked on buffer lock held by hung process",
        "difficulty": "medium",
        "template_sft": (
            "task hung in journal_commit_transaction for {seconds}s. "
            "Filesystem {fs} on {device}. "
            "JBD2 committing {transaction}. "
            "Process {blocked_pid} ({blocked_comm}) holds buffer_head lock "
            "at {block} on {device}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Filesystem journal deadlock. Root cause: JBD2 commit thread "
            "waiting for buffer_head lock held by {blocked_comm}, which is "
            "stuck in D state waiting for journal space — classic journal "
            "deadlock. Fix: check for data=ordered mode issues, ensure "
            "application closes files properly, or remount with "
            "data=writeback to reduce journal pressure."
        ),
        "template_dpo_rejected": (
            "Hardware failure. Check disk for bad sectors."
        ),
        "expected_patterns": ["JBD2", "journal", "transaction", "buffer_head"],
    },
    {
        "id": "DEADLOCK-005",
        "fault_type": "deadlock",
        "symptom": "spinlock recursion in interrupt handler",
        "root_cause": "Interrupt handler tries to acquire spinlock held by interrupted process",
        "difficulty": "medium",
        "template_sft": (
            "BUG: spinlock recursion on CPU{cpu} in {driver}. "
            "spinlock={lock_addr} already held by {holder}/{holder_pid}. "
            "IRQ handler: {irq_handler}. "
            "Top half: {top_half}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Spinlock deadlock between process context and interrupt handler. "
            "Root cause: {irq_handler} in {driver} tries to acquire a "
            "spinlock that the interrupted process already holds."
            "Fix: use spin_lock_irqsave() or spin_lock_bh() in process "
            "context to disable interrupts while holding the lock, ensuring "
            "the IRQ handler cannot preempt and deadlock."
        ),
        "template_dpo_rejected": (
            "Interrupt storm. Reduce interrupt rate."
        ),
        "expected_patterns": ["spinlock", "IRQ", "interrupt", "spin_lock_irqsave"],
    },
    # ── Kernel Panics ─────────────────────────────────────────────────────
    {
        "id": "PANIC-001",
        "fault_type": "panic",
        "symptom": "NULL pointer dereference in kernel module",
        "root_cause": "Unchecked kmalloc return value / null pointer",
        "difficulty": "easy",
        "template_sft": (
            "Kernel panic: Unable to handle kernel NULL pointer dereference "
            "at virtual address {addr} in {module}. "
            "IP: {function}+{offset}/{size}. "
            "Code: {instruction}. Diagnose."
        ),
        "template_dpo_chosen": (
            "NULL pointer dereference in {module}:{function}. "
            "Root cause: {function} dereferences a pointer without checking "
            "if it was successfully allocated (kmalloc/ioremap returns NULL). "
            "Fix: add NULL check after the allocation at line ~{line_num} "
            "and return appropriate error (-ENOMEM)."
        ),
        "template_dpo_rejected": (
            "Hardware memory error. Replace the RAM stick."
        ),
        "expected_patterns": ["NULL", "dereference", "kmalloc", "allocation"],
    },
    {
        "id": "PANIC-002",
        "fault_type": "panic",
        "symptom": "kernel BUG in {filesystem} filesystem",
        "root_cause": "Corrupted inode data or fs inconsistency",
        "difficulty": "medium",
        "template_sft": (
            "Kernel BUG at fs/{filesystem}/{source}:{line}. "
            "Call trace: {stack}. "
            "Inode {inode_no} on {device}. "
            "Previous crash/fsck: {fsck_history}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Kernel BUG in {filesystem} at {line}. "
            "Root cause: filesystem metadata corruption — inode {inode_no} "
            "has inconsistent state. Likely caused by {cause} "
            "(power loss, hardware fault, or prior panic leaving pending "
            "writes). Fix: run fsck.{filesystem} to repair. Prevent with "
            "barrier=1 mount option and battery-backed write cache."
        ),
        "template_dpo_rejected": (
            "Bug in {filesystem} driver. Switch to a different filesystem."
        ),
        "expected_patterns": ["BUG", "corruption", "inode", "fsck"],
    },
    {
        "id": "PANIC-003",
        "fault_type": "panic",
        "symptom": "stack overflow in kernel thread",
        "root_cause": "Deep recursion in {module} exceeding kernel stack limit",
        "difficulty": "hard",
        "template_sft": (
            "Kernel stack overflow in {function}. "
            "Stack size: {stack_size} (limit: {stack_limit}). "
            "Recursion depth: {depth} calls. "
            "Call chain: {call_chain}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Kernel stack overflow caused by deep recursion in {function}. "
            "Root cause: {depth} recursive calls exceeded the kernel stack "
            "limit ({stack_limit}). Fix: convert recursion to iteration, "
            "increase stack size via 'stack_size' kernel parameter, or "
            "add a recursion depth limit with fallback to workqueue."
        ),
        "template_dpo_rejected": (
            "Stack is too small. Recompile kernel with larger stack."
        ),
        "expected_patterns": ["stack", "overflow", "recursion", "depth"],
    },
    {
        "id": "PANIC-004",
        "fault_type": "panic",
        "symptom": "kernel panic in network driver under load",
        "root_cause": "DMA mapping error / IOMMU page fault",
        "difficulty": "medium",
        "template_sft": (
            "Kernel panic in {driver} at {function}. "
            "IOMMU fault: {iommu_error} at bus address {bus_addr}. "
            "TX ring {tx_ring} descriptor {desc}. "
            "Load: {load} packets/s. Diagnose."
        ),
        "template_dpo_chosen": (
            "IOMMU/DMA fault in {driver}. Root cause: the driver passed "
            "a DMA address that the IOMMU could not translate, likely "
            "due to a stale DMA mapping or buffer overrun. "
            "Fix: check for dma_mapping_error() after dma_map_single(). "
            "If the device supports DAC, add 'iommu=pt' to kernel cmdline "
            "or update the driver to handle 64-bit DMA addresses."
        ),
        "template_dpo_rejected": (
            "Network card is faulty. Replace hardware."
        ),
        "expected_patterns": ["IOMMU", "DMA", "mapping", "driver"],
    },
    {
        "id": "PANIC-005",
        "fault_type": "panic",
        "symptom": "'general protection fault' in syscall handler",
        "root_cause": "User-supplied pointer not validated in copy_from_user",
        "difficulty": "easy",
        "template_sft": (
            "general protection fault in {syscall} syscall handler. "
            "User pointer: {user_ptr}. "
            "Current->addr_limit: {addr_limit}. "
            "Code: {code}. Diagnose."
        ),
        "template_dpo_chosen": (
            "GPF in {syscall} syscall. Root cause: the syscall handler "
            "passed an untrusted user pointer to kernel code without "
            "access_ok() and copy_from_user() validation. "
            "Fix: add access_ok(VERIFY_READ, {user_ptr}, {size}) check "
            "and use copy_from_user() instead of memcpy() or direct "
            "dereference."
        ),
        "template_dpo_rejected": (
            "Corrupted memory. Run memtest86."
        ),
        "expected_patterns": ["GPF", "syscall", "access_ok", "copy_from_user"],
    },
    # ── Memory Corruption ─────────────────────────────────────────────────
    {
        "id": "CORRUPT-001",
        "fault_type": "memory_corruption",
        "symptom": "slub corruption: 'redzone overwritten'",
        "root_cause": "Buffer overflow in kmalloc-{size} slab object",
        "difficulty": "medium",
        "template_sft": (
            "Slub corruption in kmalloc-{size} cache. "
            "Object {obj_addr} redzone overwritten. "
            "Bytes: {corrupted_bytes}. "
            "Last allocator: {alloc_site}. "
            "Free counter: {free_count}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Slab buffer overflow detected. Root cause: code at "
            "{alloc_site} allocated {requested} bytes from kmalloc-{size}"
            "cache but wrote beyond the requested size, overwriting the "
            "redzone marker. Fix: enable CONFIG_SLUB_DEBUG and "
            "CONFIG_KASAN to pinpoint overrun location. The fix is "
            "in the caller at {alloc_site}: ensure allocation size "
            "matches the actual write size."
        ),
        "template_dpo_rejected": (
            "Memory hardware error. Replace DIMM."
        ),
        "expected_patterns": ["slub", "redzone", "kmalloc", "overflow"],
    },
    {
        "id": "CORRUPT-002",
        "fault_type": "memory_corruption",
        "symptom": "list corruption in process list",
        "root_cause": "Concurrent modification of kernel linked list (race condition)",
        "difficulty": "hard",
        "template_sft": (
            "list_del corruption in {function}. "
            "prev->next={prev_next}, next->prev={next_prev}. "
            "Corrupted list: {list_name} at {list_addr}. "
            "CPU: {cpu}. Process: {comm}/{pid}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Kernel linked list corruption. Root cause: concurrent "
            "removal from {list_name} without proper lock protection. "
            "Multiple threads or interrupt handlers are modifying the "
            "list without holding the appropriate spinlock or mutex. "
            "Fix: audit {function} and its callers to ensure "
            "spin_lock_irqsave() protects all list operations."
        ),
        "template_dpo_rejected": (
            "Memory corruption from faulty hardware."
        ),
        "expected_patterns": ["list_del", "corruption", "rcu", "spin_lock"],
    },
    {
        "id": "CORRUPT-003",
        "fault_type": "memory_corruption",
        "symptom": "kernel NULL pointer dereference after use-after-free",
        "root_cause": "UAF after object freed but dangling pointer still used",
        "difficulty": "hard",
        "template_sft": (
            "NULL pointer dereference in {function}. "
            "Object type: {object_type} at {freed_addr} (freed {free_time} "
            "ago at {free_site}). "
            "Use site: {use_site}. "
            "kmemleak confirms no reference. Diagnose."
        ),
        "template_dpo_chosen": (
            "Use-after-free (UAF) bug. Root cause: {function} used a "
            "pointer to {object_type} that was freed at {free_site} "
            "without resetting the reference. Fix: add NULL check or "
            "convert to reference-counted object using kref_get/put. "
            "Enabling CONFIG_KASAN=y and SLUB_DEBUG_ON can help "
            "reproduce this reliably."
        ),
        "template_dpo_rejected": (
            "Old kernel bug. Upgrade kernel version."
        ),
        "expected_patterns": ["use-after-free", "UAF", "kref", "dangling"],
    },
    {
        "id": "CORRUPT-004",
        "fault_type": "memory_corruption",
        "symptom": "vmalloc area corruption detected",
        "root_cause": "vmalloc'd buffer overrun from {driver}",
        "difficulty": "medium",
        "template_sft": (
            "vmalloc info: vmap area for {driver} at "
            "{start}-{end} ({size}) corrupted. "
            "Guard page at {guard} overwritten. "
            "Pattern: {overwrite_pattern}. Diagnose."
        ),
        "template_dpo_chosen": (
            "vmalloc buffer overrun in {driver}. Root cause: vmalloc'd "
            "buffer of size {size} was written past its end, corrupting "
            "the guard page. Fix: check buffer size calculations in "
            "{driver}, particularly DMA transfer length. For DMA, "
            "use dma_alloc_coherent() instead. Ensure sg table "
            "entries match the actual transfer size."
        ),
        "template_dpo_rejected": (
            "Memory pressure. Increase vmalloc reserve."
        ),
        "expected_patterns": ["vmalloc", "guard page", "overrun", "DMA"],
    },
    {
        "id": "CORRUPT-005",
        "fault_type": "memory_corruption",
        "symptom": "intermittent data corruption in user process",
        "root_cause": "DMA cache coherence issue with {device}",
        "difficulty": "hard",
        "template_sft": (
            "User process {comm} reads corrupted data from DMA buffer "
            "shared with {device}. "
            "Expected: {expected_hex}, Got: {actual_hex}. "
            "Architecture: {arch}, cache line size: {cacheline}. "
            "dma_coherent={dma_coherent}. Diagnose."
        ),
        "template_dpo_chosen": (
            "DMA cache coherence bug. Root cause: {device} writes to "
            "memory via DMA while the CPU cache holds stale data. "
            "Fix: ensure dma_map_single() with correct direction, or use "
            "dma_alloc_coherent() for uncached buffers. On {arch}, "
            "ensure dma_sync_single_for_cpu() is called before CPU reads "
            "and dma_sync_single_for_device() before device writes."
        ),
        "template_dpo_rejected": (
            "Faulty {device}. Replace the hardware."
        ),
        "expected_patterns": ["DMA", "cache", "coherent", "dma_sync"],
    },
    # ── Memory Leaks ──────────────────────────────────────────────────────
    {
        "id": "LEAK-001",
        "fault_type": "memory_leak",
        "symptom": "kmalloc-{size} slab cache growing unbounded",
        "root_cause": "Memory allocation in hot path never freed",
        "difficulty": "easy",
        "template_sft": (
            "kmalloc-{size} slab cache: {allocated} objects, "
            "growing at {rate} objects/min. "
            "Top allocator in /proc/slabinfo: {top_alloc}. "
            "System uptime: {uptime}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Slab memory leak in kmalloc-{size}. Root cause: {top_alloc} "
            "allocates objects in its hot path but fails to kfree() them "
            "on all error/cleanup paths. Fix: audit alloc/free paths in "
            "{top_alloc}. Use kmemleak to find unreferenced objects. "
            "Add cleanup in error handling."
        ),
        "template_dpo_rejected": (
            "Memory is just dirty. Write to /proc/sys/vm/drop_caches."
        ),
        "expected_patterns": ["slab", "leak", "kmemleak", "kfree"],
    },
    {
        "id": "LEAK-002",
        "fault_type": "memory_leak",
        "symptom": "page allocation failure in {module}",
        "root_cause": "{module} leaks pages during operation",
        "difficulty": "medium",
        "template_sft": (
            "Page allocation failure in {module}/{function}. "
            "Free pages: {free_pages} (watermark: {watermark}). "
            "Page table pages: {pt_pages}. "
            "Module up for {uptime}, allocated {total_pages} pages. "
            "Diagnose."
        ),
        "template_dpo_chosen": (
            "Page leak in {module}. Root cause: {function} allocates "
            "{order}-order pages via alloc_pages()/__get_free_pages() "
            "but does not free them on all paths. Each operation adds "
            "{page_size} KB to unreclaimable memory. Fix: audit "
            "__free_pages() calls in error paths. Use free_page() "
            "in cleanup routines."
        ),
        "template_dpo_rejected": (
            "Low on memory. Add more RAM."
        ),
        "expected_patterns": ["page allocation", "leak", "free_pages", "watermark"],
    },
    {
        "id": "LEAK-003",
        "fault_type": "memory_leak",
        "symptom": "kmemleak reports unreferenced objects in {module}",
        "root_cause": "Reference counting bug in struct {struct_name}",
        "difficulty": "hard",
        "template_sft": (
            "kmemleak: {count} unreferenced objects of size {size} in {module}. "
            "Backtrace: {backtrace}. "
            "Reference count of {struct_name} at {obj_addr}: {refcount}. "
            "Module ops: {ops}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Reference-counting leak in {struct_name}. Root cause: "
            "the get/put pattern in {module} is unbalanced — "
            "a kref_get() in {get_site} lacks a matching kref_put() "
            "in {put_site}. When refcount never reaches zero, the "
            "object is never freed. Fix: audit refcount ops in "
            "{module} to ensure every get() has a paired put() "
            "on cleanup paths."
        ),
        "template_dpo_rejected": (
            "Memory fragmentation. defrag by writing 1 to "
            "/proc/sys/vm/compact_memory."
        ),
        "expected_patterns": ["kmemleak", "refcount", "kref", "unreferenced"],
    },
    # ── Interrupt / IRQ ───────────────────────────────────────────────────
    {
        "id": "IRQ-001",
        "fault_type": "irq",
        "symptom": "irq {irq_num}: nobody cared (timeout)",
        "root_cause": "Unhandled IRQ, device not properly claiming interrupt",
        "difficulty": "medium",
        "template_sft": (
            "irq {irq_num}: nobody cared (timeout). "
            "Disabling IRQ #{irq_num}. "
            "Device: {device} on {bus}. "
            "IRQ handler registered: {handler_registered}. "
            "Status reg: {status_reg}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Spurious IRQ or missing interrupt handler. Root cause: "
            "{device} raised IRQ {irq_num} but no handler claimed it. "
            "Possible causes: handler registered for wrong IRQ number, "
            "shared IRQ line not properly handled, or device asserted "
            "interrupt before handler was ready. Fix: verify IRQ "
            "number matches request_irq(). For shared IRQ, ensure "
            "irq_handler_t returns IRQ_HANDLED only for its device."
        ),
        "template_dpo_rejected": (
            "IRQ storm. Disable MSI and use legacy INTx."
        ),
        "expected_patterns": ["spurious IRQ", "nobody cared", "request_irq", "sharing"],
    },
    {
        "id": "IRQ-002",
        "fault_type": "irq",
        "symptom": "soft lockup on CPU{cpu}: IRQ handler takes too long",
        "root_cause": "Heavy work in top-half IRQ handler",
        "difficulty": "easy",
        "template_sft": (
            "soft lockup on CPU{cpu} for {duration}s. "
            "IRQ handler: {irq_handler} from {driver}. "
            "Interrupts/sec: {irq_rate}. "
            "Handler duration: {handler_us} us. Diagnose."
        ),
        "template_dpo_chosen": (
            "Soft lockup caused by excessive time in IRQ handler. "
            "Root cause: {irq_handler} is doing too much work in "
            "hardirq context, blocking the CPU for {handler_us}us. "
            "Fix: move heavy processing to threaded IRQ (request_threaded_irq) "
            "or tasklet/workqueue. Keep hardirq handler minimal "
            "(typically only ack the IRQ and schedule work)."
        ),
        "template_dpo_rejected": (
            "CPU is too slow. Upgrade processor."
        ),
        "expected_patterns": ["soft lockup", "hardirq", "threaded IRQ", "top-half"],
    },
    {
        "id": "IRQ-003",
        "fault_type": "irq",
        "symptom": "MSI/MSI-X interrupt not firing after driver load",
        "root_cause": "MSI configuration mismatch between driver and device",
        "difficulty": "hard",
        "template_sft": (
            "{device} registered {nvec} MSI-X vectors but interrupts "
            "not firing. /proc/interrupts: {irq_count} for vector "
            "{vector}. PCI config space: {pci_config}. "
            "Driver: {driver} version {version}. Diagnose."
        ),
        "template_dpo_chosen": (
            "MSI-X interrupt configuration failure. Root cause: "
            "{driver} requested {nvec} vectors but the device's MSI-X "
            "table or PCI config was not properly programmed. "
            "Fix: ensure pci_enable_msix_exact() returns 0, check that "
            "the driver writes correct message data/address to the "
            "MSI-X table BAR. For debugging, try pci=nomsi, or "
            "update the driver's MSI-X setup code."
        ),
        "template_dpo_rejected": (
            "ACPI issue. Update BIOS/firmware."
        ),
        "expected_patterns": ["MSI-X", "vector", "pci_enable_msix", "interrupt remapping"],
    },
    # ── Race Conditions ───────────────────────────────────────────────────
    {
        "id": "RACE-001",
        "fault_type": "race_condition",
        "symptom": "concurrent writes to procfs file corrupt output",
        "root_cause": "Lack of serialization in procfs read handler",
        "difficulty": "easy",
        "template_sft": (
            "Reading /proc/{proc_file} concurrently from {n_readers} "
            "processes produces garbled/interleaved output: "
            "sample: '{sample_output}'. "
            "Handler: {handler_fn} in {module}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Procfs read race condition. Root cause: {handler_fn} reads "
            "a shared buffer without locking while multiple readers "
            "interleave. Fix: use seq_file interface (single_open + "
            "seq_printf) which provides per-reader buffering. Or "
            "protect the buffer with a mutex."
        ),
        "template_dpo_rejected": (
            "Procfs is deprecated. Move to sysfs or debugfs."
        ),
        "expected_patterns": ["race", "seq_file", "procfs", "concurrent"],
    },
    {
        "id": "RACE-002",
        "fault_type": "race_condition",
        "symptom": "TOCTOU race in device permission check",
        "root_cause": "Time-of-check-time-of-use in security check",
        "difficulty": "hard",
        "template_sft": (
            "Security bypass via open() of {device} after permissions "
            "changed. Check at {check_line} passes, but by the time "
            "{use_line} runs, {perm_changed}. "
            "CAP_SYS_ADMIN={cap}. Diagnose."
        ),
        "template_dpo_chosen": (
            "TOCTOU (time-of-check-time-of-use) race condition in "
            "{module}. Root cause: the authorization check at "
            "{check_line} and the privileged operation at {use_line} "
            "are not atomic — permissions can change between them. "
            "Fix: perform the security check inside the mutex/ lock "
            "that protects the resource, or use a single atomic "
            "capable() call at the point of use."
        ),
        "template_dpo_rejected": (
            "Permission denied. Grant CAP_SYS_ADMIN to process."
        ),
        "expected_patterns": ["TOCTOU", "race", "atomic", "capable"],
    },
    {
        "id": "RACE-003",
        "fault_type": "race_condition",
        "symptom": "module removal causes crash while in use",
        "root_cause": "Missing reference counting in module operations",
        "difficulty": "medium",
        "template_sft": (
            "BUG after rmmod {module}: NULL pointer dereference in "
            "{function}. Module refcount was {refcount} during removal. "
            "Open count: {open_count}. "
            "Exit function: {exit_fn}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Module use-after-removal crash. Root cause: {module}.exit "
            "cleans up resources while {function} still holds a live "
            "reference. The module lacks proper try_module_get()/"
            "module_put() refcounting in its file_operations. "
            "Fix: add try_module_get(THIS_MODULE) in .open and "
            "module_put(THIS_MODULE) in .release, or use the "
            ".owner = THIS_MODULE pattern."
        ),
        "template_dpo_rejected": (
            "Module bug. Recompile module with debugging."
        ),
        "expected_patterns": ["rmmod", "refcount", "module_put", "try_module_get"],
    },
    {
        "id": "RACE-004",
        "fault_type": "race_condition",
        "symptom": "timer fires after handler freed",
        "root_cause": "Del_timer_sync not used before freeing timer data",
        "difficulty": "medium",
        "template_sft": (
            "Use-after-free in timer callback {timer_fn}. "
            "Timer fired at {fire_jiffies}, but callback data at "
            "{data_addr} was freed at {free_time}. "
            "del_timer() used instead of del_timer_sync(). Diagnose."
        ),
        "template_dpo_chosen": (
            "Timer UAF race. Root cause: timer callback {timer_fn} "
            "accesses freed memory because del_timer() was used instead "
            "of del_timer_sync(). del_timer() only removes the timer "
            "if it hasn't started; if it's running on another CPU, "
            "the callback can execute after data is freed. "
            "Fix: replace del_timer() with del_timer_sync() and "
            "ensure timer callback uses proper RCU or refcounting."
        ),
        "template_dpo_rejected": (
            "CPU scheduling bug. Pin process to one CPU."
        ),
        "expected_patterns": ["del_timer_sync", "timer", "UAF", "callback"],
    },
    {
        "id": "RACE-005",
        "fault_type": "race_condition",
        "symptom": "work item running after workqueue destroyed",
        "root_cause": "Destroy_workqueue while work items still pending",
        "difficulty": "easy",
        "template_sft": (
            "Workqueue: {wq_name} destroyed but work {work_fn} still "
            "pending/scheduled. "
            "Pending works in list: {pending_count}. "
            "Destroy call site: {destroy_site}. "
            "Diagnose."
        ),
        "template_dpo_chosen": (
            "Use-after-free in workqueue. Root cause: "
            "destroy_workqueue() called at {destroy_site} while "
            "{pending_count} work items were still scheduled. "
            "Fix: flush_workqueue() or cancel_work_sync() before "
            "destroy_workqueue(). For delayed work, use "
            "cancel_delayed_work_sync()."
        ),
        "template_dpo_rejected": (
            "Workqueue stuck. Try ctrl+alt+del."
        ),
        "expected_patterns": ["workqueue", "flush", "cancel_work_sync", "destroy_workqueue"],
    },
    # ── Kernel Modules ────────────────────────────────────────────────────
    {
        "id": "MODULE-001",
        "fault_type": "module",
        "symptom": "module fails to load: 'Unknown symbol'",
        "root_cause": "Missing or unexported symbol dependency",
        "difficulty": "easy",
        "template_sft": (
            "insmod {module}.ko fails: 'Unknown symbol {symbol}'. "
            "Module uses symbols: {used_symbols}. "
            "Kernel exports via /proc/kallsyms: {exported}. "
            "Kernel version: {kernel_ver}. Module compiled for: {module_ver}. "
            "Diagnose."
        ),
        "template_dpo_chosen": (
            "Module cannot load due to unresolved symbol {symbol}. "
            "Root cause: the symbol is not exported by the running "
            "kernel or the dependency module is not loaded. "
            "Fix: ensure prerequisite module is loaded (modprobe "
            "{dependency}), and that module was compiled against "
            "the same kernel version ({kernel_ver}). Use "
            "'modinfo {symbol}' to find the exporting module."
        ),
        "template_dpo_rejected": (
            "Module is corrupted. Reinstall kernel headers."
        ),
        "expected_patterns": ["symbol", "export", "modprobe", "dependency"],
    },
    {
        "id": "MODULE-002",
        "fault_type": "module",
        "symptom": "module loads but device probe fails with -ENODEV",
        "root_cause": "Device ID table mismatch or PCI ID missing",
        "difficulty": "medium",
        "template_sft": (
            "{driver} probe of {device} failed: -ENODEV. "
            "PCI vendor={vendor}, device={devid}, subsystem={subsys}."
            "Driver ID table: {id_table}. "
            "Driver supports: {supported_devs}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Probe failure -ENODEV. Root cause: {device} (vendor={vendor}, "
            "device={devid}) is not in the driver's MODULE_DEVICE_TABLE "
            "or the ID table does not match this revision. "
            "Fix: add the PCI ID to the driver's id_table array and "
            "MODULE_DEVICE_TABLE. For quick test, use "
            "'echo {vendor} {devid} > /sys/bus/pci/drivers/{driver}/new_id'."
        ),
        "template_dpo_rejected": (
            "Device not supported. Write your own driver."
        ),
        "expected_patterns": ["-ENODEV", "ID table", "PCI ID", "probe"],
    },
    {
        "id": "MODULE-003",
        "fault_type": "module",
        "symptom": "module reload triggers kernel warning",
        "root_cause": "Missing cleanup in module .exit function, stale state",
        "difficulty": "medium",
        "template_sft": (
            "After rmmod {module} and modprobe {module}, kernel warning "
            "in {function}: '{warning}'. "
            "Repeated resources: {resource_conflict}. "
            "Exit function: {exit_fn}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Module reload failure due to incomplete cleanup. "
            "Root cause: {exit_fn} does not fully release resources "
            "acquired during init (e.g., {resource_conflict}). "
            "Fix: audit {exit_fn} — ensure all register/unregister "
            "pairs match, all kfree's cover all alloc paths, and "
            "{resource_conflict} is properly released."
        ),
        "template_dpo_rejected": (
            "Module is broken. Reboot to unload the state."
        ),
        "expected_patterns": ["reload", "cleanup", ".exit", "double register"],
    },
    # ── Filesystem ────────────────────────────────────────────────────────
    {
        "id": "FS-001",
        "fault_type": "filesystem",
        "symptom": "filesystem remounts read-only after error",
        "root_cause": "Journal replay failure or metadata write error",
        "difficulty": "easy",
        "template_sft": (
            "Filesystem {fs} on {device} remounted read-only. "
            "Last error: '{error_msg}' at sector {sector}. "
            "Journal status: {journal_status}. "
            "Device error count: {device_errors}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Filesystem forced read-only due to I/O error. "
            "Root cause: write-back failed on sector {sector} with "
            "'{error_msg}', causing the journal to mark the FS "
            "as error. The kernel remounts read-only to prevent "
            "further corruption. Fix: check SMART data for device "
            "{device}, run fsck.{fs} when unmounted, replace failing "
            "hardware if errors are media-related."
        ),
        "template_dpo_rejected": (
            "Filesystem corrupted. Format and restore from backup."
        ),
        "expected_patterns": ["read-only", "journal", "I/O error", "fsck"],
    },
    {
        "id": "FS-002",
        "fault_type": "filesystem",
        "symptom": "directory listing returns stale entries",
        "root_cause": "page cache invalidation not triggered after remote change",
        "difficulty": "hard",
        "template_sft": (
            "NFS/{fs} client shows stale directory listing: "
            "file {filename} exists on server but client shows "
            "{client_state}. "
            "Attribute cache timeout: {acregmax}s. "
            "Client up for {uptime}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Stale directory cache. Root cause: the client's attribute "
            "and dentry cache has not been invalidated after the "
            "server-side change. With {acregmax}s attr timeout, "
            "the client uses cached data without revalidating. "
            "Fix: reduce acregmax/acdirmax on mount, or add "
            "lookupcache=positive mount option, or force "
            "invalidation with 'echo 2 > /proc/sys/vm/drop_caches'."
        ),
        "template_dpo_rejected": (
            "Network issue. Check connectivity to server."
        ),
        "expected_patterns": ["cache", "attribute timeout", "invalidation", "stale"],
    },
    {
        "id": "FS-003",
        "fault_type": "filesystem",
        "symptom": "quota exceeded on root filesystem despite free space",
        "root_cause": "Inode quota limit reached, not block quota",
        "difficulty": "easy",
        "template_sft": (
            "Disk quota exceeded for user {user} on {mount_point}. "
            "Block usage: {block_used}/{block_limit}. "
            "Inode usage: {inode_used}/{inode_limit}. "
            "Free space: {free_space} GB. Diagnose."
        ),
        "template_dpo_chosen": (
            "Inode quota exceeded. Root cause: user {user} has created "
            "{inode_used} files/dirs, reaching the inode soft limit "
            "({inode_limit}). Block space is still free. "
            "Fix: increase inode quota with 'setquota -u {user} "
            "{block_soft} {block_hard} {inode_soft*2} {inode_hard*2} {fs}', "
            "or delete unnecessary small files."
        ),
        "template_dpo_rejected": (
            "Disk is full. Free up space by deleting old logs."
        ),
        "expected_patterns": ["inode", "quota", "limit", "files"],
    },
    {
        "id": "FS-004",
        "fault_type": "filesystem",
        "symptom": "kernel warning about ext4 delayed allocation",
        "root_cause": "Delayed allocation reservation mismatch",
        "difficulty": "hard",
        "template_sft": (
            "EXT4-fs warning: {device}: delayed block allocation "
            "reservation mismatch in inode {inode}. "
            "Reserved: {reserved} blocks, allocated: {allocated}. "
            "File size: {file_size}. Fragments: {fragments}. "
            "Diagnose."
        ),
        "template_dpo_chosen": (
            "Ext4 delayed allocation warning. Root cause: the delayed "
            "allocation reservation (reserved {reserved}) does not match "
            "the actual allocation (allocated {allocated}), often due "
            "to ENOSPC during writeback or a race in the reservation "
            "code. Fix: update e2fsprogs and kernel to latest stable, "
            "mount with 'delalloc' option explicitly. Check for "
            "snapshot or reflink operations that confuse accounting."
        ),
        "template_dpo_rejected": (
            "Filesystem corruption. Run fsck immediately."
        ),
        "expected_patterns": ["delalloc", "reservation", "writeback", "ext4"],
    },
    # ── Networking ────────────────────────────────────────────────────────
    {
        "id": "NET-001",
        "fault_type": "network",
        "symptom": "TCP connection timeouts under load",
        "root_cause": "netdev TX ring full / BQL not regulating backpressure",
        "difficulty": "medium",
        "template_sft": (
            "TCP connection timeouts during {workload}. "
            "TX ring {tx_ring}: {tx_packets} queued, {max_ring}. "
            "BQL limit: {bql_limit}. "
            "Dropped: {drops}. Diagnose."
        ),
        "template_dpo_chosen": (
            "TX ring buffer full causing TCP timeouts. Root cause: the "
            "netdev TX ring ({tx_ring}) is full because Byte Queue Limits "
            "(BQL) has a limit of {bql_limit} bytes, insufficient for "
            "the burst rate of {workload}. Fix: increase TX ring size "
            "via ethtool -G {dev} tx {new_size}, or tune BQL with "
            "'echo {higher_limit} > /sys/class/net/{dev}/queues/tx-0/"
            "byte_queue_limits/limit_max'."
        ),
        "template_dpo_rejected": (
            "Network bandwidth exceeded. Increase link speed."
        ),
        "expected_patterns": ["TX ring", "BQL", "timeout", "byte_queue_limits"],
    },
    {
        "id": "NET-002",
        "fault_type": "network",
        "symptom": "RX packet drops in netdev with no errors",
        "root_cause": "GRO/LRO coalescing failure or napi budget exhaustion",
        "difficulty": "hard",
        "template_sft": (
            "ifconfig {dev}: {rx_drops} dropped packets, 0 errors. "
            "NAPI budget: {napi_budget}, poll weight: {weight}. "
            "RX ring: {rx_ring}/{rx_max}. "
            "GRO packets: {gro_packets}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Silent RX drops without errors. Root cause: NAPI polling "
            "budget of {napi_budget} is reached before the ring is drained, "
            "or GRO is not coalescing effectively, causing the ring "
            "to overflow. Fix: increase napi_defer_hard_irqs and "
            "gro_flush_timeout via sysfs. Use 'ethtool -C {dev} "
            "rx-usecs {higher_value}' to moderate interrupts."
        ),
        "template_dpo_rejected": (
            "Driver bug. Update NIC driver firmware."
        ),
        "expected_patterns": ["NAPI", "GRO", "ring", "dropped"],
    },
    {
        "id": "NET-003",
        "fault_type": "network",
        "symptom": "network throughput collapses under small packet flood",
        "root_cause": "Interrupt moderation disabled; livelock from excessive IRQs",
        "difficulty": "medium",
        "template_sft": (
            "Throughput collapse for {dev}: from {goodput} to {badput} "
            "Mbps with {packet_size}B packets. "
            "Interrupts/sec: {irq_rate}. "
            "Coalescing: {coalesce_settings}. "
            "CPU: {cpu_util}. Diagnose."
        ),
        "template_dpo_chosen": (
            "RX livelock from interrupt storm. Root cause: the NIC "
            "fires an interrupt for every small packet ({packet_size}B), "
            "and CPU{cpu} spends 100% servicing interrupts ("
            "{irq_rate}/s), never reaching NAPI poll. "
            "Fix: enable interrupt coalescing (ethtool -C {dev} "
            "rx-usecs {usecs} rx-frames {frames}), or increase "
            "pktgen/coalescing parameters."
        ),
        "template_dpo_rejected": (
            "CPU bottleneck. Add more CPU cores."
        ),
        "expected_patterns": ["livelock", "coalescing", "interrupt storm", "NAPI"],
    },
    {
        "id": "NET-004",
        "fault_type": "network",
        "symptom": "TCP zero window from memory pressure on sender",
        "root_cause": "tcp_wmem pressure causing socket buffer starvation",
        "difficulty": "hard",
        "template_sft": (
            "TCP zero window advertised by sender {saddr}:{sport}. "
            "tcp_mem: {tcp_mem_pressure}. "
            "Socket wmem allocated: {wmem_alloc}/{wmem_max}. "
            "Skb in write queue: {skb_count}. Diagnose."
        ),
        "template_dpo_chosen": (
            "TCP zero window due to sender-side memory pressure. "
            "Root cause: the socket's write buffer is full "
            "({wmem_alloc}/{wmem_max}) because tcp_mem pressure "
            "({tcp_mem_pressure}) prevents allocating more skbs. "
            "The sender cannot accept more data from userspace. "
            "Fix: increase tcp_wmem min/default/max sysctl values, "
            "or reduce the application's send buffer size to avoid "
            "overrun. Enable BQL on the NIC to smooth TX."
        ),
        "template_dpo_rejected": (
            "Network congestion. Reduce send rate."
        ),
        "expected_patterns": ["zero window", "tcp_mem", "wmem", "pressure"],
    },
    {
        "id": "NET-005",
        "fault_type": "network",
        "symptom": "arp flooding causes neighbour table overflow",
        "root_cause": "Neighbour table GC not keeping up with ARP storm",
        "difficulty": "medium",
        "template_sft": (
            "Neighbour table overflow: 'Neighbour table overflow!' "
            "in dmesg. ARP entries: {arp_count}/{arp_max}. "
            "GC interval: {gc_interval}, GC thresh1/2/3: "
            "{gc_thresh1}/{gc_thresh2}/{gc_thresh3}. "
            "Source: {arp_source} (infected host). Diagnose."
        ),
        "template_dpo_chosen": (
            "Neighbour table overflow from ARP flooding. Root cause: "
            "host {arp_source} is sending excessive ARP requests, "
            "filling the ARP cache faster than GC can prune. "
            "Fix: increase gc_thresh3 (sysctl net.ipv4.neigh.default."
            "gc_thresh3={new_value}), reduce gc_interval, and "
            "block the flooding host with ebtables/iptables. "
            "Prevent with port-security on the switch."
        ),
        "template_dpo_rejected": (
            "ARP table full. Flush ARP cache with ip -s -s neigh flush all."
        ),
        "expected_patterns": ["neighbour", "ARP", "GC", "overflow"],
    },
    # ── GPU / DRM ─────────────────────────────────────────────────────────
    {
        "id": "GPU-001",
        "fault_type": "gpu",
        "symptom": "GPU hang reported by DRM scheduler",
        "root_cause": "Job timeout — shader program infinite loop or driver missed fence signal",
        "difficulty": "hard",
        "template_sft": (
            "GPU hang on {gpu}: DRM scheduler job timeout after "
            "{timeout}ms. Ring {ring}: seqno={seqno}, last signaled={last}. "
            "Guilty process: {comm}/{pid}. "
            "Fence context: {fence_ctx}. Diagnose."
        ),
        "template_dpo_chosen": (
            "GPU hang detected by DRM scheduler. Root cause: the "
            "fence at seqno {seqno} was not signaled within "
            "{timeout}ms, indicating the GPU is hung on ring "
            "{ring}. Possible causes: shader infinite loop, "
            "descheduling issue, or driver failing to update "
            "the fence. Fix: check if this is repeatable with "
            "specific {comm} workload. For driver issues, check "
            "ring buffer updates. Workaround: 'echo 1 > "
            "/sys/kernel/debug/dri/0/amdgpu_gpu_recover'."
        ),
        "template_dpo_rejected": (
            "GPU is dead. Replace the graphics card."
        ),
        "expected_patterns": ["GPU hang", "scheduler", "fence", "timeout"],
    },
    {
        "id": "GPU-002",
        "fault_type": "gpu",
        "symptom": "TTM out of memory for GPU buffer allocation",
        "root_cause": "VRAM fragmentation or TTM eviction stuck",
        "difficulty": "medium",
        "template_sft": (
            "TTM: Out of memory for buffer object of {size} MB on "
            "{gpu}. VRAM: {vram_total}/{vram_free}. GTT: "
            "{gtt_used}/{gtt_total}. "
            "Eviction global state: {eviction_state}. Diagnose."
        ),
        "template_dpo_chosen": (
            "TTM OOM for GPU buffer. Root cause: VRAM fragmentation "
            "prevents allocating {size} MB contiguous block despite "
            "{vram_free} MB free. TTM eviction is stuck because "
            "all buffers are busy/pinned. Fix: check for pinned buffers "
            "with 'cat /sys/kernel/debug/dri/0/vram_mm'. "
            "Reduce fragmentation by reserving contiguous VRAM at "
            "boot via amdgpu.vram_fragment_size. For immediate fix, "
            "terminate memory-hungry processes using {gpu}."
        ),
        "template_dpo_rejected": (
            "Not enough VRAM. Install a GPU with more memory."
        ),
        "expected_patterns": ["TTM", "VRAM", "eviction", "fragmentation"],
    },
    {
        "id": "GPU-003",
        "fault_type": "gpu",
        "symptom": "drm_sched job timeout with no progress",
        "root_cause": "page table fault in GPU VM",
        "difficulty": "hard",
        "template_sft": (
            "GPU page fault: VM fault on {gpu} at address {fault_addr} "
            "from ring {ring}. "
            "Process {comm} mapping: {mapping}. "
            "VM status: {vm_state}. Diagnose."
        ),
        "template_dpo_chosen": (
            "GPU VM page fault causing GPU hang. Root cause: the "
            "GPU tried to access unmapped memory at {fault_addr} "
            "in the VM space of process {comm}. The fault could not "
            "be serviced, causing the GPU to hang. Fix: check for "
            "buffer misalignment in the application, validate that "
            "all GPU buffers are properly mapped before submission. "
            "Use umr or AMDGPU's GPUVM debugfs to inspect mappings."
        ),
        "template_dpo_rejected": (
            "GPU memory corrupted. Reflash GPU firmware."
        ),
        "expected_patterns": ["page fault", "VM", "mapping", "GPUVM"],
    },
    # ── Security ──────────────────────────────────────────────────────────
    {
        "id": "SEC-001",
        "fault_type": "security",
        "symptom": "SELinux denial blocking legitimate application",
        "root_cause": "Missing SELinux policy for new daemon",
        "difficulty": "medium",
        "template_sft": (
            "SELinux denial for {comm} ({pid}): 'avc: denied "
            "{permission} for {class} comm={comm} scontext={scontext} "
            "tcontext={tcontext}'. "
            "Enforcing mode: {enforcing}. "
            "Application: {app_name}. Diagnose."
        ),
        "template_dpo_chosen": (
            "SELinux policy missing for {app_name}. Root cause: the "
            "application running in context {scontext} needs "
            "{permission} access to {class} with context {tcontext}, "
            "which is not allowed by the current policy. "
            "Fix: use 'audit2allow -a' to generate the required "
            "policy module, or install the correct selinux-policy "
            "package. Temporarily: setenforce 0 for testing."
        ),
        "template_dpo_rejected": (
            "SELinux is broken. Disable SELinux entirely."
        ),
        "expected_patterns": ["SELinux", "AVC", "denied", "audit2allow"],
    },
    {
        "id": "SEC-002",
        "fault_type": "security",
        "symptom": "AppArmor profile prevents normal system operations",
        "root_cause": "AppArmor profile too restrictive for {binary}",
        "difficulty": "easy",
        "template_sft": (
            "AppArmor DENIED operation {operation} for {binary} "
            "({pid}). Profile: {profile_name}. "
            "Requested: {resource}. "
            "Mode: {mode}. Diagnose."
        ),
        "template_dpo_chosen": (
            "AppArmor denial. Root cause: profile '{profile_name}' "
            "for {binary} does not include permission for "
            "{operation} on {resource}. "
            "Fix: use 'aa-logprof' to generate the new rule, or "
            "manually edit /etc/apparmor.d/{profile_name} to add "
            "'{allow_rule}'. Set profile to 'complain' mode for "
            "debugging: 'aa-complain {profile_name}'."
        ),
        "template_dpo_rejected": (
            "AppArmor is broken. Disable AppArmor."
        ),
        "expected_patterns": ["AppArmor", "DENIED", "aa-logprof", "profile"],
    },
    {
        "id": "SEC-003",
        "fault_type": "security",
        "symptom": "kernel NULL pointer in io_uring submission path",
        "root_cause": "io_uring SQPOLL with improperly validated user params",
        "difficulty": "hard",
        "template_sft": (
            "NULL pointer dereference in io_uring/{function}. "
            "IORING_SETUP_SQPOLL enabled. "
            "SQRING: entries={sq_entries}, fd={sq_fd}. "
            "User parameters: {params}. Diagnose."
        ),
        "template_dpo_chosen": (
            "io_uring exploit/vulnerability. Root cause: {function} "
            "does not validate the SQ ring parameters when SQPOLL is "
            "enabled, allowing a user to pass crafted parameters that "
            "cause a NULL pointer dereference. Fix: add parameter "
            "validation in io_uring_create() for SQ ring size and "
            "mmap offsets. Apply kernel CVE-{cve} patch."
        ),
        "template_dpo_rejected": (
            "Kernel bug in io_uring. Disable io_uring."
        ),
        "expected_patterns": ["io_uring", "SQPOLL", "NULL", "validation"],
    },
    {
        "id": "SEC-004",
        "fault_type": "security",
        "symptom": "/proc/pid/mem allows unauthorized writes",
        "root_cause": "PTRACE_MODE_ATTACH_FSCREDS check missing",
        "difficulty": "medium",
        "template_sft": (
            "Process {exploit_pid} wrote to /proc/{target_pid}/mem "
            "of a process owned by different user. "
            "Capabilities: {cap_effective}. "
            "LSM: {lsm}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Ptrace permission bypass in /proc/pid/mem. Root cause: "
            "the kernel's /proc/{target_pid}/mem write path does not "
            "properly call ptrace_access_check() or the LSM hooks, "
            "allowing {exploit_pid} to write to another process's "
            "memory. Fix: apply security patch that adds "
            "ptrace_may_access() check to the write path in "
            "fs/proc/base.c. Ensure CONFIG_SECURITY=y."
        ),
        "template_dpo_rejected": (
            "Not a security issue. The user has the right to debug."
        ),
        "expected_patterns": ["ptrace", "/proc/pid/mem", "access check", "LSM"],
    },
    {
        "id": "SEC-005",
        "fault_type": "security",
        "symptom": "kernel module loaded from unprivileged user namespace",
        "root_cause": "Missing CAP_SYS_MODULE check in user_ns operations",
        "difficulty": "hard",
        "template_sft": (
            "Kernel module {module} loaded from user namespace "
            "(UID {uid}) by process {comm}. "
            "User namespace capabilities: {capabilities}. "
            "Kernel lockdown: {lockdown}. Diagnose."
        ),
        "template_dpo_chosen": (
            "User namespace module loading bypass. Root cause: kernel "
            "allows modules to be loaded from within a user namespace "
            "without proper capability checking against the init "
            "namespace's CAP_SYS_MODULE. Fix: update kernel to enforce "
            "that module loading requires init_ns CAP_SYS_MODULE. "
            "Set kernel.modules_disabled=1 in sysctl for additional "
            "protection."
        ),
        "template_dpo_rejected": (
            "User namespace is a kernel feature. Not a vulnerability."
        ),
        "expected_patterns": ["user namespace", "CAP_SYS_MODULE", "modules_disabled", "lockdown"],
    },
    # ── Performance ───────────────────────────────────────────────────────
    {
        "id": "PERF-001",
        "fault_type": "performance",
        "symptom": "high ksoftirqd usage with no network load",
        "root_cause": "Unmasked MSI interrupts from misconfigured device",
        "difficulty": "medium",
        "template_sft": (
            "ksoftirqd/{cpu} at {cpu_util}% CPU. "
            "/proc/interrupts: {irq_stats}. "
            "Device: {device} with MSI enabled. "
            "IRQ affinity: {affinity}. Diagnose."
        ),
        "template_dpo_chosen": (
            "ksoftirqd CPU spike from interrupt storm. Root cause: "
            "device {device} generates excessive MSI interrupts "
            "because interrupt coalescing is not enabled, or the "
            "device's MSI vector table is misconfigured. "
            "Fix: enable coalescing via device-specific sysfs "
            "params, check IRQ affinity (should be spread across "
            "CPUs), or switch to MSI-X with proper vector count."
        ),
        "template_dpo_rejected": (
            "CPU is overloaded. Reduce process count."
        ),
        "expected_patterns": ["ksoftirqd", "interrupt", "coalescing", "affinity"],
    },
    {
        "id": "PERF-002",
        "fault_type": "performance",
        "symptom": "high %sys CPU from page table operations",
        "root_cause": "Excessive fork rate or munmap/mmap churn",
        "difficulty": "easy",
        "template_sft": (
            "%sys CPU at {sys_cpu}% for {duration}s. "
            "Page fault rate: {pgfault}/s. "
            "Context switches: {cs}/s. "
            "Processes created/s: {fork_rate}. "
            "Top syscall: {top_syscall}. Diagnose."
        ),
        "template_dpo_chosen": (
            "High sys CPU from fork/mmap churn. Root cause: "
            "{fork_rate} processes/s are being forked, each causing "
            "page table duplication and TLB flushes. {top_syscall} "
            "is the bottleneck. Fix: use threading instead of "
            "forking, preallocate memory pools, or switch to "
            "vfork()+exec() pattern. Use 'perf top' to identify "
            "the specific kernel function bottleneck."
        ),
        "template_dpo_rejected": (
            "CPU is too slow. Upgrade to faster processor."
        ),
        "expected_patterns": ["sys CPU", "fork", "page fault", "TLB"],
    },
    {
        "id": "PERF-003",
        "fault_type": "performance",
        "symptom": "cgroup cpu throttle despite low system load",
        "root_cause": "cpu.cfs_period_us / cpu.cfs_quota_us misconfiguration",
        "difficulty": "easy",
        "template_sft": (
            "Cgroup {cg} throttle stats: {nr_throttled} periods "
            "throttled ({throttled_us} us) in {elapsed}s. "
            "cpu.cfs_quota_us={quota}, cpu.cfs_period_us={period}. "
            "Number of CPUs: {ncpus}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Cgroup CPU throttling misconfiguration. Root cause: "
            "cpu.cfs_quota_us={quota} is set to less than "
            "{period} * {ncpus} = {total_capacity}, causing the "
            "cgroup to be throttled even under moderate load. "
            "Fix: set cpu.cfs_quota_us to -1 for unlimited, or "
            "set to {ncpus * period} * {target_utilization} "
            "to allow {target_utilization*100}% CPU utilization."
        ),
        "template_dpo_rejected": (
            "Not enough CPU cores. Add more CPUs."
        ),
        "expected_patterns": ["cfs_quota", "throttled", "cgroup", "period"],
    },
    {
        "id": "PERF-004",
        "fault_type": "performance",
        "symptom": "NUMA remote memory access dominates",
        "root_cause": "Process memory not bound to local NUMA node",
        "difficulty": "medium",
        "template_sft": (
            "NUMA node {node}: {local}% local, {remote}% remote "
            "memory accesses. Process {comm} ({pid}) running on "
            "node {run_node}, memory allocated on node "
            "{mem_node}. "
            "numactl --hardware: {numa_topology}. Diagnose."
        ),
        "template_dpo_chosen": (
            "NUMA remote memory access penalty. Root cause: process "
            "{comm} runs on NUMA node {run_node} but its memory is "
            "allocated on node {mem_node}. Each memory access "
            "traverses the interconnect, adding latency and reducing "
            "bandwidth. Fix: use 'numactl --membind {run_node} "
            "--cpunodebind {run_node}' to bind both CPU and memory. "
            "For long-running processes, enable automatic NUMA "
            "balancing: 'echo 1 > /proc/sys/kernel/numa_balancing'."
        ),
        "template_dpo_rejected": (
            "Memory bottleneck. Add faster RAM."
        ),
        "expected_patterns": ["NUMA", "remote access", "numactl", "membind"],
    },
    {
        "id": "PERF-005",
        "fault_type": "performance",
        "symptom": "high iowait from readahead thrashing",
        "root_cause": "readahead window too large for random I/O pattern",
        "difficulty": "medium",
        "template_sft": (
            "High iowait ({iowait}%) during {workload}. "
            "Readahead: {ra_kb} KB. "
            "I/O size: {io_size} KB (mostly random). "
            "Actual read throughput: {read_mbps} MB/s. "
            "Device: {device}. Diagnose."
        ),
        "template_dpo_chosen": (
            "Read-ahead thrashing. Root cause: the kernel readahead "
            "({ra_kb} KB) fetches {ra_kb/io_size}x more data than "
            "needed per I/O because the workload is random, not "
            "sequential. This wastes IOPS and fills page cache "
            "with unused pages. Fix: reduce readahead with "
            "'blockdev --setra {new_ra} {device}', or use "
            "POSIX_FADV_RANDOM in the application."
        ),
        "template_dpo_rejected": (
            "Storage is too slow. Upgrade to faster drive."
        ),
        "expected_patterns": ["readahead", "thrashing", "random I/O", "iowait"],
    },
]


def get_template(template_id: str) -> dict:
    """Get a specific template by ID."""
    for t in FAULT_TEMPLATES:
        if t["id"] == template_id:
            return t
    raise KeyError(f"Template {template_id} not found")


def get_templates_by_fault_type(fault_type: str) -> list:
    """Get all templates for a given fault type."""
    return [t for t in FAULT_TEMPLATES if t["fault_type"] == fault_type]


def get_templates_by_difficulty(difficulty: str) -> list:
    """Get all templates at a given difficulty level."""
    return [t for t in FAULT_TEMPLATES if t["difficulty"] == difficulty]


def list_fault_types() -> list:
    """Get list of all unique fault types."""
    return list(set(t["fault_type"] for t in FAULT_TEMPLATES))


def count_templates() -> dict:
    """Return count of templates by fault type."""
    counts = {}
    for t in FAULT_TEMPLATES:
        counts[t["fault_type"]] = counts.get(t["fault_type"], 0) + 1
    return counts


if __name__ == "__main__":
    print(f"Total templates: {len(FAULT_TEMPLATES)}")
    for ft, cnt in sorted(count_templates().items()):
        print(f"  {ft}: {cnt}")