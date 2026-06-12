#!/usr/bin/env python3
"""
MominoMoE-v3 broad training data generator.
8 categories, 25,000 train + 2,500 val samples.
Every response is direct — no thinking narration.
"""
import random, json, argparse, os

rng = random.Random(42)

# ─── helpers ──────────────────────────────────────────────────────────────────

def rand_pid():   return rng.randint(1, 65535)
def rand_tid():   return rng.randint(1, 512)
def rand_addr():  return hex(rng.randint(0x400000, 0xFFFF800000000000))
def rand_fd():    return rng.randint(3, 255)
def rand_port():  return rng.randint(1024, 65535)
def rand_ip():    return ".".join(str(rng.randint(1,254)) for _ in range(4))
def rand_path():
    dirs = ["/bin", "/usr/bin", "/var/log", "/tmp", "/home/user", "/etc", "/proc", "/dev"]
    files = ["sh", "ls", "cat", "nginx", "sshd", "bash", "python3", "app", "daemon"]
    return f"{rng.choice(dirs)}/{rng.choice(files)}"

TOOLS_HEADER = """Available tools (respond with ONLY a single JSON tool call, nothing else):
{"tool": "read_file",     "args": {"path": "<str>"}}
{"tool": "write_file",    "args": {"path": "<str>", "content": "<str>"}}
{"tool": "exec_shell",    "args": {"cmd": "<str>"}}
{"tool": "kill_process",  "args": {"pid": <int>, "signal": <int>}}
{"tool": "list_dir",      "args": {"path": "<str>"}}
{"tool": "get_proc_info", "args": {"pid": <int>}}
{"tool": "net_connect",   "args": {"host": "<str>", "port": <int>}}
{"tool": "read_syslog",   "args": {"lines": <int>}}"""

# ─── 1. Kernel fault diagnosis (kept from v2) ─────────────────────────────────

FAULT_NAMES = {
    14: "Page Fault", 13: "General Protection Fault", 8: "Double Fault",
    0: "Divide by Zero", 6: "Invalid Opcode", 11: "Segment Not Present",
    12: "Stack Segment Fault",
}
ERR_DESCS = {
    0x0: "read, page not present, kernel",
    0x2: "write, page not present, kernel",
    0x4: "read, page not present, user",
    0x6: "write, page not present, user",
    0x3: "read, protection violation, kernel",
    0x7: "write, protection violation, user",
}

def make_kernel_fault():
    vector = rng.choices([14, 13, 8, 0, 6, 11, 12], weights=[30,20,5,10,8,12,15])[0]
    fault_name = FAULT_NAMES[vector]
    err = rng.choice(list(ERR_DESCS.keys()))
    rip = rand_addr()
    cr2 = hex(rng.choice([0x0, 0x8, 0x10, 0x18, rng.randint(0x1, 0x40)]))
    tid = rand_tid()
    rax = hex(rng.randint(0, 0xFFFFFFFF))
    rdi = cr2
    rsp = rand_addr()

    syscalls = rng.sample([
        f"SYS_OPEN {rand_path()} 0 -> {rand_fd()}",
        f"SYS_READ {rand_fd()} 4096 -> 4096",
        f"SYS_MMAP 0x0 4096 3 -> {rand_addr()}",
        f"SYS_WRITE {rand_fd()} 128 -> 128",
        f"SYS_CLOSE {rand_fd()} -> 0",
        f"SYS_SPAWN {rand_path()} -> {rand_pid()}",
    ], k=rng.randint(2, 4))

    log_lines = rng.sample([
        "[VFS] opened " + rand_path(),
        f"[SCHED] thread {tid} running",
        f"[VMM] mapped {rand_addr()}->{rand_addr()}",
        f"[NET] connect to {rand_ip()}:{rand_port()}",
        "[IRQ] timer tick",
        "[FS] cache miss",
    ], k=rng.randint(2, 4))

    prompt = (
        f"[FAULT] vector={vector} ({fault_name}) err=0x{err:04X} rip={rip} cr2={cr2} tid={tid} cwd=/bin\n\n"
        f"[REGISTERS] rax={rax} rdi={rdi} rsi=0x100 rsp={rsp}\n\n"
        f"[RECENT_SYSCALLS]\n" + "".join(f"  {s}\n" for s in syscalls) + "\n"
        f"[LOG]\n" + "".join(f"  {l}\n" for l in log_lines) + "\n"
        f"[QUERY] Diagnose this fault and suggest a corrective action."
    )

    # Build diagnosis
    is_null = int(cr2, 16) < 0x100
    err_desc = ERR_DESCS.get(err, "unknown access type")

    if vector == 14:
        if is_null:
            fault_type = "Null pointer dereference"
            root_cause = (f"CR2={cr2} is near-null ({err_desc}). A pointer was not initialized "
                          f"or was freed before use. The fault occurred at RIP={rip}.")
            action = "Add a NULL check before dereferencing at the call site. Inspect recent allocations for use-after-free."
        else:
            fault_type = "Invalid memory access"
            root_cause = (f"CR2={cr2} is not mapped ({err_desc}). The address may be a dangling pointer "
                          f"or stack corruption. RIP={rip}.")
            action = "Validate pointer bounds before access. Check stack canaries for corruption."
    elif vector == 13:
        fault_type = "General Protection Fault"
        root_cause = f"Privilege violation or segment limit exceeded at RIP={rip}. err=0x{err:04X} indicates {err_desc}."
        action = "Verify segment selectors and privilege rings. Check for stack overflow or corrupted return address."
    elif vector == 0:
        fault_type = "Divide by Zero"
        root_cause = f"Integer division by zero at RIP={rip}. RAX={rax} was the dividend."
        action = "Add a divisor != 0 guard before the division instruction at RIP."
    elif vector == 6:
        fault_type = "Invalid Opcode"
        root_cause = f"CPU encountered an undefined instruction at RIP={rip}. Possible memory corruption or wrong code path."
        action = "Verify binary integrity. Check for stack/heap corruption that may have overwritten code."
    elif vector == 8:
        fault_type = "Double Fault"
        root_cause = f"Exception occurred while handling another exception. Stack likely exhausted. RSP={rsp}."
        action = "Increase kernel stack size. Inspect interrupt handlers for unbounded recursion."
    elif vector == 11:
        fault_type = "Segment Not Present"
        root_cause = f"Segment descriptor not present at RIP={rip}. Segment selector invalid or GDT corrupted."
        action = "Reload GDT. Validate segment selectors used in context switch."
    else:
        fault_type = "Stack Segment Fault"
        root_cause = f"Stack segment fault at RIP={rip}. Stack pointer RSP={rsp} may be invalid or outside segment."
        action = "Check for stack overflow. Ensure RSP is properly aligned and within the stack segment."

    response = (
        f"Fault type: {fault_type}.\n"
        f"Root cause: {root_cause}\n"
        f"err=0x{err:04X} decodes as: {err_desc}.\n"
        f"Corrective action: {action}"
    )
    return {"prompt": prompt, "response": response}


# ─── 2. Single-step tool calls ─────────────────────────────────────────────────

TOOL_CALL_TASKS = [
    # (prompt_template, response_template)
    (
        "Read the contents of {path}.",
        '{{"tool": "read_file", "args": {{"path": "{path}"}}}}'
    ),
    (
        "List the files in the directory {path}.",
        '{{"tool": "list_dir", "args": {{"path": "{path}"}}}}'
    ),
    (
        "Process {pid} is hung and not responding. Kill it with SIGKILL.",
        '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 9}}}}'
    ),
    (
        "Process {pid} is using too much memory. Send it SIGTERM to request a graceful shutdown.",
        '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 15}}}}'
    ),
    (
        "Get information about process {pid}.",
        '{{"tool": "get_proc_info", "args": {{"pid": {pid}}}}}'
    ),
    (
        "Run the command: {cmd}",
        '{{"tool": "exec_shell", "args": {{"cmd": "{cmd}"}}}}'
    ),
    (
        "Write '{content}' to {path}.",
        '{{"tool": "write_file", "args": {{"path": "{path}", "content": "{content}"}}}}'
    ),
    (
        "Connect to {host} on port {port}.",
        '{{"tool": "net_connect", "args": {{"host": "{host}", "port": {port}}}}}'
    ),
    (
        "Show the last {lines} lines of the system log.",
        '{{"tool": "read_syslog", "args": {{"lines": {lines}}}}}'
    ),
    (
        "Check what process {pid} has open for file descriptors.",
        '{{"tool": "list_dir", "args": {{"path": "/proc/{pid}/fd"}}}}'
    ),
    (
        "Read the kernel messages from {path}.",
        '{{"tool": "read_file", "args": {{"path": "{path}"}}}}'
    ),
    (
        "Run a memory check: execute 'cat /proc/{pid}/status'.",
        '{{"tool": "exec_shell", "args": {{"cmd": "cat /proc/{pid}/status"}}}}'
    ),
]

SHELL_CMDS = [
    "df -h", "free -m", "ps aux", "top -bn1", "netstat -tulpn",
    "journalctl -n 100", "dmesg | tail -50", "lsof -p {pid}",
    "strace -p {pid}", "ls -la {path}", "cat /proc/meminfo",
]

def make_tool_call_single():
    tmpl_p, tmpl_r = rng.choice(TOOL_CALL_TASKS)
    pid = rand_pid()
    path = rng.choice(["/etc/passwd", "/etc/os-release", "/var/log/syslog",
                       "/proc/meminfo", "/etc/hosts", f"/proc/{pid}/maps",
                       f"/proc/{pid}/status", "/sys/kernel/debug/tracing/trace"])
    cmd = rng.choice(SHELL_CMDS).format(pid=pid, path=path)
    host = rand_ip()
    port = rand_port()
    lines = rng.choice([50, 100, 200, 500])
    content = rng.choice(["enabled=1", "debug=true", "max_connections=100", "0"])

    prompt = TOOLS_HEADER + "\n\n" + tmpl_p.format(
        pid=pid, path=path, cmd=cmd, host=host, port=port, lines=lines, content=content)
    response = tmpl_r.format(
        pid=pid, path=path, cmd=cmd, host=host, port=port, lines=lines, content=content)
    return {"prompt": prompt, "response": response}


# ─── 3. Multi-step tool call sequences ────────────────────────────────────────

MULTI_STEP_TASKS = [
    {
        "prompt": "A process at PID {pid} is consuming 95% CPU. First check its info, then kill it.",
        "steps": [
            '{{"tool": "get_proc_info", "args": {{"pid": {pid}}}}}',
            '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 9}}}}',
        ],
        "reasoning": [
            "Step 1: Inspect process {pid} before taking action.",
            "Step 2: Kill process {pid} with SIGKILL (signal 9) since it is unresponsive.",
        ]
    },
    {
        "prompt": "Diagnose disk usage: list /var/log, then read the largest log file.",
        "steps": [
            '{{"tool": "list_dir", "args": {{"path": "/var/log"}}}}',
            '{{"tool": "read_file", "args": {{"path": "/var/log/syslog"}}}}',
        ],
        "reasoning": [
            "Step 1: List /var/log to find what files are present.",
            "Step 2: Read /var/log/syslog as the primary system log.",
        ]
    },
    {
        "prompt": "Check if sshd is running by reading /proc, then get its process info.",
        "steps": [
            '{{"tool": "exec_shell", "args": {{"cmd": "pgrep sshd"}}}}',
            '{{"tool": "get_proc_info", "args": {{"pid": {pid}}}}}',
        ],
        "reasoning": [
            "Step 1: Use pgrep to find sshd's PID.",
            "Step 2: Get detailed info on the discovered process.",
        ]
    },
    {
        "prompt": "A network connection to {host}:{port} timed out. Check connectivity then read the network config.",
        "steps": [
            '{{"tool": "net_connect", "args": {{"host": "{host}", "port": {port}}}}}',
            '{{"tool": "read_file", "args": {{"path": "/etc/hosts"}}}}',
        ],
        "reasoning": [
            "Step 1: Attempt to connect to {host}:{port} to verify reachability.",
            "Step 2: Read /etc/hosts to check for local hostname resolution issues.",
        ]
    },
    {
        "prompt": "The system is low on memory. Check memory stats, then find the biggest process.",
        "steps": [
            '{{"tool": "read_file", "args": {{"path": "/proc/meminfo"}}}}',
            '{{"tool": "exec_shell", "args": {{"cmd": "ps aux --sort=-%mem | head -5"}}}}',
        ],
        "reasoning": [
            "Step 1: Read /proc/meminfo to see current memory usage.",
            "Step 2: List the top 5 memory-consuming processes.",
        ]
    },
]

def make_tool_call_multi():
    task = rng.choice(MULTI_STEP_TASKS)
    pid = rand_pid()
    host = rand_ip()
    port = rand_port()

    prompt = TOOLS_HEADER + "\n\n" + task["prompt"].format(pid=pid, host=host, port=port)
    steps_fmt = [s.format(pid=pid, host=host, port=port) for s in task["steps"]]
    reasons_fmt = [r.format(pid=pid, host=host, port=port) for r in task["reasoning"]]

    lines = []
    for i, (step, reason) in enumerate(zip(steps_fmt, reasons_fmt), 1):
        lines.append(f"Step {i}: {reason.split(': ', 1)[-1]}")
        lines.append(f"Tool call: {step}")
    response = "\n".join(lines)
    return {"prompt": prompt, "response": response}


# ─── 4. Shell command generation ──────────────────────────────────────────────

SHELL_TASKS = [
    ("Find all files larger than {size}MB under {path} and delete them.",
     "find {path} -type f -size +{size}M -delete"),
    ("Count the number of lines in {path}.",
     "wc -l {path}"),
    ("Show the 10 largest files in {path}.",
     "du -ah {path} | sort -rh | head -10"),
    ("Find all running processes owned by user {user}.",
     "ps aux | grep {user} | grep -v grep"),
    ("Show all open TCP connections.",
     "ss -tnp"),
    ("Continuously monitor CPU and memory of process {pid}.",
     "watch -n 1 'ps -p {pid} -o pid,pcpu,pmem,comm'"),
    ("Tail the last 100 lines of /var/log/syslog and follow new output.",
     "tail -n 100 -f /var/log/syslog"),
    ("Find all files modified in the last 24 hours under /etc.",
     "find /etc -mtime -1 -type f"),
    ("Kill all processes matching the name {proc_name}.",
     "pkill -9 {proc_name}"),
    ("Show disk usage of each directory under /var, sorted by size.",
     "du -sh /var/* | sort -rh"),
    ("Check which process is listening on port {port}.",
     "ss -tlnp sport = :{port}"),
    ("Archive and compress the directory {path} into /tmp/backup.tar.gz.",
     "tar -czf /tmp/backup.tar.gz {path}"),
    ("Show the last 50 kernel messages.",
     "dmesg | tail -50"),
    ("Find all SUID binaries on the system.",
     "find / -perm -4000 -type f 2>/dev/null"),
    ("Show network interface statistics.",
     "ip -s link"),
    ("Check if port {port} is reachable on {host} within 3 seconds.",
     "nc -zv -w3 {host} {port}"),
    ("List all zombie processes.",
     "ps aux | awk '$8 == \"Z\"'"),
    ("Show the top 5 CPU-consuming processes.",
     "ps aux --sort=-%cpu | head -6"),
    ("Recursively change ownership of {path} to {user}.",
     "chown -R {user}:{user} {path}"),
    ("Check disk I/O stats for all devices.",
     "iostat -xz 1 1"),
]

USERS = ["root", "www-data", "nobody", "user", "daemon", "sshd"]
PROC_NAMES = ["nginx", "apache2", "sshd", "python3", "node", "mysqld", "redis-server"]

def make_shell_cmd():
    tmpl_q, tmpl_a = rng.choice(SHELL_TASKS)
    pid = rand_pid()
    size = rng.choice([10, 50, 100, 200, 500])
    path = rng.choice(["/var/log", "/tmp", "/home/user", "/var", "/opt", "/etc"])
    user = rng.choice(USERS)
    host = rand_ip()
    port = rand_port()
    proc_name = rng.choice(PROC_NAMES)

    kw = dict(pid=pid, size=size, path=path, user=user, host=host, port=port, proc_name=proc_name)
    prompt = tmpl_q.format(**kw)
    response = tmpl_a.format(**kw)
    return {"prompt": prompt, "response": response}


# ─── 5. System administration Q&A ─────────────────────────────────────────────

SYSADMIN_QA = [
    ("What does the OOM killer do and when does it trigger?",
     "The OOM (Out-of-Memory) killer is a Linux kernel mechanism that terminates processes when the system "
     "runs out of memory and swap space. It triggers when the kernel cannot satisfy a memory allocation request. "
     "It selects the process with the highest OOM score (based on memory usage, runtime, and priority) and sends "
     "it SIGKILL. You can adjust a process's OOM score with /proc/<pid>/oom_score_adj."),
    ("What is the difference between a process and a thread?",
     "A process is an independent execution unit with its own address space, file descriptors, and resources. "
     "A thread is a lightweight execution unit within a process that shares the process's address space and resources "
     "but has its own stack and register state. Threads communicate via shared memory; processes communicate via IPC "
     "(pipes, sockets, shared memory). In Linux, both are created with clone(); threads share mm_struct."),
    ("What is a zombie process and how do you remove it?",
     "A zombie process is a process that has finished executing but still has an entry in the process table because "
     "its parent has not called wait() to read its exit status. It consumes no CPU or memory beyond the table entry. "
     "To remove it: the parent must call wait() or waitpid(). If the parent is dead, the zombie is reparented to "
     "init/systemd, which reaps it automatically. You cannot kill a zombie directly with SIGKILL."),
    ("Explain the difference between SIGTERM and SIGKILL.",
     "SIGTERM (signal 15) requests a process to terminate gracefully — the process can catch it, flush buffers, "
     "release resources, and exit cleanly. SIGKILL (signal 9) is unconditional: the kernel immediately terminates "
     "the process without giving it any chance to clean up. Use SIGTERM first; escalate to SIGKILL only if the "
     "process doesn't respond. Zombie processes cannot be killed by either — only wait() removes them."),
    ("What is the difference between hard and soft limits in ulimit?",
     "Soft limits are the current enforced limits for a process; they can be raised up to the hard limit by the "
     "process itself or its children. Hard limits are the ceiling — only root can raise them. Set with "
     "'ulimit -S' (soft) and 'ulimit -H' (hard). Common limits: RLIMIT_NOFILE (open files), RLIMIT_NPROC "
     "(max processes), RLIMIT_AS (address space)."),
    ("What is inode exhaustion and how do you diagnose it?",
     "Inode exhaustion occurs when a filesystem runs out of inodes (metadata structures for files) even though "
     "disk space remains. Symptoms: 'No space left on device' despite df showing free space. Diagnose with "
     "'df -i' to check inode usage. Common cause: millions of tiny files (e.g., mail queues, temp files). "
     "Fix: delete unnecessary small files or reformat the partition with more inodes (mkfs.ext4 -N)."),
    ("How does the Linux kernel scheduler decide which process to run next?",
     "Linux uses the Completely Fair Scheduler (CFS) by default. CFS tracks each process's virtual runtime "
     "(vruntime) — how much CPU time it has received, normalized by priority weight. The scheduler always picks "
     "the process with the smallest vruntime (implemented as a red-black tree). Real-time processes (SCHED_FIFO, "
     "SCHED_RR) preempt normal processes. Nice values adjust scheduling weight: lower nice = higher weight = more CPU."),
    ("What is copy-on-write (COW) in the context of fork()?",
     "When fork() creates a child process, the kernel does not immediately copy the parent's memory pages. "
     "Instead, both parent and child share the same physical pages marked read-only. When either process writes "
     "to a page, a page fault triggers and the kernel copies that page for the writer — this is copy-on-write. "
     "This makes fork() fast and memory-efficient; only modified pages are duplicated."),
    ("What is the purpose of /proc/sys/vm/overcommit_memory?",
     "Controls how the kernel handles memory overcommit — allocating more virtual memory than physical RAM+swap. "
     "Values: 0 (heuristic, default — allows reasonable overcommit), 1 (always allow, never check), "
     "2 (strict — refuse allocations exceeding swap + overcommit_ratio% of RAM). Most systems use 0. "
     "Database servers often use 2 to prevent OOM kills. Set via 'sysctl vm.overcommit_memory=<value>'."),
    ("Explain what a context switch is and what it costs.",
     "A context switch is when the CPU stops executing one process/thread and starts executing another. "
     "The kernel saves the current process's register state (including PC, SP, general-purpose registers) "
     "to its kernel stack, then restores the saved state of the next process. Cost: direct CPU cycles "
     "for saving/restoring state (~microseconds), plus indirect cost of TLB flushes and cache eviction "
     "as the new process loads its working set. High context switch rates (>100k/s) indicate CPU saturation."),
]

def make_sysadmin_qa():
    q, a = rng.choice(SYSADMIN_QA)
    return {"prompt": q, "response": a}


# ─── 6. Process and memory debugging ──────────────────────────────────────────

PROC_SCENARIOS = [
    {
        "situation": "Process {pid} ({name}) has been in D state (uninterruptible sleep) for {mins} minutes.",
        "diagnosis": "D state (uninterruptible sleep) means the process is blocked on a kernel I/O operation "
                     "that has not completed. Common causes: NFS hang, a failing disk, or a deadlocked kernel driver. "
                     "Check 'cat /proc/{pid}/wchan' to see which kernel function it is waiting on. "
                     "Check 'dmesg | tail -20' for I/O errors. If caused by NFS, unmount the stale mount. "
                     "A process in D state cannot be killed with SIGKILL; fix the underlying I/O issue.",
    },
    {
        "situation": "The system log shows 'Out of memory: Kill process {pid} ({name}) score {score} or sacrifice child'.",
        "diagnosis": "The OOM killer terminated process {pid} ({name}) because the system ran out of memory. "
                     "OOM score {score} means this process was the highest-scoring candidate (score 0-1000, "
                     "higher = more likely to be killed). Actions: (1) check 'free -m' and 'vmstat' to understand "
                     "memory pressure, (2) identify memory-heavy processes with 'ps aux --sort=-%mem | head', "
                     "(3) increase RAM or swap, (4) set oom_score_adj=-1000 for critical processes to protect them.",
    },
    {
        "situation": "Process {pid} shows {virt}GB virtual memory but only {rss}MB RSS in 'ps aux'.",
        "diagnosis": "High virtual memory (VIRT) with low RSS is normal and expected. VIRT is the total address "
                     "space reserved by the process including memory-mapped files, shared libraries, and heap "
                     "reservations. RSS is the actual physical RAM currently in use. The difference is virtual "
                     "address space that has been reserved but not yet faulted in (due to copy-on-write and "
                     "lazy allocation). This is not a memory leak. Monitor RSS growth over time for actual leaks.",
    },
    {
        "situation": "A process at {pid} is generating {rate} page faults per second according to perf stat.",
        "diagnosis": "High page fault rate at {rate}/s indicates the process is frequently accessing memory "
                     "that is not in its working set. Minor faults (anonymous pages being faulted in) are "
                     "normal during startup. Major faults (pages read from disk) indicate the working set "
                     "exceeds available RAM — the system is thrashing. Fix: increase RAM, reduce working set "
                     "size, or use mlock() to pin critical pages. Check 'vmstat 1' for 'si'/'so' (swap in/out).",
    },
    {
        "situation": "Running 'valgrind --tool=memcheck ./app' shows {leaks} bytes definitely lost at {addr}.",
        "diagnosis": "The application has a confirmed memory leak: {leaks} bytes allocated at address {addr} "
                     "were never freed. Valgrind's 'definitely lost' means no pointer to this memory exists — "
                     "it is unreachable. Action: use '--leak-check=full --show-leak-kinds=all --track-origins=yes' "
                     "for a full stack trace of the allocation site. Fix the allocation to ensure free()/delete "
                     "is called for every allocation path, including error paths.",
    },
]

PROC_NAMES_COMMON = ["nginx", "mysqld", "python3", "node", "java", "redis-server", "postgres"]

def make_proc_debug():
    scenario = rng.choice(PROC_SCENARIOS)
    pid = rand_pid()
    name = rng.choice(PROC_NAMES_COMMON)
    mins = rng.randint(2, 60)
    score = rng.randint(100, 999)
    virt = rng.randint(2, 50)
    rss = rng.randint(50, 500)
    rate = rng.randint(100, 50000)
    addr = rand_addr()
    leaks = rng.randint(1024, 10*1024*1024)

    prompt = scenario["situation"].format(
        pid=pid, name=name, mins=mins, score=score,
        virt=virt, rss=rss, rate=rate, addr=addr, leaks=leaks)
    response = scenario["diagnosis"].format(
        pid=pid, name=name, mins=mins, score=score,
        virt=virt, rss=rss, rate=rate, addr=addr, leaks=leaks)
    return {"prompt": f"Diagnose this situation: {prompt}", "response": response}


# ─── 7. Log analysis ──────────────────────────────────────────────────────────

LOG_SCENARIOS = [
    {
        "log": (
            "Jun 12 03:14:22 host sshd[{pid}]: Failed password for root from {ip} port {port} ssh2\n"
            "Jun 12 03:14:23 host sshd[{pid}]: Failed password for root from {ip} port {port2} ssh2\n"
            "Jun 12 03:14:24 host sshd[{pid}]: Failed password for root from {ip} port {port3} ssh2\n"
            "Jun 12 03:14:25 host sshd[{pid}]: Failed password for root from {ip} port {port4} ssh2\n"
        ),
        "query": "What is happening in this log and what action should be taken?",
        "response": (
            "This log shows a brute-force SSH attack from {ip} targeting the root account — "
            "{count} failed password attempts within 3 seconds from different source ports. "
            "Action: (1) immediately block {ip} with 'iptables -A INPUT -s {ip} -j DROP', "
            "(2) install and configure fail2ban to auto-ban repeated failures, "
            "(3) disable root SSH login in /etc/ssh/sshd_config (PermitRootLogin no), "
            "(4) switch to key-based authentication only (PasswordAuthentication no), "
            "(5) consider moving SSH to a non-standard port."
        ),
    },
    {
        "log": (
            "kernel: EXT4-fs error (device sda1): ext4_validate_block_bitmap:376: comm {proc}: "
            "bg {bg}: bad block bitmap checksum\n"
            "kernel: EXT4-fs (sda1): delayed block allocation failed for inode {inode} at logical offset {offset} "
            "with max blocks 8 with error -5\n"
            "kernel: EXT4-fs (sda1): This should not happen!! Data will be lost\n"
        ),
        "query": "Analyze this kernel log and recommend immediate actions.",
        "response": (
            "Critical filesystem corruption detected on sda1 (EXT4). "
            "Bad block bitmap checksum and delayed allocation failure indicate hardware-level disk errors (error -5 = EIO). "
            "Data loss is imminent or has already occurred. Immediate actions: "
            "(1) backup all data from sda1 NOW before further writes, "
            "(2) run 'smartctl -a /dev/sda' to check disk SMART health, "
            "(3) remount read-only: 'mount -o remount,ro /dev/sda1', "
            "(4) run 'fsck -f /dev/sda1' from a live environment after unmounting, "
            "(5) replace the drive if SMART shows reallocated sectors or uncorrectable errors."
        ),
    },
    {
        "log": (
            "kernel: {proc}[{pid}]: segfault at {addr} ip {rip} sp {rsp} error 4 in libc.so.6\n"
        ),
        "query": "What does this kernel log line mean?",
        "response": (
            "Process {proc} (PID {pid}) crashed with a segmentation fault. "
            "It attempted to read address {addr} (error 4 = read, page not present, user mode). "
            "The fault occurred at instruction pointer {rip} in libc.so.6. "
            "This usually means: (1) a NULL or dangling pointer dereference inside a libc call "
            "(e.g., strlen/memcpy on a bad pointer), or (2) stack corruption that overwrote a return address. "
            "Debug steps: run under gdb or valgrind, check for NULL checks before libc calls, "
            "look for buffer overflows upstream of the crash."
        ),
    },
    {
        "log": (
            "kernel: possible SYN flooding on port {port}. Sending cookies.\n"
            "kernel: possible SYN flooding on port {port}. Sending cookies.\n"
            "kernel: possible SYN flooding on port {port}. Sending cookies.\n"
        ),
        "query": "What is the kernel reporting and what should be done?",
        "response": (
            "The kernel is detecting a SYN flood attack on port {port} — "
            "an attacker is sending many TCP SYN packets without completing the handshake, "
            "exhausting the connection backlog. The kernel has automatically enabled SYN cookies "
            "(a mitigation that avoids storing half-open connections). "
            "Additional actions: (1) SYN cookies are already on — verify with 'sysctl net.ipv4.tcp_syncookies', "
            "(2) increase backlog: 'sysctl -w net.ipv4.tcp_max_syn_backlog=4096', "
            "(3) rate-limit SYN packets with iptables: "
            "'iptables -A INPUT -p tcp --syn --dport {port} -m limit --limit 1/s --limit-burst 4 -j ACCEPT', "
            "(4) consider upstream DDoS protection."
        ),
    },
]

def make_log_analysis():
    scenario = rng.choice(LOG_SCENARIOS)
    pid = rand_pid()
    ip = rand_ip()
    port = rand_port()
    port2, port3, port4 = rand_port(), rand_port(), rand_port()
    proc = rng.choice(PROC_NAMES_COMMON)
    addr = rand_addr()
    rip = rand_addr()
    rsp = rand_addr()
    bg = rng.randint(0, 255)
    inode = rng.randint(10000, 9999999)
    offset = rng.randint(0, 100000)
    count = 4

    kw = dict(pid=pid, ip=ip, port=port, port2=port2, port3=port3, port4=port4,
              proc=proc, addr=addr, rip=rip, rsp=rsp, bg=bg, inode=inode,
              offset=offset, count=count)

    log = scenario["log"].format(**kw)
    query = scenario["query"]
    response = scenario["response"].format(**kw)
    prompt = f"Analyze this system log:\n\n{log}\n{query}"
    return {"prompt": prompt, "response": response}


# ─── 8. Security events ───────────────────────────────────────────────────────

SECURITY_SCENARIOS = [
    {
        "prompt": "A process {pid} ({name}) opened /etc/shadow without being root. How serious is this and what should be done?",
        "response": (
            "This is a critical security incident. /etc/shadow contains hashed passwords for all system users — "
            "reading it is a prerequisite for offline password cracking. A non-root process accessing it indicates "
            "either privilege escalation (the process gained unexpected capabilities) or a misconfigured file permission. "
            "Immediate actions: (1) kill process {pid} immediately, "
            "(2) check its binary for SUID bit: 'stat $(readlink -f /proc/{pid}/exe)', "
            "(3) audit /etc/shadow permissions: should be 640 root:shadow, "
            "(4) rotate all passwords on the system, "
            "(5) check audit log for how {name} gained access: 'ausearch -p {pid}'."
        ),
    },
    {
        "prompt": "netstat shows a process listening on port 4444 with no known service name. PID is {pid}.",
        "response": (
            "Port 4444 is commonly used by Metasploit reverse shells and other malware. "
            "This is suspicious and requires immediate investigation. "
            "Steps: (1) identify the binary: 'ls -la /proc/{pid}/exe', "
            "(2) check its open files: 'ls -la /proc/{pid}/fd', "
            "(3) dump network connections: 'ss -tnp | grep {pid}', "
            "(4) check when the binary was created: 'stat /proc/{pid}/exe', "
            "(5) if confirmed malicious, kill it ('kill -9 {pid}'), block the port with iptables, "
            "and perform a full system audit for persistence mechanisms (crontabs, systemd units, ~/.bashrc)."
        ),
    },
    {
        "prompt": "The audit log shows process {pid} called ptrace() on process {pid2}. Is this normal?",
        "response": (
            "ptrace() allows one process to inspect and control another's execution — it is the foundation of "
            "debuggers (gdb, strace) and some security tools. It is normal if: a debugger is attached, "
            "or a monitoring tool is running. It is suspicious if: the tracing process is unknown, "
            "runs as an unexpected user, or targets a privileged process. "
            "Actions: (1) identify both processes: 'ls -la /proc/{pid}/exe' and '/proc/{pid2}/exe', "
            "(2) check if ptrace is expected (is a debugger session running?), "
            "(3) if unexpected, treat as a potential credential-dumping or injection attack, "
            "(4) enforce ptrace scope: 'sysctl -w kernel.yama.ptrace_scope=1' to restrict to parent-child only."
        ),
    },
    {
        "prompt": "A user ran 'chmod 777 /etc/passwd'. What are the security implications and how do you fix it?",
        "response": (
            "Setting /etc/passwd to world-writable (777) is a critical misconfiguration. "
            "/etc/passwd stores user account information; any user can now modify it, enabling privilege escalation "
            "(e.g., changing root's UID, adding a new root-equivalent user, or clearing passwords). "
            "Fix immediately: (1) restore correct permissions: 'chmod 644 /etc/passwd', "
            "(2) verify ownership: 'chown root:root /etc/passwd', "
            "(3) check if the file was modified while it was 777: 'stat /etc/passwd' for mtime, "
            "(4) diff against backup: 'diff /etc/passwd /etc/passwd-', "
            "(5) audit who else has run commands recently: 'last' and 'journalctl _COMM=chmod'."
        ),
    },
    {
        "prompt": "dmesg shows 'kernel: audit: type=1400 audit(…): apparmor=DENIED operation=exec target=/bin/sh pid={pid}'.",
        "response": (
            "AppArmor denied process {pid} from executing /bin/sh. "
            "This is AppArmor working correctly — the process's profile does not permit shell execution. "
            "This could indicate: (1) a web application or service attempting a shell injection attack "
            "(e.g., command injection via unsanitized input), or (2) a legitimate but misconfigured application. "
            "Actions: (1) identify the process: 'ls -la /proc/{pid}/exe', "
            "(2) if the denial is expected (attack blocked), investigate the application for injection vulnerabilities, "
            "(3) if the denial is a false positive, update the AppArmor profile: 'aa-logprof' to generate a new rule, "
            "(4) never set the profile to permissive mode without understanding the denial."
        ),
    },
]

def make_security_event():
    scenario = rng.choice(SECURITY_SCENARIOS)
    pid = rand_pid()
    pid2 = rand_pid()
    name = rng.choice(PROC_NAMES_COMMON)
    prompt = scenario["prompt"].format(pid=pid, pid2=pid2, name=name)
    response = scenario["response"].format(pid=pid, pid2=pid2, name=name)
    return {"prompt": prompt, "response": response}


# ─── Weighted sampler ─────────────────────────────────────────────────────────

GENERATORS = [
    (make_kernel_fault,    15),  # keep the specialty
    (make_tool_call_single, 25), # biggest category — core new skill
    (make_tool_call_multi,  15),
    (make_shell_cmd,        15),
    (make_sysadmin_qa,      10),
    (make_proc_debug,       10),
    (make_log_analysis,      8),
    (make_security_event,    7), # least, but important
]

_gens, _weights = zip(*GENERATORS)


def generate_dataset(n):
    samples = []
    for _ in range(n):
        gen = rng.choices(_gens, weights=_weights, k=1)[0]
        try:
            samples.append(gen())
        except Exception:
            samples.append(make_kernel_fault())  # fallback
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=25000)
    parser.add_argument("--val",   type=int, default=2500)
    parser.add_argument("--outdir", default="training/data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Generating {args.train} training samples...")
    train = generate_dataset(args.train)
    train_path = os.path.join(args.outdir, "broad_train.jsonl")
    with open(train_path, "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(train)} samples → {train_path}")

    print(f"Generating {args.val} validation samples...")
    rng.seed(99)  # different seed for val
    val = generate_dataset(args.val)
    val_path = os.path.join(args.outdir, "broad_val.jsonl")
    with open(val_path, "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(val)} samples → {val_path}")

    # Category breakdown
    print("\nCategory breakdown (train):")
    cats = {}
    for s in train:
        r = s["response"][:30]
        for label, check in [
            ("kernel_fault",    lambda r: r.startswith("Fault type:")),
            ("tool_single",     lambda r: r.strip().startswith('{"tool"') and "\nStep" not in r),
            ("tool_multi",      lambda r: "Step 1:" in r and "Tool call:" in r),
            ("shell_cmd",       lambda r: r.startswith("find ") or r.startswith("wc ") or r.startswith("du ")
                                          or r.startswith("ps ") or r.startswith("ss ") or r.startswith("watch")
                                          or r.startswith("tail") or r.startswith("pkill") or r.startswith("nc ")
                                          or r.startswith("ip ") or r.startswith("tar") or r.startswith("dmesg")
                                          or r.startswith("iostat") or r.startswith("chown")),
            ("sysadmin_qa",     lambda r: "OOM" in r or "fork" in r or "context switch" in r
                                          or "inode" in r or "ulimit" in r or "zombie" in r
                                          or "SIGTERM" in r or "copy-on-write" in r or "CFS" in r
                                          or "overcommit" in r),
            ("proc_debug",      lambda r: "D state" in r or "OOM killer" in r or "RSS" in r
                                          or "page fault rate" in r or "memory leak" in r),
            ("log_analysis",    lambda r: "brute-force" in r or "corruption" in r or "segmentation fault" in r
                                          or "SYN flood" in r),
            ("security",        lambda r: "AppArmor" in r or "ptrace" in r or "/etc/shadow" in r
                                          or "4444" in r or "chmod 777" in r),
        ]:
            if check(s["response"]):
                cats[label] = cats.get(label, 0) + 1
                break
        else:
            cats["other"] = cats.get("other", 0) + 1

    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(train)
        print(f"  {cat:<20} {count:>6}  ({pct:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
