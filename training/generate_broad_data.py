#!/usr/bin/env python3
"""
MominoMoE-v3 broad training data generator.
12 categories, 50,000 train + 5,000 val samples.
Every response is direct — no thinking narration.
"""
import random, json, argparse, os

rng = random.Random(42)

# ─── helpers ──────────────────────────────────────────────────────────────────

def rand_pid():    return rng.randint(1, 65535)
def rand_tid():    return rng.randint(1, 512)
def rand_addr():   return hex(rng.randint(0x400000, 0xFFFF800000000000))
def rand_fd():     return rng.randint(3, 255)
def rand_port():   return rng.randint(1024, 65535)
def rand_ip():     return ".".join(str(rng.randint(1, 254)) for _ in range(4))
def rand_size_mb():return rng.choice([10, 50, 100, 200, 500, 1024])
def rand_path(kind="any"):
    dirs  = ["/bin", "/usr/bin", "/var/log", "/tmp", "/home/user", "/etc", "/proc", "/dev", "/opt", "/srv"]
    files = ["sh", "ls", "cat", "nginx", "sshd", "bash", "python3", "app", "daemon", "worker"]
    logs  = ["/var/log/syslog", "/var/log/auth.log", "/var/log/kern.log", "/var/log/nginx/error.log"]
    if kind == "log":  return rng.choice(logs)
    if kind == "dir":  return rng.choice(dirs)
    return f"{rng.choice(dirs)}/{rng.choice(files)}"

PROC_NAMES = ["nginx", "mysqld", "python3", "node", "java", "redis-server", "postgres",
              "sshd", "httpd", "mongod", "elasticsearch", "grafana", "prometheus"]
USERS      = ["root", "www-data", "nobody", "user", "daemon", "sshd", "postgres"]
SERVICES   = ["nginx", "mysql", "redis", "ssh", "cron", "docker", "postgresql",
              "mongod", "elasticsearch", "prometheus", "grafana-server", "fail2ban"]
IFACES     = ["eth0", "ens3", "lo", "docker0", "wlan0", "bond0", "br0"]

TOOLS_HEADER = """Available tools (respond with ONLY a single JSON tool call, nothing else):
{"tool": "read_file",      "args": {"path": "<str>"}}
{"tool": "write_file",     "args": {"path": "<str>", "content": "<str>"}}
{"tool": "exec_shell",     "args": {"cmd": "<str>"}}
{"tool": "kill_process",   "args": {"pid": <int>, "signal": <int>}}
{"tool": "list_dir",       "args": {"path": "<str>"}}
{"tool": "get_proc_info",  "args": {"pid": <int>}}
{"tool": "net_connect",    "args": {"host": "<str>", "port": <int>}}
{"tool": "read_syslog",    "args": {"lines": <int>}}
{"tool": "service_ctl",    "args": {"action": "<start|stop|restart|status>", "service": "<str>"}}
{"tool": "get_disk_usage", "args": {"path": "<str>"}}"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. KERNEL FAULT DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════

FAULT_NAMES = {
    14: "Page Fault", 13: "General Protection Fault", 8: "Double Fault",
    0:  "Divide by Zero", 6: "Invalid Opcode", 11: "Segment Not Present",
    12: "Stack Segment Fault", 1: "Debug Exception", 3: "Breakpoint",
}
ERR_DESCS = {
    0x0: "read, page not present, kernel",  0x2: "write, page not present, kernel",
    0x4: "read, page not present, user",    0x6: "write, page not present, user",
    0x3: "read, protection violation, kernel", 0x7: "write, protection violation, user",
}

def make_kernel_fault():
    vector = rng.choices([14,13,8,0,6,11,12], weights=[30,20,5,10,8,12,15])[0]
    fault_name = FAULT_NAMES[vector]
    err  = rng.choice(list(ERR_DESCS.keys()))
    rip  = rand_addr()
    cr2  = hex(rng.choice([0x0, 0x8, 0x10, 0x18, rng.randint(0x1, 0x80)]))
    tid  = rand_tid()
    rax  = hex(rng.randint(0, 0xFFFFFFFF))
    rsp  = rand_addr()

    syscalls = rng.sample([
        f"SYS_OPEN {rand_path()} 0 -> {rand_fd()}",
        f"SYS_READ {rand_fd()} 4096 -> 4096",
        f"SYS_MMAP 0x0 4096 3 -> {rand_addr()}",
        f"SYS_WRITE {rand_fd()} 128 -> 128",
        f"SYS_CLOSE {rand_fd()} -> 0",
        f"SYS_SPAWN {rand_path()} -> {rand_pid()}",
        f"SYS_IOCTL {rand_fd()} 0x5401 -> 0",
        f"SYS_FUTEX {rand_addr()} 0 -> 0",
    ], k=rng.randint(2,5))

    log_lines = rng.sample([
        "[VFS] opened " + rand_path(),
        f"[SCHED] thread {tid} running",
        f"[VMM] mapped {rand_addr()}->{rand_addr()}",
        f"[NET] connect to {rand_ip()}:{rand_port()}",
        "[IRQ] timer tick",
        "[FS] cache miss",
        "[MM] oom score adjusted",
        "[AUDIT] syscall filtered",
    ], k=rng.randint(2,4))

    prompt = (
        f"[FAULT] vector={vector} ({fault_name}) err=0x{err:04X} rip={rip} cr2={cr2} tid={tid} cwd=/bin\n\n"
        f"[REGISTERS] rax={rax} rdi={cr2} rsi=0x100 rsp={rsp}\n\n"
        f"[RECENT_SYSCALLS]\n" + "".join(f"  {s}\n" for s in syscalls) + "\n"
        f"[LOG]\n" + "".join(f"  {l}\n" for l in log_lines) + "\n"
        f"[QUERY] Diagnose this fault and suggest a corrective action."
    )

    is_null   = int(cr2, 16) < 0x100
    err_desc  = ERR_DESCS.get(err, "unknown access")

    if vector == 14:
        if is_null:
            ft = "Null pointer dereference"
            rc = (f"CR2={cr2} is near-null ({err_desc}). A pointer was not initialized or was "
                  f"freed before use. Fault at RIP={rip}.")
            ac = "Add a NULL check before dereferencing at the call site. Inspect recent allocations for use-after-free."
        else:
            ft = "Invalid memory access"
            rc = (f"CR2={cr2} is not mapped ({err_desc}). Address may be a dangling pointer "
                  f"or the result of stack corruption. RIP={rip}.")
            ac = "Validate pointer bounds. Check stack canaries for corruption. Review allocator for double-free."
    elif vector == 13:
        ft = "General Protection Fault"
        rc = f"Privilege violation or segment limit exceeded at RIP={rip}. err=0x{err:04X} = {err_desc}."
        ac = "Verify segment selectors and privilege rings. Check for corrupted return address or ROP chain."
    elif vector == 0:
        ft = "Divide by Zero"
        rc = f"Integer division by zero at RIP={rip}. RAX={rax} was the dividend."
        ac = "Add a divisor != 0 guard before the division instruction at RIP."
    elif vector == 6:
        ft = "Invalid Opcode"
        rc = f"CPU encountered an undefined instruction at RIP={rip}. Possible memory corruption or wrong code path."
        ac = "Verify binary integrity. Check for stack/heap corruption overwriting code. Ensure correct ISA target."
    elif vector == 8:
        ft = "Double Fault"
        rc = f"Exception during exception handling. Stack likely exhausted. RSP={rsp}."
        ac = "Increase kernel stack. Check interrupt handlers for unbounded recursion. Inspect TSS."
    elif vector == 11:
        ft = "Segment Not Present"
        rc = f"Segment descriptor not present at RIP={rip}. GDT may be corrupted."
        ac = "Reload GDT. Validate segment selectors in context switch code."
    else:
        ft = "Stack Segment Fault"
        rc = f"Stack segment fault at RIP={rip}. RSP={rsp} may be invalid or outside segment."
        ac = "Check for stack overflow. Ensure RSP alignment. Validate stack segment in TSS."

    response = (f"Fault type: {ft}.\nRoot cause: {rc}\n"
                f"err=0x{err:04X} decodes as: {err_desc}.\n"
                f"Corrective action: {ac}")
    return {"prompt": prompt, "response": response}


# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE-STEP TOOL CALLS
# ══════════════════════════════════════════════════════════════════════════════

TOOL_CALL_TASKS = [
    ("Read the contents of {path}.",
     '{{"tool": "read_file", "args": {{"path": "{path}"}}}}'),
    ("List the files in the directory {path}.",
     '{{"tool": "list_dir", "args": {{"path": "{path}"}}}}'),
    ("Process {pid} is hung. Kill it with SIGKILL.",
     '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 9}}}}'),
    ("Gracefully stop process {pid} with SIGTERM.",
     '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 15}}}}'),
    ("Reload process {pid} with SIGHUP.",
     '{{"tool": "kill_process", "args": {{"pid": {pid}, "signal": 1}}}}'),
    ("Get information about process {pid}.",
     '{{"tool": "get_proc_info", "args": {{"pid": {pid}}}}}'),
    ("Run the shell command: {cmd}",
     '{{"tool": "exec_shell", "args": {{"cmd": "{cmd}"}}}}'),
    ("Write '{content}' to {path}.",
     '{{"tool": "write_file", "args": {{"path": "{path}", "content": "{content}"}}}}'),
    ("Connect to {host} on port {port}.",
     '{{"tool": "net_connect", "args": {{"host": "{host}", "port": {port}}}}}'),
    ("Show the last {lines} lines of the system log.",
     '{{"tool": "read_syslog", "args": {{"lines": {lines}}}}}'),
    ("Check what file descriptors process {pid} has open.",
     '{{"tool": "list_dir", "args": {{"path": "/proc/{pid}/fd"}}}}'),
    ("Check disk usage at {path}.",
     '{{"tool": "get_disk_usage", "args": {{"path": "{path}"}}}}'),
    ("Start the {service} service.",
     '{{"tool": "service_ctl", "args": {{"action": "start", "service": "{service}"}}}}'),
    ("Stop the {service} service.",
     '{{"tool": "service_ctl", "args": {{"action": "stop", "service": "{service}"}}}}'),
    ("Restart the {service} service.",
     '{{"tool": "service_ctl", "args": {{"action": "restart", "service": "{service}"}}}}'),
    ("Check the status of {service}.",
     '{{"tool": "service_ctl", "args": {{"action": "status", "service": "{service}"}}}}'),
    ("Read /proc/{pid}/maps to see the memory map of process {pid}.",
     '{{"tool": "read_file", "args": {{"path": "/proc/{pid}/maps"}}}}'),
    ("Read the network configuration from {path}.",
     '{{"tool": "read_file", "args": {{"path": "{path}"}}}}'),
    ("Execute 'dmesg | tail -50' to see recent kernel messages.",
     '{{"tool": "exec_shell", "args": {{"cmd": "dmesg | tail -50"}}}}'),
    ("Write the new config value 'net.ipv4.ip_forward=1' to /etc/sysctl.conf.",
     '{{"tool": "write_file", "args": {{"path": "/etc/sysctl.conf", "content": "net.ipv4.ip_forward=1"}}}}'),
]

SHELL_CMDS_SIMPLE = [
    "df -h", "free -m", "ps aux", "top -bn1", "netstat -tulpn", "ss -tnp",
    "journalctl -n 100", "dmesg | tail -50", "lsof -p {pid}", "strace -p {pid}",
    "cat /proc/meminfo", "vmstat 1 5", "iostat -xz 1 1", "uptime",
    "ls -la {path}", "du -sh {path}/*", "find /tmp -mtime +7 -delete",
]

def make_tool_call_single():
    tmpl_p, tmpl_r = rng.choice(TOOL_CALL_TASKS)
    pid = rand_pid(); path = rand_path(); service = rng.choice(SERVICES)
    host = rand_ip(); port = rand_port(); lines = rng.choice([50,100,200,500])
    content = rng.choice(["enabled=1","debug=true","max_connections=100","0","1"])
    cmd = rng.choice(SHELL_CMDS_SIMPLE).format(pid=pid, path=path)
    kw = dict(pid=pid, path=path, service=service, host=host, port=port,
              lines=lines, content=content, cmd=cmd)
    return {"prompt": TOOLS_HEADER + "\n\n" + tmpl_p.format(**kw),
            "response": tmpl_r.format(**kw)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-STEP TOOL SEQUENCES
# ══════════════════════════════════════════════════════════════════════════════

MULTI_STEP_TASKS = [
    {
        "prompt": "Process {pid} is at 95% CPU and unresponsive. Investigate then terminate it.",
        "steps": [
            ('get_proc_info', '{{"pid": {pid}}}', "Inspect process {pid} to confirm it is the culprit."),
            ('kill_process',  '{{"pid": {pid}, "signal": 9}}', "Kill process {pid} with SIGKILL since it is unresponsive."),
        ]
    },
    {
        "prompt": "Disk at /var/log is nearly full. Check usage then clean old logs.",
        "steps": [
            ('get_disk_usage', '{{"path": "/var/log"}}', "Check disk usage at /var/log."),
            ('exec_shell', '{{"cmd": "find /var/log -name \'*.gz\' -mtime +30 -delete"}}', "Delete compressed logs older than 30 days."),
        ]
    },
    {
        "prompt": "The {service} service failed. Check its status then restart it.",
        "steps": [
            ('service_ctl', '{{"action": "status", "service": "{service}"}}', "Check {service} status to understand the failure."),
            ('service_ctl', '{{"action": "restart", "service": "{service}"}}', "Restart {service} to recover from the failure."),
        ]
    },
    {
        "prompt": "Network is down on {iface}. Check system log then bring the interface up.",
        "steps": [
            ('read_syslog', '{{"lines": 100}}', "Read the last 100 syslog lines to find network errors."),
            ('exec_shell', '{{"cmd": "ip link set {iface} up"}}', "Bring interface {iface} up."),
        ]
    },
    {
        "prompt": "Memory is critically low. Find the biggest process then kill it.",
        "steps": [
            ('exec_shell', '{{"cmd": "ps aux --sort=-%mem | head -5"}}', "List top 5 memory consumers."),
            ('kill_process', '{{"pid": {pid}, "signal": 15}}', "Send SIGTERM to process {pid} to request graceful shutdown."),
        ]
    },
    {
        "prompt": "Check if port {port} is open on {host}, then log the result.",
        "steps": [
            ('net_connect', '{{"host": "{host}", "port": {port}}}', "Test connectivity to {host}:{port}."),
            ('write_file', '{{"path": "/var/log/portcheck.log", "content": "checked {host}:{port}"}}', "Log the check result."),
        ]
    },
    {
        "prompt": "Process {pid} is in D state. Read its wait channel then check dmesg.",
        "steps": [
            ('read_file', '{{"path": "/proc/{pid}/wchan"}}', "Read what kernel function process {pid} is blocked on."),
            ('exec_shell', '{{"cmd": "dmesg | tail -30"}}', "Check recent kernel messages for I/O errors."),
        ]
    },
    {
        "prompt": "Suspicious process {pid} opened an unexpected port. Get its info then kill it.",
        "steps": [
            ('get_proc_info', '{{"pid": {pid}}}', "Inspect process {pid} to confirm identity."),
            ('exec_shell', '{{"cmd": "ls -la /proc/{pid}/exe"}}', "Identify the binary behind process {pid}."),
            ('kill_process', '{{"pid": {pid}, "signal": 9}}', "Kill the suspicious process."),
        ]
    },
    {
        "prompt": "Config at {path} may have changed. Read it then restart {service}.",
        "steps": [
            ('read_file', '{{"path": "{path}"}}', "Read the current config at {path}."),
            ('service_ctl', '{{"action": "restart", "service": "{service}"}}', "Restart {service} to apply config changes."),
        ]
    },
    {
        "prompt": "OOM killer fired. Check the log, find the killed process, report disk and memory state.",
        "steps": [
            ('read_syslog', '{{"lines": 200}}', "Read syslog to find OOM killer events."),
            ('exec_shell', '{{"cmd": "free -m"}}', "Check current memory availability."),
            ('get_disk_usage', '{{"path": "/"}}', "Check disk usage to rule out swap exhaustion."),
        ]
    },
]

def make_tool_call_multi():
    task = rng.choice(MULTI_STEP_TASKS)
    pid = rand_pid(); service = rng.choice(SERVICES); iface = rng.choice(IFACES)
    host = rand_ip(); port = rand_port(); path = rand_path()
    kw = dict(pid=pid, service=service, iface=iface, host=host, port=port, path=path)

    prompt = TOOLS_HEADER + "\n\n" + task["prompt"].format(**kw)
    lines = []
    for i, step in enumerate(task["steps"], 1):
        tool, args_tmpl, reason_tmpl = step
        args  = args_tmpl.format(**kw)
        reason = reason_tmpl.format(**kw)
        lines.append(f"Step {i}: {reason}")
        lines.append(f'Tool call: {{"tool": "{tool}", "args": {args}}}')
    return {"prompt": prompt, "response": "\n".join(lines)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHELL COMMAND GENERATION
# ══════════════════════════════════════════════════════════════════════════════

SHELL_TASKS = [
    ("Find all files larger than {size}MB under {path} and delete them.",
     "find {path} -type f -size +{size}M -delete"),
    ("Count the number of lines in {path}.", "wc -l {path}"),
    ("Show the 10 largest files under {path}.", "du -ah {path} | sort -rh | head -10"),
    ("Find all running processes owned by user {user}.", "ps aux | awk '$1==\"{user}\"'"),
    ("Show all open TCP connections with process names.", "ss -tnp"),
    ("Continuously monitor CPU/memory of PID {pid}.", "watch -n 1 'ps -p {pid} -o pid,pcpu,pmem,comm'"),
    ("Tail /var/log/syslog and follow new output.", "tail -n 100 -f /var/log/syslog"),
    ("Find all files modified in the last 24 hours under /etc.", "find /etc -mtime -1 -type f"),
    ("Kill all processes matching the name {proc}.", "pkill -9 {proc}"),
    ("Show disk usage of each subdirectory under {path}, sorted.", "du -sh {path}/* | sort -rh"),
    ("Check which process is listening on port {port}.", "ss -tlnp 'sport = :{port}'"),
    ("Archive {path} into /tmp/backup.tar.gz.", "tar -czf /tmp/backup.tar.gz {path}"),
    ("Show the last 50 kernel messages.", "dmesg | tail -50"),
    ("Find all SUID binaries on the system.", "find / -perm -4000 -type f 2>/dev/null"),
    ("Show network interface statistics.", "ip -s link"),
    ("Check if port {port} is open on {host} within 3 seconds.", "nc -zv -w3 {host} {port}"),
    ("List all zombie processes.", "ps aux | awk '$8==\"Z\"'"),
    ("Show top 5 CPU-consuming processes.", "ps aux --sort=-%cpu | head -6"),
    ("Recursively change ownership of {path} to {user}.", "chown -R {user}:{user} {path}"),
    ("Check disk I/O stats.", "iostat -xz 1 1"),
    ("Show open file handles for process {pid}.", "ls -la /proc/{pid}/fd"),
    ("Count failed SSH login attempts in auth.log.", "grep 'Failed password' /var/log/auth.log | wc -l"),
    ("Show memory map of process {pid}.", "cat /proc/{pid}/maps"),
    ("List all listening ports.", "ss -lntp"),
    ("Check for processes with deleted exe binaries (possible rootkit).", "ls -la /proc/*/exe 2>/dev/null | grep deleted"),
    ("Get the PID of {proc}.", "pgrep {proc}"),
    ("Show environment variables of process {pid}.", "cat /proc/{pid}/environ | tr '\\0' '\\n'"),
    ("Watch network interface {iface} packet rates.", "watch -n 1 'ip -s link show {iface}'"),
    ("Show all active cron jobs.", "crontab -l && ls /etc/cron.*"),
    ("Print the 5 most recently modified files under {path}.", "find {path} -type f -printf '%T@ %p\\n' | sort -rn | head -5 | awk '{{print $2}}'"),
    ("Check swap usage.", "swapon --show && free -h"),
    ("Show which libraries process {pid} has loaded.", "cat /proc/{pid}/maps | awk '{{print $6}}' | grep '\\.so' | sort -u"),
    ("List all Docker containers (running and stopped).", "docker ps -a"),
    ("Show Docker container logs for container {proc}.", "docker logs --tail 100 {proc}"),
    ("Show all failed systemd services.", "systemctl --failed"),
    ("Reload systemd daemon after unit file change.", "systemctl daemon-reload"),
    ("Show disk read/write speeds.", "iostat -dx 1 3"),
    ("Find core dump files on the system.", "find / -name 'core' -o -name 'core.*' 2>/dev/null | head -20"),
    ("Print ARP table.", "arp -n"),
    ("Show routing table.", "ip route show"),
]

def make_shell_cmd():
    tmpl_q, tmpl_a = rng.choice(SHELL_TASKS)
    kw = dict(pid=rand_pid(), size=rand_size_mb(), path=rand_path("dir"),
              user=rng.choice(USERS), host=rand_ip(), port=rand_port(),
              proc=rng.choice(PROC_NAMES), iface=rng.choice(IFACES))
    try:
        return {"prompt": tmpl_q.format(**kw), "response": tmpl_a.format(**kw)}
    except KeyError:
        return {"prompt": tmpl_q, "response": tmpl_a}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYSTEM ADMINISTRATION Q&A
# ══════════════════════════════════════════════════════════════════════════════

SYSADMIN_QA = [
    ("What does the OOM killer do and when does it trigger?",
     "The OOM killer terminates processes when the system runs out of memory and swap. It picks the highest "
     "OOM score process (based on memory use, runtime, priority) and sends SIGKILL. Adjust a process's "
     "vulnerability with /proc/<pid>/oom_score_adj (-1000 = never kill, +1000 = always kill first)."),
    ("What is the difference between a process and a thread?",
     "A process is an independent execution unit with its own address space, file descriptors, and resources. "
     "A thread is a lightweight unit within a process sharing its address space but with its own stack and "
     "registers. In Linux both are created with clone(); threads share mm_struct."),
    ("What is a zombie process and how do you remove it?",
     "A zombie has finished executing but still occupies a process table entry because its parent hasn't "
     "called wait(). It uses no CPU or memory. Fix: parent calls wait() or waitpid(). If the parent is dead "
     "the zombie is reparented to init/systemd which reaps it. You cannot SIGKILL a zombie."),
    ("Explain the difference between SIGTERM and SIGKILL.",
     "SIGTERM (15) requests graceful shutdown — the process can catch it and clean up. SIGKILL (9) is "
     "unconditional: the kernel kills immediately with no cleanup. Always try SIGTERM first. Zombie "
     "processes ignore both — only wait() removes them."),
    ("What is the difference between hard and soft ulimits?",
     "Soft limits are the current enforced ceiling; a process can raise them up to the hard limit. Hard "
     "limits are the absolute ceiling — only root can raise them. Set with ulimit -S (soft) and -H (hard). "
     "Common: RLIMIT_NOFILE (open files), RLIMIT_NPROC (processes), RLIMIT_AS (address space)."),
    ("What is inode exhaustion and how do you diagnose it?",
     "'No space left on device' despite free disk space means inode exhaustion. Diagnose: df -i. "
     "Cause: millions of tiny files (mail queues, temp files). Fix: delete small files or reformat with "
     "more inodes (mkfs.ext4 -N <count>)."),
    ("How does the Linux CFS scheduler work?",
     "Completely Fair Scheduler tracks each task's virtual runtime (vruntime) — CPU time received, "
     "normalized by priority weight. It always runs the task with the lowest vruntime (stored in a "
     "red-black tree). Real-time tasks (SCHED_FIFO/RR) preempt normal tasks. Nice values adjust weight."),
    ("What is copy-on-write (COW) in fork()?",
     "fork() shares parent's memory pages with the child, marked read-only. When either writes, a page "
     "fault triggers and the kernel copies only that page — copy-on-write. Makes fork() fast; only "
     "modified pages are duplicated."),
    ("What is /proc/sys/vm/overcommit_memory?",
     "Controls memory overcommit. 0=heuristic (default), 1=always allow, 2=strict (refuse allocations "
     "exceeding swap + overcommit_ratio% of RAM). Databases often use 2. Set via sysctl."),
    ("What is a context switch and what does it cost?",
     "A context switch saves the current process's registers to its kernel stack, then restores the next "
     "process's saved state. Cost: ~microseconds of direct overhead plus TLB flushes and cache eviction. "
     ">100k/s indicates CPU saturation."),
    ("What is TCP TIME_WAIT and why does it exist?",
     "TIME_WAIT is a TCP state after the connection closes — the endpoint waits 2×MSL (usually 60s) before "
     "releasing the port. Purpose: ensure delayed packets from the old connection don't corrupt a new one "
     "on the same port. High TIME_WAIT counts are normal under heavy connection churn; use SO_REUSEADDR."),
    ("What is the difference between ss and netstat?",
     "Both show network connections. ss is the modern replacement — faster (reads from kernel's sock "
     "diag interface directly), shows more detail, and is actively maintained. netstat reads /proc/net "
     "which is slower on large connection tables. Use ss -tnp for TCP with process names."),
    ("What is a core dump and how do you analyze one?",
     "A core dump is a snapshot of a process's memory at crash time. Enable with 'ulimit -c unlimited'. "
     "Analyze with 'gdb <binary> <corefile>' then 'bt' for backtrace. Location controlled by "
     "/proc/sys/kernel/core_pattern. Useful for debugging segfaults without a live debugger."),
    ("How does Linux swap work?",
     "Swap extends effective RAM by moving cold memory pages to disk. The kernel uses LRU to decide which "
     "pages to swap. Controlled by vm.swappiness (0=avoid swap, 100=aggressive). High swap usage with "
     "disk I/O (si/so in vmstat) means memory pressure — add RAM or reduce working set."),
    ("What is the difference between a hard link and a symbolic link?",
     "A hard link is a directory entry pointing to the same inode as the original file — deleting the "
     "original leaves the hard link intact. A symbolic (soft) link is a file containing a path to another "
     "file — it breaks if the target is deleted. Hard links can't span filesystems or point to directories."),
    ("What does vm.dirty_ratio control?",
     "vm.dirty_ratio is the maximum percentage of system RAM that can contain dirty (unwritten) pages "
     "before the process that dirtied them must write them out. vm.dirty_background_ratio is when "
     "background flushing starts. High ratios = better throughput but more data loss on crash. Default: "
     "dirty_background=10%, dirty=20%."),
    ("Explain huge pages and when to use them.",
     "Huge pages (2MB on x86 vs 4KB normal) reduce TLB pressure for large working sets. Use for: "
     "databases (PostgreSQL, Oracle), JVMs, high-performance networking. Configure via "
     "/proc/sys/vm/nr_hugepages. Transparent huge pages (THP) auto-promote regions — disable for "
     "latency-sensitive apps: echo never > /sys/kernel/mm/transparent_hugepage/enabled."),
    ("What is the Linux virtual filesystem (VFS)?",
     "VFS is an abstraction layer that provides a uniform interface (open, read, write, close) over "
     "different filesystem implementations (ext4, xfs, tmpfs, procfs). Filesystems register with VFS "
     "by providing function pointers (file_operations, inode_operations). This lets the kernel support "
     "any filesystem without changing user-space syscalls."),
    ("What is cgroups and how does it work?",
     "Control groups (cgroups) limit and account for resource use by process groups. Resources: CPU "
     "(cpu.shares, cpu.quota), memory (memory.limit_in_bytes), I/O (blkio.weight), network (tc). "
     "Docker and systemd use cgroups internally. v2 unified hierarchy at /sys/fs/cgroup/."),
    ("What is eBPF and what can it do?",
     "eBPF (extended Berkeley Packet Filter) is a kernel VM that runs sandboxed programs in kernel context "
     "without kernel modules. Used for: network packet filtering (XDP), performance tracing (bpftrace), "
     "security enforcement (seccomp-BPF, Falco). Programs are JIT-compiled and verified safe before loading. "
     "Tools: bcc, bpftrace, Cilium."),
]

def make_sysadmin_qa():
    q, a = rng.choice(SYSADMIN_QA)
    return {"prompt": q, "response": a}


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROCESS AND MEMORY DEBUGGING
# ══════════════════════════════════════════════════════════════════════════════

PROC_SCENARIOS = [
    ("Process {pid} ({name}) has been in D state (uninterruptible sleep) for {mins} minutes.",
     "D state means the process is blocked on a kernel I/O operation that hasn't completed. Common causes: "
     "NFS hang, failing disk, or deadlocked driver. Check 'cat /proc/{pid}/wchan' for the blocking function. "
     "Run 'dmesg | tail -20' for I/O errors. If NFS: unmount the stale mount. Processes in D state cannot "
     "be killed with SIGKILL — fix the underlying I/O issue first."),
    ("The OOM killer logged: Kill process {pid} ({name}) score {score}.",
     "The OOM killer terminated {name} ({pid}) due to memory exhaustion. Score {score} means it was the "
     "highest-priority kill candidate. Actions: (1) check 'free -m' for current pressure, "
     "(2) 'ps aux --sort=-%mem | head' for top consumers, (3) increase RAM or swap, "
     "(4) set oom_score_adj=-1000 on critical processes."),
    ("Process {pid} shows {virt}GB virtual memory but only {rss}MB RSS.",
     "Expected behavior. VIRT is the total reserved address space (includes mmap'd files, shared libs, "
     "heap reservations). RSS is physical RAM actually in use. The gap is virtual space reserved but not "
     "faulted in (lazy allocation + COW). Not a leak. Monitor RSS growth over time for actual leaks."),
    ("Process {pid} generates {rate} page faults per second (from perf stat).",
     "High page fault rate indicates the working set exceeds RAM. Minor faults (anonymous pages) are normal "
     "at startup. Major faults (pages read from disk) = thrashing. Fix: increase RAM, reduce working set, "
     "or use mlock() to pin critical pages. Check 'vmstat 1' for si/so (swap in/out)."),
    ("valgrind --memcheck shows {leaks} bytes definitely lost at {addr}.",
     "Confirmed memory leak: {leaks} bytes at {addr} are unreachable — no pointer exists to free them. "
     "Run with '--leak-check=full --show-leak-kinds=all --track-origins=yes' for full allocation stack trace. "
     "Ensure every allocation path calls free()/delete, including error paths."),
    ("strace shows process {pid} making thousands of stat() calls per second.",
     "The process is hot-polling a file or directory with stat() — likely a busy-wait pattern. This wastes "
     "CPU. Recommend switching to inotify (inotify_add_watch) for file change notifications, which blocks "
     "until a change occurs instead of spinning. Check the application logic for tight polling loops."),
    ("Process {pid} ({name}) keeps restarting every {mins} seconds per systemd logs.",
     "Crash-loop detected: {name} dies within {mins}s of starting. Diagnose: "
     "(1) 'journalctl -u {name} -n 50' for exit reason, "
     "(2) check core dumps: 'coredumpctl list', "
     "(3) run manually outside systemd to see stdout/stderr, "
     "(4) verify config file syntax and required dependencies are available."),
    ("ps shows process {pid} in Z (zombie) state, parent PID is {ppid}.",
     "Zombie process: {pid} finished but parent {ppid} hasn't called wait(). The zombie holds no resources "
     "beyond a process table entry. Fix: send SIGCHLD to parent {ppid} ('kill -CHLD {ppid}') to prompt it "
     "to reap children. If parent is unresponsive, killing parent lets init adopt and reap the zombie. "
     "Cannot kill the zombie directly."),
]

def make_proc_debug():
    tmpl_q, tmpl_a = rng.choice(PROC_SCENARIOS)
    kw = dict(pid=rand_pid(), ppid=rand_pid(), name=rng.choice(PROC_NAMES),
              mins=rng.randint(2,60), score=rng.randint(100,999),
              virt=rng.randint(2,50), rss=rng.randint(50,500),
              rate=rng.randint(100,50000), addr=rand_addr(),
              leaks=rng.randint(1024, 10*1024*1024))
    return {"prompt": "Diagnose: " + tmpl_q.format(**kw), "response": tmpl_a.format(**kw)}


# ══════════════════════════════════════════════════════════════════════════════
# 7. LOG ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

LOG_SCENARIOS = [
    (
        ("Jun 12 03:14:2{i} host sshd[{pid}]: Failed password for root from {ip} port {p} ssh2\n"
         for i in range(4)),
        "What is happening and what action should be taken?",
        "SSH brute-force attack from {ip} targeting root — 4 failures in 3 seconds. "
        "Action: (1) block {ip}: 'iptables -A INPUT -s {ip} -j DROP', "
        "(2) install fail2ban for automatic banning, (3) set 'PermitRootLogin no' in sshd_config, "
        "(4) disable password auth: 'PasswordAuthentication no', (5) consider non-standard SSH port."
    ),
    (
        ["kernel: EXT4-fs error (device sda1): bad block bitmap checksum\n",
         "kernel: EXT4-fs (sda1): delayed block allocation failed for inode {inode} with error -5\n",
         "kernel: EXT4-fs (sda1): This should not happen!! Data will be lost\n"],
        "Analyze this kernel log.",
        "Critical EXT4 filesystem corruption on sda1. Error -5 is EIO (hardware I/O error). Data loss likely. "
        "Immediately: (1) backup data NOW, (2) remount read-only: 'mount -o remount,ro /dev/sda1', "
        "(3) run 'smartctl -a /dev/sda' to check SMART health, (4) fsck from live environment, "
        "(5) replace drive if SMART shows reallocated sectors."
    ),
    (
        ["kernel: {proc}[{pid}]: segfault at {addr} ip {rip} sp {rsp} error 4 in libc.so.6\n"],
        "What does this mean?",
        "Process {proc} (PID {pid}) crashed: read from unmapped address {addr} (error 4 = user-mode read, "
        "page not present) inside libc.so.6 at {rip}. Likely a NULL/dangling pointer passed to a libc "
        "function (strlen, memcpy, etc). Debug: run under gdb or valgrind, check for NULL guards before "
        "libc calls, look for buffer overflows upstream."
    ),
    (
        ["kernel: possible SYN flooding on port {port}. Sending cookies.\n"] * 3,
        "What is the kernel reporting and what should be done?",
        "SYN flood attack on port {port}: attacker sending SYN packets without completing handshakes. "
        "Kernel auto-enabled SYN cookies. Further actions: (1) verify: 'sysctl net.ipv4.tcp_syncookies', "
        "(2) increase backlog: 'sysctl -w net.ipv4.tcp_max_syn_backlog=4096', "
        "(3) rate-limit: 'iptables -A INPUT -p tcp --syn --dport {port} -m limit --limit 1/s -j ACCEPT', "
        "(4) consider upstream DDoS protection."
    ),
    (
        ["kernel: BTRFS error (device {dev}): parent transid verify failed on {block} wanted {t1} found {t2}\n",
         "kernel: BTRFS: error (device {dev}) in btrfs_run_delalloc_range: errno=-5 Error\n"],
        "Diagnose this BTRFS log.",
        "BTRFS transaction ID mismatch on {dev}: corruption or partial write (transid wanted {t1}, found {t2}). "
        "errno=-5 is EIO — storage-level read error. Actions: (1) run 'btrfs scrub start {dev}' to identify "
        "and repair (if RAID), (2) check SMART: 'smartctl -a {dev}', (3) 'btrfs check {dev}' in read-only "
        "mode from live environment, (4) restore from backup if unrecoverable."
    ),
    (
        ["nginx: [error] connect() failed (111: Connection refused) while connecting to upstream, "
         "client: {ip}, server: _, request: \"GET / HTTP/1.1\", upstream: \"http://127.0.0.1:{port}/\"\n"],
        "What is wrong and how do you fix it?",
        "Nginx cannot reach its upstream backend at 127.0.0.1:{port} — connection refused means nothing is "
        "listening on that port. Fix: (1) check if the backend is running: 'ss -tlnp | grep {port}', "
        "(2) start the backend service if down, (3) verify nginx upstream config points to the correct port, "
        "(4) check backend logs for crash reasons."
    ),
]

def make_log_analysis():
    scenario = rng.choice(LOG_SCENARIOS)
    log_lines, query, response_tmpl = scenario
    kw = dict(pid=rand_pid(), ip=rand_ip(), port=rand_port(), rip=rand_addr(), rsp=rand_addr(),
              addr=rand_addr(), proc=rng.choice(PROC_NAMES), inode=rng.randint(10000,9999999),
              dev="/dev/sda", block=rand_addr(), t1=rng.randint(1000,9999), t2=rng.randint(1000,9999))
    if callable(log_lines):
        log = "".join(log_lines)
    elif isinstance(log_lines, list):
        log = "".join(l.format(**kw) for l in log_lines)
    else:
        log = log_lines.format(**kw)
    try:
        log = log.format(**kw)
    except (KeyError, IndexError):
        pass
    try:
        response = response_tmpl.format(**kw)
    except (KeyError, IndexError):
        response = response_tmpl
    return {"prompt": f"Analyze this system log:\n\n{log}\n{query}",
            "response": response}


# ══════════════════════════════════════════════════════════════════════════════
# 8. SECURITY EVENTS
# ══════════════════════════════════════════════════════════════════════════════

SECURITY_QA = [
    ("A non-root process {pid} ({name}) opened /etc/shadow. How serious is this?",
     "/etc/shadow holds hashed passwords — critical incident. Indicates privilege escalation or SUID "
     "misconfiguration. Actions: (1) kill {pid} immediately, (2) check for SUID: 'stat $(readlink -f "
     "/proc/{pid}/exe)', (3) restore permissions: 'chmod 640 /etc/shadow && chown root:shadow /etc/shadow', "
     "(4) rotate all passwords, (5) audit with ausearch."),
    ("netstat shows an unknown process listening on port 4444, PID {pid}.",
     "Port 4444 is common for Metasploit reverse shells. Treat as incident. Steps: "
     "(1) identify binary: 'ls -la /proc/{pid}/exe', (2) check connections: 'ss -tnp | grep {pid}', "
     "(3) check file creation time: 'stat /proc/{pid}/exe', (4) kill: 'kill -9 {pid}', "
     "(5) block port: 'iptables -A INPUT --dport 4444 -j DROP', "
     "(6) audit for persistence: crontabs, systemd units, ~/.bashrc."),
    ("Audit log shows process {pid} called ptrace() on process {pid2}.",
     "ptrace() is used by debuggers and monitoring tools — normal if a known debugger is attached. "
     "Suspicious if unexpected. Actions: (1) identify both: 'ls -la /proc/{pid}/exe' and '/proc/{pid2}/exe', "
     "(2) if unknown: possible credential dumping or code injection, "
     "(3) enforce scope: 'sysctl -w kernel.yama.ptrace_scope=1' (restrict to parent-child only)."),
    ("A user ran 'chmod 777 /etc/passwd'. What are the implications?",
     "Critical misconfiguration: any user can now write /etc/passwd, enabling privilege escalation "
     "(add root-equivalent user, clear password field). Fix immediately: "
     "(1) 'chmod 644 /etc/passwd && chown root:root /etc/passwd', "
     "(2) check for modifications: 'diff /etc/passwd /etc/passwd-', "
     "(3) audit who ran chmod: 'journalctl _COMM=chmod'."),
    ("dmesg shows: apparmor=DENIED operation=exec target=/bin/sh pid={pid}.",
     "AppArmor blocked {pid} from executing /bin/sh — the process profile doesn't allow shell execution. "
     "Could be: shell injection attack blocked, or legitimate app misconfigured. "
     "(1) identify process: 'ls -la /proc/{pid}/exe', "
     "(2) if attack: investigate for command injection vulnerabilities, "
     "(3) if false positive: 'aa-logprof' to update the profile. Never disable enforcement without understanding."),
    ("Find all world-writable files outside /tmp.",
     "Command: find / -not -path '/tmp/*' -not -path '/proc/*' -perm -0002 -type f 2>/dev/null\n"
     "World-writable files outside /tmp are a security risk — any user can modify them. "
     "For each found: check if it's intentional, remove the write bit with 'chmod o-w <file>'."),
    ("A process is writing to /dev/mem. Is this normal?",
     "/dev/mem provides raw access to physical memory — only privileged tools should use it (e.g., X11 on old "
     "systems, firmware tools). In modern Linux, access is restricted by CONFIG_STRICT_DEVMEM. "
     "Unexpected access = serious concern: could be a rootkit or kernel exploit. "
     "Identify: 'lsof /dev/mem', check the binary, isolate the system."),
    ("iptables -L shows no rules but traffic is being filtered. Why?",
     "Several possibilities: (1) nftables is active instead of iptables — check 'nft list ruleset', "
     "(2) ipset rules referenced by iptables are blocking, (3) a different chain or table (mangle/raw) "
     "has rules — check 'iptables -L -t mangle', (4) the kernel is using ebtables for bridge traffic. "
     "Modern systems often use nftables as the backend; iptables commands may translate there."),
]

def make_security_event():
    q_tmpl, a_tmpl = rng.choice(SECURITY_QA)
    kw = dict(pid=rand_pid(), pid2=rand_pid(), name=rng.choice(PROC_NAMES))
    try:
        return {"prompt": q_tmpl.format(**kw), "response": a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt": q_tmpl, "response": a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 9. NETWORK DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

NETWORK_QA = [
    ("The connection to {host}:{port} is timing out. How do I diagnose it?",
     "Step-by-step: (1) 'ping {host}' — is the host reachable at all? (2) 'nc -zv -w3 {host} {port}' — "
     "is port {port} open? (3) 'traceroute {host}' — where does the path die? "
     "(4) 'ss -tnp | grep {port}' — is a local firewall blocking? "
     "(5) 'iptables -L -n' — check for DROP rules. (6) On the server: 'ss -tlnp | grep {port}' to confirm "
     "the service is listening."),
    ("What does 'ESTABLISHED' vs 'CLOSE_WAIT' mean in ss output?",
     "ESTABLISHED: active TCP connection, data can flow. CLOSE_WAIT: remote side sent FIN (wants to close) "
     "but the local application hasn't called close() yet. Many CLOSE_WAIT connections on a server indicate "
     "the application is not closing sockets properly — file descriptor leak. Fix: profile the application "
     "for missing close() calls."),
    ("Explain TCP's three-way handshake.",
     "SYN: client sends segment with SYN flag to initiate connection. "
     "SYN-ACK: server responds with SYN+ACK, allocating a socket. "
     "ACK: client acknowledges — connection is established. "
     "SYN flood attacks overwhelm the server's SYN queue by never sending the final ACK. "
     "Mitigated by SYN cookies (no state stored until ACK received)."),
    ("How do I capture packets on interface {iface} filtering for port {port}?",
     "tcpdump -i {iface} -n 'port {port}' -w /tmp/capture.pcap\n"
     "Flags: -i {iface} (interface), -n (no DNS resolution), 'port {port}' (BPF filter), "
     "-w (write pcap). Analyze with Wireshark or 'tcpdump -r /tmp/capture.pcap'."),
    ("What is SNAT vs DNAT in iptables?",
     "SNAT (Source NAT): rewrites the source IP of outgoing packets — used for masquerading private IPs "
     "behind a public IP (internet sharing). Applied in POSTROUTING chain. "
     "DNAT (Destination NAT): rewrites the destination IP of incoming packets — used for port forwarding "
     "(redirect external port to internal service). Applied in PREROUTING chain."),
    ("Why would I see 'Destination Host Unreachable' vs 'Connection refused'?",
     "'Destination Host Unreachable' is an ICMP error from a router — the packet couldn't reach the host "
     "(routing failure, host down, or ICMP blocked). "
     "'Connection refused' means the host is reachable but nothing is listening on that port (TCP RST "
     "received). Refused = host up, service down. Unreachable = routing or host problem."),
    ("How do I add a static route to {host} via gateway {gw}?",
     "Temporary: ip route add {host}/32 via {gw}\n"
     "Persistent (Debian/Ubuntu): add 'up ip route add {host}/32 via {gw}' to /etc/network/interfaces, "
     "or create /etc/netplan entry. "
     "Verify: 'ip route show | grep {host}'"),
    ("What is the difference between TCP and UDP?",
     "TCP: connection-oriented, reliable, ordered delivery, flow/congestion control, higher overhead. "
     "Use for: HTTP, SSH, databases — anything requiring correct delivery. "
     "UDP: connectionless, no reliability guarantees, lower latency, no handshake overhead. "
     "Use for: DNS, video streaming, VoIP, games — where speed matters more than perfect delivery."),
]

def make_network_diag():
    q_tmpl, a_tmpl = rng.choice(NETWORK_QA)
    kw = dict(host=rand_ip(), port=rand_port(), iface=rng.choice(IFACES),
              gw=rand_ip())
    try:
        return {"prompt": q_tmpl.format(**kw), "response": a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt": q_tmpl, "response": a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 10. SYSTEMD / SERVICE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEMD_QA = [
    ("How do I create a systemd service that runs {name} as {user}?",
     f"Create /etc/systemd/system/{{name}}.service:\n"
     "[Unit]\nDescription={name} service\nAfter=network.target\n\n"
     "[Service]\nUser={user}\nExecStart=/usr/bin/{name}\nRestart=always\nRestartSec=5\n\n"
     "[Install]\nWantedBy=multi-user.target\n\n"
     "Then: systemctl daemon-reload && systemctl enable --now {name}"),
    ("What is the difference between systemctl stop and systemctl disable?",
     "stop: sends SIGTERM to the running service (stops it now, restarts on next boot if enabled). "
     "disable: removes the boot symlink so the service won't start on next boot (doesn't stop it now). "
     "To stop AND prevent restart: systemctl disable --now <service>"),
    ("{service} keeps failing with 'exit code 1'. How do I debug it?",
     "(1) 'journalctl -u {service} -n 100' — read the last 100 log lines for the error message. "
     "(2) 'systemctl status {service}' — see exit code and last lines. "
     "(3) Run the ExecStart command manually as the service user to see stderr directly. "
     "(4) Check 'After=' dependencies: 'systemctl list-dependencies {service}'. "
     "(5) Check file permissions on config files and working directory."),
    ("How do I make a service restart automatically if it crashes?",
     "In [Service] section: Restart=on-failure (or always for any exit). "
     "RestartSec=5 sets the delay. StartLimitBurst=5 and StartLimitIntervalSec=60 prevent infinite "
     "restart loops. After systemctl daemon-reload, the service will auto-restart on crash."),
    ("What does 'Type=notify' mean in a systemd unit?",
     "Type=notify means the service tells systemd when it is fully ready by calling sd_notify(READY=1). "
     "systemd waits for this signal before marking the service 'active'. Use for daemons with a startup "
     "phase (config loading, socket binding). Other types: simple (default, ready immediately), "
     "forking (old-style daemons that fork and exit parent), oneshot (run-to-completion tasks)."),
    ("How do I run a command on a schedule without cron, using systemd?",
     "Create two files:\n"
     "/etc/systemd/system/mytask.service — the command to run (Type=oneshot)\n"
     "/etc/systemd/system/mytask.timer — the schedule:\n"
     "[Timer]\nOnCalendar=daily\nPersistent=true\n[Install]\nWantedBy=timers.target\n\n"
     "Then: systemctl enable --now mytask.timer. "
     "Persistent=true means if the system was off at the scheduled time, it runs on next boot."),
    ("How do I see all logs for {service} since the last reboot?",
     "journalctl -u {service} -b\n"
     "Flags: -u {service} (filter by unit), -b (since last boot). "
     "Add -f to follow live output. -p err to show only errors. "
     "'journalctl --disk-usage' shows total log size."),
]

def make_systemd_qa():
    q_tmpl, a_tmpl = rng.choice(SYSTEMD_QA)
    kw = dict(service=rng.choice(SERVICES), name=rng.choice(PROC_NAMES), user=rng.choice(USERS))
    try:
        return {"prompt": q_tmpl.format(**kw), "response": a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt": q_tmpl, "response": a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 11. SCRIPTING TASKS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPTING_TASKS = [
    ("Write a bash script that checks if {service} is running and starts it if not.",
     "#!/usr/bin/env bash\nif ! systemctl is-active --quiet {service}; then\n"
     "    echo '{service} is down, starting...'\n    systemctl start {service}\nfi"),
    ("Write a bash one-liner that prints the top 5 IP addresses hitting an nginx access log.",
     "awk '{{print $1}}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head -5"),
    ("Write a bash script that watches {path} for changes and prints the changed filename.",
     "#!/usr/bin/env bash\ninotifywait -m -e modify,create,delete {path} 2>/dev/null |\n"
     "while read dir events file; do\n    echo \"Changed: $dir$file ($events)\"\ndone"),
    ("Write a Python one-liner that prints the 10 largest files under the current directory.",
     "python3 -c \""
     "import os; files=[(os.path.getsize(os.path.join(r,f)), os.path.join(r,f)) "
     "for r,d,fs in os.walk('.') for f in fs]; "
     "[print(f'{s//1024//1024}MB {p}') for s,p in sorted(files,reverse=True)[:10]]\""),
    ("Write a bash function that retries a command up to {n} times with a {sec}-second delay.",
     "retry() {{\n    local n={n} delay={sec} cmd=\"$@\"\n"
     "    for i in $(seq 1 $n); do\n        \"$@\" && return 0\n"
     "        echo \"Attempt $i/$n failed, retrying in {sec}s...\"\n        sleep {sec}\n    done\n"
     "    echo \"All $n attempts failed.\"; return 1\n}}"),
    ("Write a bash script that emails an alert if disk usage at {path} exceeds 90%.",
     "#!/usr/bin/env bash\nUSAGE=$(df {path} | awk 'NR==2{{print $5}}' | tr -d '%')\n"
     "if [ \"$USAGE\" -gt 90 ]; then\n"
     "    echo \"Disk at {path} is ${{USAGE}}% full\" | mail -s 'Disk Alert' admin@example.com\nfi"),
    ("Write a Python script that parses /var/log/auth.log and counts failed SSH attempts per IP.",
     "import re\nfrom collections import Counter\ncounts = Counter()\n"
     "with open('/var/log/auth.log') as f:\n"
     "    for line in f:\n        m = re.search(r'Failed password.*from (\\S+)', line)\n"
     "        if m: counts[m.group(1)] += 1\n"
     "for ip, n in counts.most_common(10): print(f'{n:5d} {ip}')"),
    ("Write a bash script that kills any process using more than {pct}% CPU.",
     "#!/usr/bin/env bash\nps aux | awk 'NR>1 && $3>{pct}{{print $2}}' | while read pid; do\n"
     "    echo \"Killing PID $pid (CPU > {pct}%)\"\n    kill -9 \"$pid\"\ndone"),
]

def make_scripting():
    q_tmpl, a_tmpl = rng.choice(SCRIPTING_TASKS)
    kw = dict(service=rng.choice(SERVICES), path=rand_path("dir"), n=rng.randint(3,10),
              sec=rng.randint(1,30), pct=rng.randint(50,95))
    try:
        return {"prompt": q_tmpl.format(**kw), "response": a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt": q_tmpl, "response": a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 12. DOCKER / CONTAINER OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

DOCKER_QA = [
    ("How do I see which Docker container is using the most memory?",
     "docker stats --no-stream --format 'table {{.Name}}\\t{{.MemUsage}}\\t{{.CPUPerc}}' | sort -k2 -rh"),
    ("Container {name} exited with code 137. What happened?",
     "Exit code 137 = 128 + 9 (SIGKILL). The container was killed by the OOM killer because it exceeded "
     "its memory limit, or it was killed manually with 'docker kill'. Check: "
     "(1) 'docker inspect {name} --format={{{{.State.OOMKilled}}}}' — true means OOM kill, "
     "(2) increase memory limit: 'docker run --memory=2g ...', "
     "(3) 'docker logs {name}' for application-level errors before the kill."),
    ("How do I exec a shell inside running container {name}?",
     "docker exec -it {name} /bin/bash\n"
     "If bash isn't available: docker exec -it {name} /bin/sh\n"
     "For a read-only container: docker cp <file> {name}:/tmp/ to inject files."),
    ("How do I copy a file from container {name} to the host?",
     "docker cp {name}:/path/in/container /host/destination/path\n"
     "Reverse (host to container): docker cp /host/file {name}:/path/in/container"),
    ("Container {name} can't reach the internet. How do I debug networking?",
     "(1) Check container DNS: 'docker exec {name} cat /etc/resolv.conf' — should have a nameserver. "
     "(2) Test connectivity: 'docker exec {name} ping 8.8.8.8'. "
     "(3) Check Docker network: 'docker inspect {name} | grep -i network'. "
     "(4) Ensure the host has IP forwarding: 'sysctl net.ipv4.ip_forward'. "
     "(5) Check iptables DOCKER chain isn't blocked."),
    ("How do I limit a container to {pct}% CPU and {mem}MB memory?",
     "docker run --cpus={cpus:.1f} --memory={mem}m <image>\n"
     "--cpus limits CPU time (1.0 = one full core). --memory sets the hard memory limit. "
     "Also: --memory-swap sets swap limit (same as --memory = no swap). "
     "View current limits: 'docker inspect <container> | grep -i cpu'"),
    ("How do I see the Dockerfile layers of image {image}?",
     "docker history {image}\n"
     "Shows each layer with size and the command that created it. For full detail: "
     "'docker inspect {image}' and look at RootFS.Layers. "
     "To see layer sizes: 'docker image inspect {image} --format {{{{.Size}}}}'."),
    ("What is the difference between COPY and ADD in a Dockerfile?",
     "COPY: copies files/directories from build context to image. Simple and predictable — preferred. "
     "ADD: does everything COPY does, plus: auto-extracts tar archives, and can fetch URLs (not recommended "
     "for security reasons). Best practice: always use COPY unless you specifically need ADD's tar "
     "extraction behavior."),
]

CONTAINER_NAMES = ["web", "api", "db", "worker", "nginx", "redis", "app", "proxy"]
IMAGES = ["ubuntu:22.04", "python:3.11-slim", "nginx:latest", "node:18", "redis:7"]

def make_docker_qa():
    q_tmpl, a_tmpl = rng.choice(DOCKER_QA)
    mem = rng.choice([256, 512, 1024, 2048, 4096])
    cpus = mem / 1024.0
    pct = rng.randint(10, 80)
    kw = dict(name=rng.choice(CONTAINER_NAMES), image=rng.choice(IMAGES),
              mem=mem, cpus=cpus, pct=pct)
    try:
        return {"prompt": q_tmpl.format(**kw), "response": a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt": q_tmpl, "response": a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

GENERATORS = [
    (make_kernel_fault,    12),
    (make_tool_call_single, 22),
    (make_tool_call_multi,  13),
    (make_shell_cmd,        13),
    (make_sysadmin_qa,      10),
    (make_proc_debug,        8),
    (make_log_analysis,      7),
    (make_security_event,    5),
    (make_network_diag,      4),
    (make_systemd_qa,        3),
    (make_scripting,         2),
    (make_docker_qa,         1),
]
_gens, _weights = zip(*GENERATORS)


def generate_dataset(n, seed=42):
    local_rng = random.Random(seed)
    samples = []
    for _ in range(n):
        gen = local_rng.choices(_gens, weights=_weights, k=1)[0]
        try:
            samples.append(gen())
        except Exception:
            samples.append(make_kernel_fault())
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  type=int, default=50000)
    parser.add_argument("--val",    type=int, default=5000)
    parser.add_argument("--outdir", default="training/data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Generating {args.train} training samples...")
    train = generate_dataset(args.train, seed=42)
    train_path = os.path.join(args.outdir, "broad_train.jsonl")
    with open(train_path, "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved → {train_path}  ({os.path.getsize(train_path)//1024//1024} MB)")

    print(f"Generating {args.val} validation samples...")
    val = generate_dataset(args.val, seed=99)
    val_path = os.path.join(args.outdir, "broad_val.jsonl")
    with open(val_path, "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved → {val_path}  ({os.path.getsize(val_path)//1024//1024} MB)")

    # Category stats
    print("\nCategory weights:")
    total_w = sum(_weights)
    for gen, w in GENERATORS:
        print(f"  {gen.__name__:<25} {100*w/total_w:5.1f}%  (~{args.train*w//total_w} samples)")
    print(f"\nTotal: {args.train} train / {args.val} val")


if __name__ == "__main__":
    main()
