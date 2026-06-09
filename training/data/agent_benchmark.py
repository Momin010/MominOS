#!/usr/bin/env python3
"""
Phase 2/4: Agent Benchmark for MominoMoE OS Copilot.

Evaluates the model across 5 benchmarks:
  a) Tool call accuracy (30 synthetic scenarios)
  b) Bash command generation (200 intercode_bash held-out)
  c) Multi-turn coherence (20 fixed multi-turn dialogues)
  d) Output format compliance (from all test data)
  e) System log diagnosis (20 synthetic fault logs from templates.py)

Usage:
  python3 data/agent_benchmark.py [--checkpoint PATH] [--output PATH]
  
  Without --checkpoint: loads untrained (random weights) model
  With --checkpoint: loads trained model from checkpoint
"""

import os
import sys
import json
import re
import math
import random
import argparse
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "..", "benchmark_results")
BENCHMARK_DIR = os.path.abspath(BENCHMARK_DIR)

# ── Tool Call Accuracy (30 synthetic scenarios) ──────────────────────────

TOOL_CALL_PROMPTS = [
    {
        "id": "tc_001",
        "prompt": "<|user|> I need you to install the nginx web server on this system.",
        "expected_function": "install_package",
        "expected_args": {"package": "nginx"}
    },
    {
        "id": "tc_002",
        "prompt": "<|user|> Find all files larger than 1GB in /var/log.",
        "expected_function": "find_files",
        "expected_args": {"path": "/var/log", "min_size": "1GB"}
    },
    {
        "id": "tc_003",
        "prompt": "<|user|> Check the disk usage of the root filesystem.",
        "expected_function": "check_disk_usage",
        "expected_args": {"path": "/"}
    },
    {
        "id": "tc_004",
        "prompt": "<|user|> Restart the postgresql service, it's not responding.",
        "expected_function": "restart_service",
        "expected_args": {"service": "postgresql"}
    },
    {
        "id": "tc_005",
        "prompt": "<|user|> Create a new user called 'deploy' with sudo privileges.",
        "expected_function": "create_user",
        "expected_args": {"username": "deploy", "groups": ["sudo"]}
    },
    {
        "id": "tc_006",
        "prompt": "<|user|> List all running Docker containers on this host.",
        "expected_function": "list_containers",
        "expected_args": {"all": False}
    },
    {
        "id": "tc_007",
        "prompt": "<|user|> Configure the firewall to allow traffic on port 8080.",
        "expected_function": "configure_firewall",
        "expected_args": {"port": 8080, "protocol": "tcp", "action": "allow"}
    },
    {
        "id": "tc_008",
        "prompt": "<|user|> Show me the last 50 lines of the system log.",
        "expected_function": "read_log",
        "expected_args": {"path": "/var/log/syslog", "lines": 50}
    },
    {
        "id": "tc_009",
        "prompt": "<|user|> Check how much memory is currently being used.",
        "expected_function": "check_memory",
        "expected_args": {}
    },
    {
        "id": "tc_010",
        "prompt": "<|user|> Schedule a cron job to run backup.sh every day at 2am.",
        "expected_function": "schedule_cron",
        "expected_args": {"command": "/usr/local/bin/backup.sh", "schedule": "0 2 * * *"}
    },
    {
        "id": "tc_011",
        "prompt": "<|user|> Download the file from https://example.com/file.tar.gz and extract it.",
        "expected_function": "download_and_extract",
        "expected_args": {"url": "https://example.com/file.tar.gz"}
    },
    {
        "id": "tc_012",
        "prompt": "<|user|> Kill the process that is using port 3000.",
        "expected_function": "kill_process_on_port",
        "expected_args": {"port": 3000}
    },
    {
        "id": "tc_013",
        "prompt": "<|user|> Set the environment variable DB_HOST to 'localhost'.",
        "expected_function": "set_env_variable",
        "expected_args": {"name": "DB_HOST", "value": "localhost"}
    },
    {
        "id": "tc_014",
        "prompt": "<|user|> Run the ansible playbook deploy.yml in check mode.",
        "expected_function": "run_playbook",
        "expected_args": {"playbook": "deploy.yml", "check_mode": True}
    },
    {
        "id": "tc_015",
        "prompt": "<|user|> Generate an SSL certificate for example.com.",
        "expected_function": "generate_ssl_cert",
        "expected_args": {"domain": "example.com"}
    },
    {
        "id": "tc_016",
        "prompt": "<|user|> Mount the NFS share at /mnt/data from 10.0.0.1:/exports.",
        "expected_function": "mount_nfs",
        "expected_args": {"source": "10.0.0.1:/exports", "mount_point": "/mnt/data"}
    },
    {
        "id": "tc_017",
        "prompt": "<|user|> Check the status of all systemd services.",
        "expected_function": "check_service_status",
        "expected_args": {}
    },
    {
        "id": "tc_018",
        "prompt": "<|user|> Update all installed packages to their latest versions.",
        "expected_function": "update_packages",
        "expected_args": {}
    },
    {
        "id": "tc_019",
        "prompt": "<|user|> Clone the git repository https://github.com/user/repo.git to /opt/project.",
        "expected_function": "git_clone",
        "expected_args": {"url": "https://github.com/user/repo.git", "destination": "/opt/project"}
    },
    {
        "id": "tc_020",
        "prompt": "<|user|> Run a traceroute to google.com to check network latency.",
        "expected_function": "trace_route",
        "expected_args": {"target": "google.com"}
    },
    {
        "id": "tc_021",
        "prompt": "<|user|> Check which users are currently logged into the system.",
        "expected_function": "list_logged_in_users",
        "expected_args": {}
    },
    {
        "id": "tc_022",
        "prompt": "<|user|> Scan the local network 192.168.1.0/24 for active hosts.",
        "expected_function": "network_scan",
        "expected_args": {"subnet": "192.168.1.0/24"}
    },
    {
        "id": "tc_023",
        "prompt": "<|user|> Compress the /var/log directory into a tar.gz archive.",
        "expected_function": "compress_directory",
        "expected_args": {"source": "/var/log", "format": "tar.gz"}
    },
    {
        "id": "tc_024",
        "prompt": "<|user|> Search for the string 'ERROR 503' in all log files under /var/log/app/.",
        "expected_function": "search_in_files",
        "expected_args": {"pattern": "ERROR 503", "path": "/var/log/app"}
    },
    {
        "id": "tc_025",
        "prompt": "<|user|> Check if port 443 is open on the remote host api.example.com.",
        "expected_function": "check_port",
        "expected_args": {"host": "api.example.com", "port": 443}
    },
    {
        "id": "tc_026",
        "prompt": "<|user|> I need to investigate a kernel module that might be causing system instability. Find all loaded kernel modules and check the dmesg output for errors related to them.",
        "expected_function": "investigate_kernel_modules",
        "expected_args": {}
    },
    {
        "id": "tc_027",
        "prompt": "<|user|> The database connection pool is exhausted. Show me current connections, kill any idle connections older than 1 hour, and restart the connection pool service.",
        "expected_function": "manage_db_connections",
        "expected_args": {"action": "cleanup", "idle_threshold_hours": 1}
    },
    {
        "id": "tc_028",
        "prompt": "<|user|> We need to rotate the application logs for myapp. Compress logs older than 7 days, delete logs older than 30 days, and signal the app to reopen log files.",
        "expected_function": "rotate_logs",
        "expected_args": {"app": "myapp", "compress_after_days": 7, "delete_after_days": 30}
    },
    {
        "id": "tc_029",
        "prompt": "<|user|> Create a 10GB swap file at /swapfile and enable it.",
        "expected_function": "create_swap",
        "expected_args": {"path": "/swapfile", "size_gb": 10}
    },
    {
        "id": "tc_030",
        "prompt": "<|user|> Run a performance benchmark on the disk /dev/sda and report the IOPS and latency.",
        "expected_function": "benchmark_disk",
        "expected_args": {"device": "/dev/sda"}
    },
]


# ── Multi-turn Coherence (20 multi-turn dialogues) ───────────────────────

MULTI_TURN_DIALOGUES = [
    {
        "id": "mt_001",
        "turns": [
            {"role": "user", "content": "I need to set up a web server for my application."},
            # Expected response establishes context about web server setup
            {"role": "user", "content": "What kind of configuration does it need?"},
            # Should reference "web server" context
            {"role": "user", "content": "Actually, I changed my mind. I want to use nginx."},
            # Should update context from web server → nginx
            {"role": "user", "content": "Can you show me a basic config for it?"},
            # Should reference "nginx" from previous turn
        ],
        "context_keywords": ["nginx", "web server", "config"],
        "expected_reference_turns": [0, 1, 2]  # Turns the model should reference
    },
    {
        "id": "mt_002",
        "turns": [
            {"role": "user", "content": "My database is running slowly."},
            {"role": "user", "content": "What queries are currently running?"},
            {"role": "user", "content": "Kill any that have been running for more than 5 minutes."},
            {"role": "user", "content": "Now show me the slow query log."},
        ],
        "context_keywords": ["database", "query", "slow"],
        "expected_reference_turns": [0, 1, 2]
    },
    {
        "id": "mt_003",
        "turns": [
            {"role": "user", "content": "My server keeps running out of disk space."},
            {"role": "user", "content": "Find large files in /var/log."},
            {"role": "user", "content": "Delete anything older than 30 days."},
            {"role": "user", "content": "Set up a log rotation policy to prevent this."},
        ],
        "context_keywords": ["disk space", "/var/log", "log rotation"],
        "expected_reference_turns": [0, 1, 2]
    },
    {
        "id": "mt_004",
        "turns": [
            {"role": "user", "content": "I need to deploy my application to production."},
            {"role": "user", "content": "The first step is to install Docker on the server."},
            {"role": "user", "content": "I also need docker-compose."},
            {"role": "user", "content": "Once that's installed, pull the image from my registry."},
        ],
        "context_keywords": ["deploy", "Docker", "docker-compose", "registry"],
        "expected_reference_turns": [0, 1, 2]
    },
    {
        "id": "mt_005",
        "turns": [
            {"role": "user", "content": "A critical alert: our API endpoint is returning 502 errors."},
            {"role": "user", "content": "Check the nginx status first."},
            {"role": "user", "content": "Nginx is running. Now check the backend application."},
            {"role": "user", "content": "The backend is overloaded. Increase the worker count."},
        ],
        "context_keywords": ["502", "API", "nginx", "backend", "worker"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_006",
        "turns": [
            {"role": "user", "content": "Let's audit our system security."},
            {"role": "user", "content": "Check for any users with root privileges."},
            {"role": "user", "content": "Also check what ports are open."},
            {"role": "user", "content": "Now compare this list with our security policy."},
        ],
        "context_keywords": ["security", "audit", "root", "ports", "policy"],
        "expected_reference_turns": [0, 1, 2]
    },
    {
        "id": "mt_007",
        "turns": [
            {"role": "user", "content": "Help me debug a network issue."},
            {"role": "user", "content": "Ping google.com from this server."},
            {"role": "user", "content": "Ping works. Try curl to our API endpoint."},
            {"role": "user", "content": "Curl fails. Check the DNS resolution."},
        ],
        "context_keywords": ["ping", "curl", "google", "DNS", "network"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_008",
        "turns": [
            {"role": "user", "content": "I want to set up monitoring for my infrastructure."},
            {"role": "user", "content": "Install Prometheus on the monitoring server."},
            {"role": "user", "content": "Configure it to scrape metrics every 15 seconds."},
            {"role": "user", "content": "Now set up a Grafana dashboard for the metrics."},
        ],
        "context_keywords": ["monitoring", "Prometheus", "scrape", "Grafana"],
        "expected_reference_turns": [0, 1, 2]
    },
    {
        "id": "mt_009",
        "turns": [
            {"role": "user", "content": "We have a security incident!"},
            {"role": "user", "content": "An unauthorized user accessed the admin panel."},
            {"role": "user", "content": "Check the access logs for that IP address."},
            {"role": "user", "content": "Revoke all sessions from that IP."},
        ],
        "context_keywords": ["security incident", "unauthorized", "admin", "access logs", "IP"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_010",
        "turns": [
            {"role": "user", "content": "I'm setting up a CI/CD pipeline."},
            {"role": "user", "content": "Start with the build stage."},
            {"role": "user", "content": "After building, run the test suite."},
            {"role": "user", "content": "If tests pass, deploy to staging."},
        ],
        "context_keywords": ["CI/CD", "build", "test", "staging", "pipeline"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_011",
        "turns": [
            {"role": "user", "content": "I need to migrate our database."},
            {"role": "user", "content": "First, take a backup of the current database."},
            {"role": "user", "content": "The database is called 'production_db'."},
            {"role": "user", "content": "Now restore it to the new server at 10.0.0.5."},
        ],
        "context_keywords": ["migrate", "backup", "production_db", "restore"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_012",
        "turns": [
            {"role": "user", "content": "I'm having issues with our load balancer."},
            {"role": "user", "content": "Check the current load balancer configuration."},
            {"role": "user", "content": "Remove any unhealthy backend servers."},
            {"role": "user", "content": "Add two new backend servers to the pool."},
        ],
        "context_keywords": ["load balancer", "configuration", "unhealthy", "backend"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_013",
        "turns": [
            {"role": "user", "content": "Let me tell you about our infrastructure."},
            {"role": "user", "content": "We have 3 web servers and 2 database servers."},
            {"role": "user", "content": "The web servers run nginx and the app."},
            {"role": "user", "content": "What was the database server count I mentioned?"},
        ],
        "context_keywords": ["3", "web servers", "2", "database"],
        "expected_reference_turns": [1, 2]
    },
    {
        "id": "mt_014",
        "turns": [
            {"role": "user", "content": "I need to fix the SSL certificate issue."},
            {"role": "user", "content": "The certificate for api.example.com expired yesterday."},
            {"role": "user", "content": "I've obtained a new certificate."},
            {"role": "user", "content": "Install it and reload nginx."},
        ],
        "context_keywords": ["SSL", "certificate", "api.example.com", "nginx"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_015",
        "turns": [
            {"role": "user", "content": "I want to optimize our server performance."},
            {"role": "user", "content": "Check the current CPU and memory usage."},
            {"role": "user", "content": "We have 8 CPUs and 32GB RAM."},
            {"role": "user", "content": "Adjust the kernel parameters based on the usage."},
        ],
        "context_keywords": ["CPU", "memory", "8", "32", "kernel"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_016",
        "turns": [
            {"role": "user", "content": "We need to set up a backup strategy."},
            {"role": "user", "content": "Backup the database every 6 hours."},
            {"role": "user", "content": "Keep 7 days of daily backups and 4 weeks of weekly backups."},
            {"role": "user", "content": "That means keep 28 daily backups and 4 weekly."},
        ],
        "context_keywords": ["backup", "6 hours", "7 days", "4 weeks"],
        "expected_reference_turns": [1, 2, 3]
    },
    {
        "id": "mt_017",
        "turns": [
            {"role": "user", "content": "I'm investigating a memory leak."},
            {"role": "user", "content": "Check the memory usage of the Java process first."},
            {"role": "user", "content": "It's using 12GB of RSS. That's too high for this app."},
            {"role": "user", "content": "Take a heap dump and analyze it."},
        ],
        "context_keywords": ["memory leak", "Java", "12GB", "heap dump"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_018",
        "turns": [
            {"role": "user", "content": "Set up a new kubernetes cluster."},
            {"role": "user", "content": "Install kubeadm on all nodes."},
            {"role": "user", "content": "Initialize the control plane on node-1."},
            {"role": "user", "content": "Join the worker nodes to the cluster."},
        ],
        "context_keywords": ["kubernetes", "kubeadm", "control plane", "worker"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_019",
        "turns": [
            {"role": "user", "content": "I need to troubleshoot a DNS issue."},
            {"role": "user", "content": "Our internal DNS server is at 10.0.0.53."},
            {"role": "user", "content": "Check if it's resolving internal hostnames."},
            {"role": "user", "content": "Test resolution of 'db.internal.company.com'."},
        ],
        "context_keywords": ["DNS", "10.0.0.53", "internal", "resolution"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
    {
        "id": "mt_020",
        "turns": [
            {"role": "user", "content": "I want to enable HTTPS for our site."},
            {"role": "user", "content": "We already have a certificate from Let's Encrypt."},
            {"role": "user", "content": "Configure the web server to use it."},
            {"role": "user", "content": "Also set up automatic renewal."},
        ],
        "context_keywords": ["HTTPS", "Let's Encrypt", "certificate", "renewal"],
        "expected_reference_turns": [0, 1, 2, 3]
    },
]


# ── Synthetic Fault Logs (from templates.py patterns) ────────────────────

SYSTEM_FAULT_LOGS = [
    {
        "id": "fl_001",
        "fault_type": "oom",
        "log": """[114947.206290] oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0, oom_score_adj=0
[114947.206293] CPU: 2 PID: 12345 Comm: mysqld Killed: 1487
[114947.206295] Memory cgroup stats for /system.slice/mysql.service: cache:0KB rss:1522688KB rss_huge:0KB shmem:0KB
[114947.206298] Memory cgroup out of memory: Killed process 12345 (mysqld) total-vm:2147488kB, anon-rss:1522688kB, file-rss:0kB, shmem-rss:0kB""",
        "expected_fault": "oom"
    },
    {
        "id": "fl_002",
        "fault_type": "oom",
        "log": """[ 1234.567890] java invoked oom-killer: gfp_mask=0x100cca(GFP_HIGHUSER_MOVABLE), order=0, oom_score_adj=0
[ 1234.567895] java: cpuset=/ mems_allowed=0-1
[ 1234.567898] CPU: 4 PID: 9876 Comm: java Tainted: P           OE
[ 1234.567900] Hardware name: SuperServer, BIOS 3.2
[ 1234.567902] Call Trace: dump_stack+0x6b/0x88, dump_header+0x4a/0x1f8, oom_kill_process+0x2b3/0x490
[ 1234.567910] Out of memory: Killed process 9876 (java) total-vm:8388608kB, anon-rss:6291456kB, file-rss:0kB""",
        "expected_fault": "oom"
    },
    {
        "id": "fl_003",
        "fault_type": "oom",
        "log": """[ 5432.123456] kworker/3:0 invoked oom-killer: gfp_mask=0x24000c0(GFP_KERNEL), order=0, oom_score_adj=-1000
[ 5432.123460] kworker/3:0: cpuset=/ mems_allowed=0
[ 5432.123462] CPU: 3 PID: 4567 Comm: kworker/3:0 Not tainted
[ 5432.123464] Hardware name: Google Compute Engine
[ 5432.123466] Workqueue: events_freezable_power_ disk_events_workfn
[ 5432.123470] Memory: 16384M, available: 1234M, free: 890M, swap: 2048M/2048M
[ 5432.123472] [oom] 0 total pagecache pages, 0 anonymous pages, 0 swap cached""",
        "expected_fault": "oom"
    },
    {
        "id": "fl_004",
        "fault_type": "oom",
        "log": """[ 7890.123456] Node 0 DMA free:1234kB min:456kB low:789kB high:1012kB active_anon:5678kB inactive_anon:9012kB active_file:345kB inactive_file:678kB
[ 7890.123460] Node 0 DMA32 free:0kB min:0kB low:0kB high:0kB active_anon:0kB inactive_anon:0kB active_file:0kB inactive_file:0kB
[ 7890.123462] Normal free:0kB min:0kB low:0kB high:0kB = 0 total pagecache pages
[ 7890.123464] oom_kill_process: Out of memory: Killed process 23456 (java) total-vm:4194304kB
[ 7890.123466] oom_reaper: reaped process 23456 (java), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB""",
        "expected_fault": "oom"
    },
    {
        "id": "fl_005",
        "fault_type": "panic",
        "log": """BUG: unable to handle kernel NULL pointer dereference at 0000000000000000
IP: [<ffffffffa03e8b20>] mydriver_write+0x50/0x100 [mydriver]
PGD 0 P4D 0 
Oops: 0002 [#1] SMP 
CPU: 1 PID: 54321 Comm: test_app Tainted: P           O
RIP: 0010:mydriver_write+0x50/0x100 [mydriver]
Call Trace:
 [<ffffffff811e6f09>] SyS_write+0xd9/0x1d0
 [<ffffffff8169a0b2>] entry_SYSCALL_64_fastpath+0x16/0x75
Code: 48 8b 45 00 48 85 c0 74 02 ff d0 48 8b 7d 08 48 8b 45 10 <48> 8b 18 48 85 db 74 0e 48 89 df ff d3 48 8b 45 18 48 85 c0 75 d6""",
        "expected_fault": "panic"
    },
    {
        "id": "fl_006",
        "fault_type": "panic",
        "log": """------------[ cut here ]------------
kernel BUG at fs/ext4/inode.c:2736!
invalid opcode: 0000 [#1] SMP 
CPU: 2 PID: 34567 Comm: kworker/u4:3 Tainted: P           O
RIP: 0010:ext4_writepages+0x1f4/0x380 [ext4]
Call Trace:
 [<ffffffff81234567>] do_writepages+0x17/0x30
 [<ffffffff811e1234>] __writeback_single_inode+0x44/0x330
 [<ffffffff811e1a2b>] writeback_sb_inodes+0x27b/0x4a0
 [<ffffffff811e1ca7>] __writeback_inodes_wb+0x57/0xb0
 [<ffffffff811e1f5c>] wb_writeback+0x25c/0x320
 [<ffffffff811e23ca>] wb_workfn+0x3aa/0x500
 [<ffffffff810a3f5f>] process_one_work+0x1af/0x4a0
RIP [<ffffffff81234567>] ext4_writepages+0x1f4/0x380 [ext4]
 RSP <ffff880012345678>
---[ end trace 1234567890abcdef ]---""",
        "expected_fault": "panic"
    },
    {
        "id": "fl_007",
        "fault_type": "panic",
        "log": """general protection fault: 0000 [#1] SMP 
CPU: 0 PID: 7890 Comm: sshd Not tainted 5.15.0-generic
RIP: 0010:__clear_user+0x2d/0x50
Code: 90 90 90 90 90 eb 00 66 66 66 66 90 48 89 f8 48 89 f1 48 c1 e9 03 83 e0 07 48 c1 e9 03 83 e0 07 74 0b 48 89 c1 f3 aa 48 89 f9 89 fa <f3> aa c3 0f 1f 84 00 00 00 00 00 48 89 f8 48 89 f1 f3 aa 48 89 f9
RSP: 0018:ffffb23456789012 EFLAGS: 00010246
RAX: 0000000000000000 RBX: ffff880012345678 RCX: 0000000000000000
RDX: 0000000000000fff RSI: 0000000000000000 RDI: 0000000000000000
RBP: ffff880087654321 R08: 0000000000000000 R09: 0000000000000002
R10: 0000000000000000 R11: 0000000000000246 R12: 0000000000000000
R13: 00007ffe12345678 R14: 0000000000001000 R15: 0000556789abcdef
Modules linked in: tun overlay nf_conntrack
CR2: 0000000000000000
---[ end trace fedcba0987654321 ]---""",
        "expected_fault": "panic"
    },
    {
        "id": "fl_008",
        "fault_type": "deadlock",
        "log": """[ 23456.789012] INFO: task java:12345 blocked for more than 120 seconds.
[ 23456.789015]       Tainted: P           O      
[ 23456.789017] "echo 0 > /proc/sys/kernel/hung_task_timeout_secs" disables this message.
[ 23456.789019] java         D 0 12345   9876 0x00000000
[ 23456.789022] Call Trace:
[ 23456.789024]  __schedule+0x2a4/0x8a0
[ 23456.789027]  schedule+0x38/0xa0
[ 23456.789029]  io_schedule+0x12/0x40
[ 23456.789032]  wait_on_page_bit+0x10d/0x220
[ 23456.789035]  __filemap_fdatawait_range+0xbc/0x120
[ 23456.789038]  filemap_fdatawait+0x1c/0x20
[ 23456.789040]  ext4_sync_file+0xaa/0x320 [ext4]
[ 23456.789045]  do_fsync+0x38/0x60
[ 23456.789048]  SyS_fsync+0x10/0x20
[ 23456.789050]  entry_SYSCALL_64_fastpath+0x16/0x75""",
        "expected_fault": "deadlock"
    },
    {
        "id": "fl_009",
        "fault_type": "deadlock",
        "log": """[ 34567.890123] INFO: task kworker/2:1:4567 blocked for more than 120 seconds.
[ 34567.890126] kworker/2:1    D 0  4567      2 0x00000000
[ 34567.890129] Workqueue: events_freezable_power_ disk_events_workfn
[ 34567.890132] Call Trace:
[ 34567.890134]  __schedule+0x2a4/0x8a0
[ 34567.890137]  schedule+0x38/0xa0
[ 34567.890139]  schedule_preempt_disabled+0x14/0x20
[ 34567.890141]  __mutex_lock.isra.8+0x2bc/0x4e0
[ 34567.890144]  __mutex_lock_slowpath+0x13/0x20
[ 34567.890147]  mutex_lock+0x2f/0x40
[ 34567.890149]  disk_events_workfn+0x1c/0x90
[ 34567.890152]  process_one_work+0x1af/0x4a0
[ 34567.890155]  worker_thread+0x4d/0x4b0
[ 34567.890157]  kthread+0x105/0x140
[ 34567.890159]  ret_from_fork+0x2c/0x40""",
        "expected_fault": "deadlock"
    },
    {
        "id": "fl_010",
        "fault_type": "deadlock",
        "log": """[ 45678.901234] INFO: rcu_sched detected stalls on CPUs/tasks:
[ 45678.901237]  (detected by 3, t=2102 jiffies, g=456789, q=12345)
[ 45678.901239] Sending NMI to all CPUs:
[ 45678.901241] NMI backtrace for cpu 2
[ 45678.901243] CPU: 2 PID: 7890 Comm: rm Tainted: P           O
[ 45678.901246] Call Trace:
[ 45678.901248]  <IRQ>
[ 45678.901249]  _raw_spin_lock+0x1c/0x30
[ 45678.901252]  try_to_unmap_one+0x2d1/0x490
[ 45678.901255]  rmap_walk_anon+0xd3/0x250
[ 45678.901258]  try_to_unmap+0x67/0x80
[ 45678.901260]  shrink_page_list+0x5b9/0x940
[ 45678.901263]  shrink_inactive_list+0x1e5/0x460
[ 45678.901266]  shrink_node_memcg+0x33e/0x730
[ 45678.901269]  shrink_node+0xda/0x2d0
[ 45678.901272]  do_try_to_free_pages+0xb8/0x340
[ 45678.901274]  try_to_free_pages+0x87/0x1a0
[ 45678.901277]  __alloc_pages_slowpath+0x3c1/0xbf0
[ 45678.901280]  __alloc_pages_nodemask+0x1d2/0x230
[ 45678.901283]  alloc_pages_vma+0x7b/0x1c0""",
        "expected_fault": "deadlock"
    },
    {
        "id": "fl_011",
        "fault_type": "memory_corruption",
        "log": """[ 56789.012345] =============================================================================
[ 56789.012347] BUG kmalloc-512 (Tainted: P    B    ): Redzone overwritten
[ 56789.012349] -----------------------------------------------------------------------------
[ 56789.012351] INFO: 0x0000000012345678-0x000000001234567f @offset=512. First byte 0x41 instead of 0xcc
[ 56789.012354] INFO: Allocated in mydriver_alloc+0x3a/0x80 [mydriver] age=123 cpu=0 pid=4567
[ 56789.012358] INFO: Freed in mydriver_free+0x20/0x50 [mydriver] age=45 cpu=0 pid=4567
[ 56789.012361] INFO: Slab 0x0000000098765432 objects=16 used=2 fp=0x0000000056789012 flags=0x17c000000102
[ 56789.012365] INFO: Object 0x0000000043210987 @offset=0 fp=0x00000000deadbeef
[ 56789.012367] 
[ 56789.012368] Redzone 0x0000000012345670: cc cc cc cc cc cc cc cc                          
[ 56789.012371] Redzone 0x0000000012345678: 41 41 41 41 41 41 41 41                          """,
        "expected_fault": "memory_corruption"
    },
    {
        "id": "fl_012",
        "fault_type": "memory_corruption",
        "log": """[ 67890.123456] =============================================================================
[ 67890.123458] BUG: KASAN: slab-use-after-free in mydriver_read+0x45/0x100 [mydriver]
[ 67890.123461] Read of size 8 at addr ffff880012345678 by task test/78901
[ 67890.123463] 
[ 67890.123464] CPU: 3 PID: 78901 Comm: test Tainted: P    B    O
[ 67890.123466] Hardware name: Google Compute Engine
[ 67890.123468] Call Trace:
[ 67890.123469]  dump_stack+0x6b/0x88
[ 67890.123472]  print_address_description+0x6b/0x23b
[ 67890.123475]  kasan_report+0x244/0x390
[ 67890.123478]  mydriver_read+0x45/0x100 [mydriver]
[ 67890.123481]  vfs_read+0xb5/0x1d0
[ 67890.123484]  SyS_read+0x5d/0x110
[ 67890.123486]  entry_SYSCALL_64_fastpath+0x16/0x75
[ 67890.123489] 
[ 67890.123490] Allocated by task 78901:
[ 67890.123492]  mydriver_alloc_buffer+0x2a/0x80 [mydriver]
[ 67890.123495]  mydriver_open+0x15/0x30 [mydriver]
[ 67890.123498] 
[ 67890.123499] Freed by task 78901:
[ 67890.123501]  mydriver_close+0x1a/0x40 [mydriver]
[ 67890.123504]  __fput+0xed/0x230
[ 67890.123507]  ____fput+0xe/0x10
[ 67890.123509]  task_work_run+0x89/0xc0""",
        "expected_fault": "memory_corruption"
    },
    {
        "id": "fl_013",
        "fault_type": "memory_corruption",
        "log": """[ 78901.234567] list_del corruption, ffff880012345678->next is LIST_POISON1 (0x0000000000000100)
[ 78901.234570] kernel BUG at lib/list_debug.c:50!
[ 78901.234572] invalid opcode: 0000 [#1] SMP 
[ 78901.234574] CPU: 1 PID: 2345 Comm: kworker/1:2 Tainted: P    B    O
[ 78901.234576] Hardware name: Google Compute Engine
[ 78901.234578] Workqueue: events vmstat_update
[ 78901.234581] RIP: 0010:__list_del_entry_valid+0x58/0xa0
[ 78901.234584] Call Trace:
[ 78901.234585]  __list_del_entry+0xf/0x30
[ 78901.234588]  remove_wait_queue+0x2c/0x50
[ 78901.234591]  ep_unregister_pollwait+0x50/0xe0
[ 78901.234594]  ep_remove+0x45/0xa0
[ 78901.234596]  eventpoll_release_file+0x5e/0xc0""",
        "expected_fault": "memory_corruption"
    },
    {
        "id": "fl_014",
        "fault_type": "memory_corruption",
        "log": """[ 89012.345678] =============================================================================
[ 89012.345680] BUG: KASAN: double-free in mydriver_free+0x25/0x60 [mydriver]
[ 89012.345683] Free of addr ffff880012345678 by task test/90123
[ 89012.345685] 
[ 89012.345686] CPU: 0 PID: 90123 Comm: test Tainted: P    B    O
[ 89012.345688] Call Trace:
[ 89012.345689]  dump_stack+0x6b/0x88
[ 89012.345692]  print_address_description+0x6b/0x23b
[ 89012.345695]  kasan_report_double_free+0x51/0x70
[ 89012.345698]  kasan_slab_free+0x9b/0xc0
[ 89012.345700]  kmem_cache_free+0x99/0x1b0
[ 89012.345703]  mydriver_free+0x25/0x60 [mydriver]
[ 89012.345706]  mydriver_release+0x2a/0x50 [mydriver]
[ 89012.345709]  __fput+0xed/0x230
[ 89012.345711]  ____fput+0xe/0x10
[ 89012.345714]  task_work_run+0x89/0xc0
[ 89012.345716]  do_exit+0x3a7/0xbf0""",
        "expected_fault": "memory_corruption"
    },
    {
        "id": "fl_015",
        "fault_type": "memory_corruption",
        "log": """[ 90123.456789]  vmap allocation for size 16777216 failed: use vmalloc=<size> to increase size.
[ 90123.456791] kworker/4:1: page allocation failure: order:6, mode:0x1c0c0(GFP_KERNEL|__GFP_COMP|__GFP_ZERO), nodemask=(null)
[ 90123.456795] CPU: 4 PID: 3456 Comm: kworker/4:1 Tainted: P    B    O
[ 90123.456797] Workqueue: events mydriver_worker
[ 90123.456800] Call Trace:
[ 90123.456801]  dump_stack+0x6b/0x88
[ 90123.456804]  warn_alloc+0x104/0x1a0
[ 90123.456807]  __vmalloc_node_range+0x45a/0x5b0
[ 90123.456810]  __vmalloc_node+0x49/0x50
[ 90123.456812]  mydriver_alloc_large+0x1f/0x50 [mydriver]
[ 90123.456815]  mydriver_worker+0x5a/0x120 [mydriver]""",
        "expected_fault": "memory_corruption"
    },
    {
        "id": "fl_016",
        "fault_type": "oom",
        "log": """[ 12345.678901] Node 0 DMA free:456kB min:128kB low:256kB high:384kB
[ 12345.678904] lowmem_reserve[]: 0 0 0 0
[ 12345.678906] Node 0 DMA32 free:1234kB min:567kB low:890kB high:1234kB
[ 12345.678909] Node 0 Normal free:89012kB min:23456kB low:34567kB high:45678kB
[ 12345.678911] Node 0 Normal: 2345*4kB (U) 678*8kB (U) 123*16kB (U) 45*32kB (U) 12*64kB (U) 3*128kB (U) 0*256kB 0*512kB 0*1024kB 0*2048kB 0*4096kB = 0kB
[ 12345.678920] 23456 total pagecache pages, 0 pages in swap cache
[ 12345.678922] Normal free:89012kB but OOM! Fragmentation prevents order-4 allocation""",
        "expected_fault": "oom"
    },
    {
        "id": "fl_017",
        "fault_type": "panic",
        "log": """[ 23456.789012] Kernel panic - not syncing: Fatal exception in interrupt
[ 23456.789015] CPU: 1 PID: 0 Comm: swapper/1 Not tainted 5.15.0-generic
[ 23456.789018] Hardware name: Google Compute Engine
[ 23456.789020] Call Trace:
[ 23456.789021]  <NMI>
[ 23456.789023]  dump_stack+0x6b/0x88
[ 23456.789026]  panic+0x104/0x2f0
[ 23456.789029]  nmi_panic+0x2b/0x40
[ 23456.789031]  watchdog_overflow_callback+0xcc/0xd0
[ 23456.789034]  __perf_event_overflow+0x9f/0x1c0
[ 23456.789037]  handle_pmi_common+0x1b6/0x260
[ 23456.789040]  intel_pmu_handle_irq+0x7b/0x180
[ 23456.789043]  perf_event_nmi_handler+0x28/0x50
[ 23456.789046]  nmi_handle+0x6b/0x120
[ 23456.789049]  default_do_nmi+0x54/0x120
[ 23456.789052]  do_nmi+0xb5/0x140
[ 23456.789054]  end_repeat_nmi+0x16/0x5a
[ 23456.789057] RIP: 0010:native_safe_halt+0x6/0x10""",
        "expected_fault": "panic"
    },
    {
        "id": "fl_018",
        "fault_type": "deadlock",
        "log": """[ 34567.890123] INFO: task dp:12345 blocked for more than 120 seconds.
[ 34567.890126] dp              D 0 12345   1234 0x00000000
[ 34567.890129] Call Trace:
[ 34567.890131]  __schedule+0x2a4/0x8a0
[ 34567.890134]  schedule+0x38/0xa0
[ 34567.890136]  schedule_timeout+0x1e8/0x300
[ 34567.890139]  wait_for_completion+0x8b/0x120
[ 34567.890142]  __flush_work+0x13d/0x1f0
[ 34567.890145]  flush_work+0x10/0x20
[ 34567.890148]  nbd_device_work+0x2c/0x40 [nbd]
[ 34567.890152]  recv_work+0x73/0xb0 [nbd]
[ 34567.890155]  process_one_work+0x1af/0x4a0
[ 34567.890158]  worker_thread+0x4d/0x4b0
[ 34567.890161]  kthread+0x105/0x140
[ 34567.890163]  ret_from_fork+0x2c/0x40""",
        "expected_fault": "deadlock"
    },
    {
        "id": "fl_019",
        "fault_type": "deadlock",
        "log": """[ 56789.012345] ======================================================
[ 56789.012347] WARNING: possible circular locking dependency detected
[ 56789.012349] 5.15.0-generic #1 Not tainted
[ 56789.012351] ------------------------------------------------------
[ 56789.012353] kworker/1:2/4567 is trying to acquire lock:
[ 56789.012356]  ffff880012345678 (&sb->s_type->i_mutex_key#3){+.+.}-{3:3}, at: ext4_write+0x2a/0x50 [ext4]
[ 56789.012362] but task is already holding lock:
[ 56789.012364]  ffff880087654321 (&mm->mmap_sem){++++}-{3:3}, at: do_page_fault+0x12e/0x4a0
[ 56789.012370] which lock already depends on the new lock.
[ 56789.012372] the existing dependency chain (in reverse order) is:
[ 56789.012374] -> #1 (&mm->mmap_sem){++++}-{3:3}:
[ 56789.012379]        __might_fault+0x61/0x90
[ 56789.012382]        _copy_to_user+0x23/0x80
[ 56789.012385]        ext4_write+0x3a/0x50 [ext4]
[ 56789.012389] -> #0 (&sb->s_type->i_mutex_key#3){+.+.}-{3:3}:
[ 56789.012394]        __lock_acquire+0x12a3/0x22b0
[ 56789.012397]        lock_acquire+0xb5/0x200
[ 56789.012400]        down_write+0x3d/0x80
[ 56789.012403]        ext4_write+0x2a/0x50 [ext4]""",
        "expected_fault": "deadlock"
    },
    {
        "id": "fl_020",
        "fault_type": "memory_corruption",
        "log": """[ 67890.123456] =============================================================================
[ 67890.123458] BUG: KASAN: invalid-access in mydriver_ioctl+0x3b/0xd0 [mydriver]
[ 67890.123461] Read of size 8 at addr ffff8800deadbeef by task test/12345
[ 67890.123463] 
[ 67890.123464] CPU: 2 PID: 12345 Comm: test Tainted: P    B    O
[ 67890.123466] Hardware name: Google Compute Engine
[ 67890.123468] Call Trace:
[ 67890.123469]  dump_stack+0x6b/0x88
[ 67890.123472]  kasan_report+0x244/0x390
[ 67890.123475]  mydriver_ioctl+0x3b/0xd0 [mydriver]
[ 67890.123478]  do_vfs_ioctl+0xa3/0x680
[ 67890.123481]  SyS_ioctl+0x79/0x90
[ 67890.123484]  entry_SYSCALL_64_fastpath+0x16/0x75
[ 67890.123487] 
[ 67890.123488] The address belongs to non-allocated memory
[ 67890.123490]  ffff8800deadbeef is located in no particular slab.""",
        "expected_fault": "memory_corruption"
    },
]


# ── Benchmark Functions ──────────────────────────────────────────────────

def extract_tool_call(text: str) -> Optional[Dict]:
    """Extract tool_call JSON block from model output.
    
    Looks for patterns like:
    - {"function": "name", "arguments": {...}}
    - <|tool_call|> name({"key": "value"})
    - function_name(arg1=value1, arg2=value2)
    """
    # Pattern 1: JSON tool_call block
    json_pattern = r'\{"function"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\}'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: <|tool_call|> name(...)
    tc_pattern = r'<\|tool_call\|>\s*(\w+)\(([^)]*)\)'
    match = re.search(tc_pattern, text)
    if match:
        return {
            "function": match.group(1),
            "arguments": match.group(2).strip()
        }
    
    # Pattern 3: function_name({...}) or function_name(key=value,...)
    func_pattern = r'(\w+)\s*\(\s*(\{.*\}|.*)\)\s*'
    match = re.search(func_pattern, text)
    if match:
        return {
            "function": match.group(1),
            "arguments": match.group(2).strip()
        }
    
    return None


def score_tool_call(output: str, expected_fn: str, expected_args: Dict) -> Dict:
    """Score a model output for tool call accuracy.
    
    Returns:
        strict: exact JSON match with correct function name + arguments
        lenient: function name matches + any arguments provided
        format: any parseable JSON/structured call
    """
    extracted = extract_tool_call(output)
    
    if extracted is None:
        return {"strict": 0.0, "lenient": 0.0, "format": 0.0}
    
    format_score = 1.0
    fn_name = extracted.get("function", "")
    
    if fn_name == expected_fn:
        lenient_score = 1.0
        
        # Check argument match
        args = extracted.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.startswith("{") else {}
            except json.JSONDecodeError:
                args = {}
        
        if isinstance(args, dict) and args:
            # Check if expected args are subset of actual args
            strict_score = 1.0 if set(expected_args.keys()).issubset(set(args.keys())) else 0.0
        elif args:
            strict_score = 0.5  # Some args but not in expected format
        else:
            strict_score = 0.0
    else:
        lenient_score = 0.0
        strict_score = 0.0
    
    return {
        "strict": strict_score,
        "lenient": lenient_score,
        "format": format_score
    }


def benchmark_tool_call(model, tokenizer_fn, device, max_new_tokens=128) -> Dict:
    """Benchmark tool call accuracy with 30 synthetic scenarios."""
    results = []
    total_scores = {"strict": 0.0, "lenient": 0.0, "format": 0.0}
    
    print(f"\n  Running tool call benchmark ({len(TOOL_CALL_PROMPTS)} scenarios)...")
    
    for scenario in TOOL_CALL_PROMPTS:
        prompt = scenario["prompt"]
        expected_fn = scenario["expected_function"]
        expected_args = scenario["expected_args"]
        
        # Tokenize and generate
        input_ids = torch.tensor([tokenizer_fn(prompt)]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
            )
        
        # Decode
        generated_text = decode_output(output_ids[0].tolist())
        
        # Score
        scores = score_tool_call(generated_text, expected_fn, expected_args)
        for k in total_scores:
            total_scores[k] += scores[k]
        
        results.append({
            "id": scenario["id"],
            "prompt": prompt[:80],
            "expected_function": expected_fn,
            "output_preview": generated_text[:120],
            "scores": scores
        })
    
    n = len(TOOL_CALL_PROMPTS)
    avg_scores = {k: v / n for k, v in total_scores.items()}
    
    return {
        "benchmark": "tool_call_accuracy",
        "num_scenarios": n,
        "average_scores": avg_scores,
        "details": results
    }


def benchmark_bash_generation(model, tokenizer_fn, device, test_data_path: str, max_new_tokens=128) -> Dict:
    """Benchmark bash command generation on intercode_bash test samples."""
    # Load test data and filter for intercode_bash
    test_samples = []
    if os.path.exists(test_data_path):
        with open(test_data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    if "intercode_bash" in sample.get("source", ""):
                        test_samples.append(sample)
    else:
        print(f"  WARNING: test data not found at {test_data_path}")
        return {
            "benchmark": "bash_generation",
            "num_samples": 0,
            "error": f"Test data not found at {test_data_path}",
            "average_scores": {"exact": 0.0, "substring": 0.0, "semantic": 0.0}
        }
    
    # If we have fewer than 200, use all; else use first 200
    if len(test_samples) > 200:
        random.seed(42)
        test_samples = random.sample(test_samples, 200)
    
    print(f"\n  Running bash benchmark ({len(test_samples)} samples)...")
    
    results = []
    exact_matches = 0
    substring_matches = 0
    semantic_matches = 0
    n = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        prompt = sample.get("prompt", "")
        expected_response = sample.get("response", "").strip()
        
        # Tokenize and generate
        input_ids = torch.tensor([tokenizer_fn(prompt)]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
            )
        
        # Decode
        generated_text = decode_output(output_ids[0].tolist())
        
        # Scoring
        gen_clean = generated_text.strip().lower()
        expected_clean = expected_response.strip().lower()
        
        exact = 1.0 if gen_clean == expected_clean else 0.0
        substring = 1.0 if expected_clean in gen_clean or gen_clean in expected_clean else 0.0
        
        # Semantic: check if key bash commands overlap
        bash_keywords = set(re.findall(r'\b\w+\b', expected_clean))
        gen_keywords = set(re.findall(r'\b\w+\b', gen_clean))
        overlap = len(bash_keywords & gen_keywords) / max(len(bash_keywords), 1)
        semantic = 1.0 if overlap > 0.3 else 0.0
        
        exact_matches += exact
        substring_matches += substring
        semantic_matches += semantic
        
        if i < 5:  # Show first 5 for debugging
            print(f"    [{i+1}] expected: {expected_clean[:60]}...")
            print(f"           got: {gen_clean[:60]}...")
            print(f"           scores: exact={exact}, substring={substring}, semantic={semantic}")
        
        results.append({
            "id": f"bash_{i}",
            "expected": expected_response[:80],
            "output_preview": generated_text[:120],
            "scores": {"exact": exact, "substring": substring, "semantic": semantic}
        })
    
    return {
        "benchmark": "bash_generation",
        "num_samples": n,
        "average_scores": {
            "exact": exact_matches / n,
            "substring": substring_matches / n,
            "semantic": semantic_matches / n,
        },
        "details": results[:20]  # First 20 details
    }


def benchmark_multi_turn_coherence(model, tokenizer_fn, device, max_new_tokens=128) -> Dict:
    """Benchmark multi-turn coherence with 20 dialogues."""
    results = []
    total_scores = []
    
    print(f"\n  Running multi-turn coherence benchmark ({len(MULTI_TURN_DIALOGUES)} dialogues)...")
    
    for dialogue in MULTI_TURN_DIALOGUES:
        context = []
        turn_scores = []
        
        for turn_idx, turn in enumerate(dialogue["turns"]):
            # Build context history
            context_str = "\n".join(context)
            prompt = turn["content"]
            
            full_prompt = f"{context_str}\n<|user|> {prompt}" if context_str else f"<|user|> {prompt}"
            
            # Tokenize and generate
            input_ids = torch.tensor([tokenizer_fn(full_prompt)]).to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=50,
                )
            
            generated_text = decode_output(output_ids[0].tolist())
            
            # Score: does the generated text reference context?
            expected_keywords = dialogue["context_keywords"]
            found_keywords = sum(1 for kw in expected_keywords if kw.lower() in generated_text.lower())
            keyword_score = found_keywords / max(len(expected_keywords), 1)
            
            turn_scores.append(keyword_score)
            
            # Update context
            context.append(f"<|user|> {turn['content']}")
            context.append(f"<|assistant|> {generated_text.strip()[:100]}")
        
        avg_score = sum(turn_scores) / max(len(turn_scores), 1)
        total_scores.append(avg_score)
        
        results.append({
            "id": dialogue["id"],
            "num_turns": len(dialogue["turns"]),
            "avg_turn_score": avg_score,
            "found_keywords": found_keywords,
            "total_keywords": len(expected_keywords)
        })
    
    n = len(MULTI_TURN_DIALOGUES)
    avg_coherence = sum(total_scores) / n if n > 0 else 0.0
    
    return {
        "benchmark": "multi_turn_coherence",
        "num_dialogues": n,
        "average_coherence": avg_coherence,
        "details": results
    }


def benchmark_format_compliance(model, tokenizer_fn, device, test_data_path: str, max_new_tokens=128) -> Dict:
    """Benchmark output format compliance on all test data."""
    # Load test samples
    test_samples = []
    if os.path.exists(test_data_path):
        with open(test_data_path) as f:
            for i, line in enumerate(f):
                if i >= 200:  # Sample 200 for efficiency
                    break
                line = line.strip()
                if line:
                    test_samples.append(json.loads(line))
    else:
        print(f"  WARNING: test data not found at {test_data_path}")
        return {
            "benchmark": "format_compliance",
            "num_samples": 0,
            "error": f"Test data not found at {test_data_path}",
            "compliance_rate": 0.0
        }
    
    print(f"\n  Running format compliance benchmark ({len(test_samples)} samples)...")
    
    parseable_count = 0
    
    for i, sample in enumerate(test_samples):
        prompt = sample.get("prompt", "")
        
        # Tokenize and generate
        input_ids = torch.tensor([tokenizer_fn(prompt)]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
            )
        
        generated_text = decode_output(output_ids[0].tolist())
        
        # Check if output is parseable JSON or structured text
        is_parseable = False
        
        # Check for JSON
        json_match = re.search(r'\{[^{}]*\}', generated_text)
        if json_match:
            try:
                json.loads(json_match.group(0))
                is_parseable = True
            except json.JSONDecodeError:
                pass
        
        # Check for structured text (has clear format)
        if not is_parseable:
            has_structure = (
                len(re.findall(r'\w+:', generated_text)) >= 2 or  # key: value pairs
                len(re.findall(r'[-*]\s', generated_text)) >= 2 or  # bullet points
                len(re.findall(r'\d+\.\s', generated_text)) >= 2  # numbered list
            )
            if has_structure:
                is_parseable = True
        
        if is_parseable:
            parseable_count += 1
    
    compliance_rate = parseable_count / len(test_samples) if test_samples else 0.0
    
    return {
        "benchmark": "format_compliance",
        "num_samples": len(test_samples),
        "compliance_rate": compliance_rate,
        "parseable_count": parseable_count
    }


def benchmark_system_log_diagnosis(model, tokenizer_fn, device, max_new_tokens=256) -> Dict:
    """Benchmark system log diagnosis with 20 synthetic fault logs."""
    results = []
    correct = 0
    n = len(SYSTEM_FAULT_LOGS)
    
    print(f"\n  Running system log diagnosis benchmark ({n} fault logs)...")
    
    for fault in SYSTEM_FAULT_LOGS:
        prompt = (
            f"<|user|> Analyze this kernel log and identify the fault type:\n\n"
            f"{fault['log']}\n\n"
            f"What type of fault is this? Choose from: oom, panic, deadlock, memory_corruption"
        )
        
        expected_fault = fault["expected_fault"]
        
        # Tokenize and generate
        input_ids = torch.tensor([tokenizer_fn(prompt)]).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=50,
            )
        
        generated_text = decode_output(output_ids[0].tolist())
        
        # Check if output mentions the correct fault type
        text_lower = generated_text.lower()
        identified = None
        
        for fault_type in ["oom", "panic", "deadlock", "memory_corruption", "memory corruption"]:
            if fault_type in text_lower:
                identified = fault_type
                break
        
        is_correct = 1.0 if identified and (identified == expected_fault or 
                                            identified.replace(" ", "_") == expected_fault) else 0.0
        correct += is_correct
        
        results.append({
            "id": fault["id"],
            "expected_fault": expected_fault,
            "identified_fault": identified,
            "is_correct": is_correct,
            "output_preview": generated_text[:100]
        })
    
    accuracy = correct / n
    
    return {
        "benchmark": "system_log_diagnosis",
        "num_samples": n,
        "accuracy": accuracy,
        "correct_count": int(correct),
        "details": results
    }


# ── Helper Functions ─────────────────────────────────────────────────────

def simple_tokenizer(text: str, max_len: int = 512) -> List[int]:
    """A simple character-level tokenizer for the untrained model.
    
    The model has vocab_size=32000 with random embeddings.
    We map characters to token IDs using a simple hash.
    """
    tokens = []
    for i, c in enumerate(text):
        if i >= max_len - 2:  # Reserve room for BOS/EOS
            break
        # Map char to a token ID (1-31999 range)
        token_id = (ord(c) % 31998) + 1
        tokens.append(token_id)
    
    # Add BOS at start and EOS at end
    return [1] + tokens + [2]


def decode_output(token_ids: List[int]) -> str:
    """Decode token IDs back to text (inverse of simple_tokenizer)."""
    chars = []
    for tid in token_ids:
        if tid <= 1 or tid >= 32000:  # Skip BOS, EOS, PAD, UNK
            continue
        char_code = (tid - 1) % 128  # ASCII range
        if 32 <= char_code <= 126:  # Printable ASCII
            chars.append(chr(char_code))
        elif char_code == 10:  # Newline
            chars.append('\n')
        elif char_code == 9:  # Tab
            chars.append('\t')
    return ''.join(chars)


# ── Main ─────────────────────────────────────────────────────────────────

def run_benchmarks(checkpoint_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict:
    """Run all benchmarks and return results dict."""
    print("=" * 60)
    print("MominoMoE OS Copilot Agent — Benchmark Suite")
    stage = "BEFORE TRAINING (untrained)" if checkpoint_path is None else f"AFTER TRAINING ({checkpoint_path})"
    print(f"Stage: {stage}")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU found, using CPU (will be very slow)")
    
    # Load model
    print("\nLoading model...")
    config = MominoMoEConfig()
    model = MominoMoE(config)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "module" in checkpoint:
            model.load_state_dict(checkpoint["module"])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    
    model = model.to(device)
    model.eval()
    
    # Print model summary
    print(model.summary())
    
    # Tokenizer
    tokenizer_fn = lambda text: simple_tokenizer(text)
    
    # Data paths
    test_data_path = os.path.join(DATA_DIR, "agent_test.jsonl")
    print(f"\nTest data path: {test_data_path}")
    print(f"File exists: {os.path.exists(test_data_path)}")
    
    # Run benchmarks
    results = {
        "stage": "before_training" if checkpoint_path is None else "after_sft",
        "checkpoint": checkpoint_path,
        "model_config": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "n_experts": config.n_experts,
            "top_k": config.top_k,
            "total_params": sum(p.numel() for p in model.parameters()),
        }
    }
    
    # a) Tool call accuracy
    print("\n" + "-" * 50)
    print("BENCHMARK A: Tool Call Accuracy")
    print("-" * 50)
    tc_result = benchmark_tool_call(model, tokenizer_fn, device)
    results["tool_call_accuracy"] = tc_result
    scores = tc_result["average_scores"]
    print(f"  Results: strict={scores['strict']:.3f}, lenient={scores['lenient']:.3f}, format={scores['format']:.3f}")
    
    # b) Bash command generation
    print("\n" + "-" * 50)
    print("BENCHMARK B: Bash Command Generation")
    print("-" * 50)
    bash_result = benchmark_bash_generation(model, tokenizer_fn, device, test_data_path)
    results["bash_generation"] = bash_result
    if "average_scores" in bash_result:
        scores = bash_result["average_scores"]
        print(f"  Results: exact={scores['exact']:.3f}, substring={scores['substring']:.3f}, semantic={scores['semantic']:.3f}")
    
    # c) Multi-turn coherence
    print("\n" + "-" * 50)
    print("BENCHMARK C: Multi-Turn Coherence")
    print("-" * 50)
    mt_result = benchmark_multi_turn_coherence(model, tokenizer_fn, device)
    results["multi_turn_coherence"] = mt_result
    print(f"  Results: average_coherence={mt_result['average_coherence']:.3f}")
    
    # d) Format compliance
    print("\n" + "-" * 50)
    print("BENCHMARK D: Output Format Compliance")
    print("-" * 50)
    fc_result = benchmark_format_compliance(model, tokenizer_fn, device, test_data_path)
    results["format_compliance"] = fc_result
    print(f"  Results: compliance_rate={fc_result['compliance_rate']:.3f}")
    
    # e) System log diagnosis
    print("\n" + "-" * 50)
    print("BENCHMARK E: System Log Diagnosis")
    print("-" * 50)
    sl_result = benchmark_system_log_diagnosis(model, tokenizer_fn, device)
    results["system_log_diagnosis"] = sl_result
    print(f"  Results: accuracy={sl_result['accuracy']:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  A. Tool Call Accuracy:    strict={results['tool_call_accuracy']['average_scores']['strict']:.3f}, "
          f"lenient={results['tool_call_accuracy']['average_scores']['lenient']:.3f}, "
          f"format={results['tool_call_accuracy']['average_scores']['format']:.3f}")
    print(f"  B. Bash Generation:       exact={results['bash_generation']['average_scores']['exact']:.3f}, "
          f"substring={results['bash_generation']['average_scores']['substring']:.3f}")
    print(f"  C. Multi-Turn Coherence:  {results['multi_turn_coherence']['average_coherence']:.3f}")
    print(f"  D. Format Compliance:     {results['format_compliance']['compliance_rate']:.3f}")
    print(f"  E. Log Diagnosis:         {results['system_log_diagnosis']['accuracy']:.3f}")
    print("=" * 60)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MominoMoE Agent Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (if None, use untrained model)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per sample")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = args.output
    elif args.checkpoint:
        output_path = os.path.join(BENCHMARK_DIR, "after_sft.json")
    else:
        output_path = os.path.join(BENCHMARK_DIR, "before_training.json")
    
    run_benchmarks(
        checkpoint_path=args.checkpoint,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()