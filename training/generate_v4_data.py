#!/usr/bin/env python3
"""
MominoMoE-v4 training data generator.
80k train / 8k val. Coding-first, diverse phrasing throughout.
Every task is asked in 4-8 different ways so the model generalises,
not pattern-matches.
"""
import random, json, argparse, os, textwrap

rng = random.Random(42)

# ─── helpers ──────────────────────────────────────────────────────────────────
def rand_pid():   return rng.randint(2, 65534)
def rand_port():  return rng.randint(1024, 65535)
def rand_ip():    return ".".join(str(rng.randint(1, 254)) for _ in range(4))
def rand_addr():  return hex(rng.randint(0x400000, 0xFFFF800000000000))
def rand_path(kind="any"):
    dirs  = ["/bin","/usr/bin","/var/log","/tmp","/home/user","/etc","/opt","/srv","/var/www"]
    logs  = ["/var/log/syslog","/var/log/auth.log","/var/log/kern.log","/var/log/nginx/error.log"]
    if kind == "log": return rng.choice(logs)
    if kind == "dir": return rng.choice(dirs)
    files = ["sh","ls","cat","nginx","sshd","bash","python3","app","daemon","worker","server"]
    return f"{rng.choice(dirs)}/{rng.choice(files)}"

PROC_NAMES = ["nginx","mysqld","python3","node","java","redis-server","postgres",
              "sshd","httpd","mongod","grafana","prometheus","celery","gunicorn"]
USERS      = ["root","www-data","nobody","user","daemon","sshd","postgres","ubuntu"]
SERVICES   = ["nginx","mysql","redis","ssh","cron","docker","postgresql",
              "mongod","elasticsearch","prometheus","grafana-server","fail2ban","ufw"]
IFACES     = ["eth0","ens3","lo","docker0","wlan0","bond0","br0","ens4"]

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
# 1. PYTHON CODING  (~18%)
# ══════════════════════════════════════════════════════════════════════════════

PYTHON_TASKS = [
    # (prompt_variants, code_answer)
    (
        ["Write a Python function that checks if a string is a palindrome.",
         "Implement is_palindrome() in Python.",
         "How do I check if a word reads the same forwards and backwards in Python?",
         "Give me a Python function: returns True if input string is a palindrome."],
        '''def is_palindrome(s: str) -> bool:
    s = s.lower().replace(" ", "")
    return s == s[::-1]'''
    ),
    (
        ["Write a Python function to do a binary search on a sorted list.",
         "Implement binary search in Python.",
         "Python binary search: return index of target in sorted array, -1 if not found.",
         "How do I search a sorted list efficiently in Python?"],
        '''def binary_search(arr: list, target) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1'''
    ),
    (
        ["Write a Python class for a stack with push, pop, and peek.",
         "Implement a Stack data structure in Python.",
         "Python Stack class: push, pop, peek, is_empty methods.",
         "Give me a stack implementation in Python using a list."],
        '''class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def __len__(self):
        return len(self._data)'''
    ),
    (
        ["Write a Python function to flatten a nested list.",
         "Python: flatten arbitrarily nested lists into one flat list.",
         "Implement flatten() in Python that works on any depth of nesting.",
         "How do I recursively flatten a list of lists in Python?"],
        '''def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result'''
    ),
    (
        ["Write a Python decorator that retries a function up to N times on exception.",
         "Implement a retry decorator in Python.",
         "Python decorator: retries the wrapped function up to `times` times with `delay` seconds between attempts.",
         "How do I make a function automatically retry on failure in Python?"],
        '''import time, functools

def retry(times=3, delay=1.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator'''
    ),
    (
        ["Write a Python function that merges two sorted lists into one sorted list.",
         "Implement merge of two sorted arrays in Python.",
         "Python: merge_sorted(a, b) returns a single sorted list without using sort().",
         "How do I combine two already-sorted lists in O(n) time in Python?"],
        '''def merge_sorted(a: list, b: list) -> list:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result'''
    ),
    (
        ["Write a Python context manager that times a block of code.",
         "Implement a Timer context manager in Python.",
         "Python: with Timer() as t: ... then print(t.elapsed) in seconds.",
         "How do I measure execution time of a code block in Python using 'with'?"],
        '''import time

class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

# Usage:
# with Timer() as t:
#     do_work()
# print(f"{t.elapsed:.3f}s")'''
    ),
    (
        ["Write a Python function to count word frequencies in a string.",
         "Python: word_count(text) returns a dict of word -> count.",
         "Implement a word frequency counter in Python.",
         "How do I count how many times each word appears in a string in Python?"],
        '''from collections import Counter

def word_count(text: str) -> dict:
    words = text.lower().split()
    return dict(Counter(words))'''
    ),
    (
        ["Write a Python generator that yields Fibonacci numbers.",
         "Implement an infinite Fibonacci generator in Python.",
         "Python: fibonacci() yields 0, 1, 1, 2, 3, 5, 8, ... indefinitely.",
         "How do I create a lazy Fibonacci sequence in Python?"],
        '''def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage: list(itertools.islice(fibonacci(), 10))'''
    ),
    (
        ["Write a Python function to check if a binary tree is balanced.",
         "Python: is_balanced(root) returns True if the tree's height difference is at most 1 at every node.",
         "Implement balanced binary tree check in Python.",
         "How do I verify a BST is height-balanced in Python?"],
        '''class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def is_balanced(root) -> bool:
    def height(node):
        if node is None:
            return 0
        lh = height(node.left)
        if lh == -1:
            return -1
        rh = height(node.right)
        if rh == -1:
            return -1
        if abs(lh - rh) > 1:
            return -1
        return max(lh, rh) + 1
    return height(root) != -1'''
    ),
    (
        ["Write a Python function that parses a log line and extracts timestamp, level, and message.",
         "Python: parse_log_line() extracts timestamp, level (INFO/WARN/ERROR), and message from a log string.",
         "Implement a log line parser in Python using regex.",
         "How do I extract fields from a structured log line in Python?"],
        '''import re

LOG_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    r"\s+(?P<msg>.+)"
)

def parse_log_line(line: str) -> dict | None:
    m = LOG_RE.match(line.strip())
    if not m:
        return None
    return m.groupdict()'''
    ),
    (
        ["Write a Python function that reads a JSON file and returns a dict.",
         "Python: load_json(path) safely loads a JSON file and returns its contents.",
         "Implement a JSON file loader in Python with error handling.",
         "How do I read a JSON config file in Python?"],
        '''import json
from pathlib import Path

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with p.open() as f:
        return json.load(f)'''
    ),
    (
        ["Write a Python function that implements LRU cache from scratch.",
         "Implement an LRU (Least Recently Used) cache in Python without using functools.",
         "Python LRUCache class: get(key) and put(key, value) both in O(1).",
         "How do I build an LRU cache using OrderedDict in Python?"],
        '''from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)'''
    ),
    (
        ["Write a Python function to find all permutations of a list.",
         "Python: permutations(items) returns a list of all orderings.",
         "Implement permutation generation in Python without itertools.",
         "How do I generate every possible ordering of a list in Python?"],
        '''def permutations(items: list) -> list:
    if len(items) <= 1:
        return [list(items)]
    result = []
    for i, item in enumerate(items):
        rest = items[:i] + items[i+1:]
        for perm in permutations(rest):
            result.append([item] + perm)
    return result'''
    ),
    (
        ["Write a Python async function that fetches multiple URLs concurrently.",
         "Python asyncio: fetch_all(urls) fetches all URLs in parallel and returns responses.",
         "Implement concurrent HTTP requests in Python using aiohttp.",
         "How do I download multiple URLs at the same time in Python?"],
        '''import asyncio
import aiohttp

async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for r in responses:
            if isinstance(r, Exception):
                results.append(f"ERROR: {r}")
            else:
                async with r:
                    results.append(await r.text())
        return results'''
    ),
    (
        ["Write a Python function that deep-copies a nested dictionary.",
         "Python: deep_copy(d) returns a fully independent copy of a nested dict.",
         "Implement deep copy for dicts in Python without using copy module.",
         "How do I clone a nested dictionary in Python so changes don't affect the original?"],
        '''def deep_copy(obj):
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_copy(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy(i) for i in obj)
    return obj'''
    ),
    (
        ["Write a Python function that groups a list of dicts by a key.",
         "Python: group_by(items, key) groups a list of dicts by the given field.",
         "Implement groupBy in Python — like SQL GROUP BY but for a list of dicts.",
         "How do I organise a list of records by a shared field in Python?"],
        '''from collections import defaultdict

def group_by(items: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return dict(groups)'''
    ),
    (
        ["Write a Python function to compute a rolling average over a list.",
         "Python: rolling_average(data, window) returns a list of averages over a sliding window.",
         "Implement a moving average function in Python.",
         "How do I calculate a sliding-window mean in Python?"],
        '''def rolling_average(data: list[float], window: int) -> list[float]:
    if window > len(data):
        return []
    return [
        sum(data[i:i+window]) / window
        for i in range(len(data) - window + 1)
    ]'''
    ),
    (
        ["Write a Python script that tails a file and prints new lines as they appear.",
         "Python: tail_file(path) continuously reads new lines from a growing log file.",
         "Implement 'tail -f' behaviour in Python.",
         "How do I watch a log file for new lines in Python?"],
        '''import time

def tail_file(path: str):
    with open(path) as f:
        f.seek(0, 2)          # seek to end
        while True:
            line = f.readline()
            if line:
                print(line, end="")
            else:
                time.sleep(0.1)'''
    ),
    (
        ["Write a Python function to validate an IPv4 address.",
         "Python: is_valid_ipv4(addr) returns True if the string is a valid IPv4 address.",
         "Implement IPv4 validation in Python without using socket or ipaddress.",
         "How do I check if a string is a valid IP address in Python?"],
        '''def is_valid_ipv4(addr: str) -> bool:
    parts = addr.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        if not 0 <= int(part) <= 255:
            return False
        if len(part) > 1 and part[0] == "0":   # no leading zeros
            return False
    return True'''
    ),
]

def make_python_coding():
    variants, answer = rng.choice(PYTHON_TASKS)
    prompt = rng.choice(variants)
    return {"prompt": prompt, "response": f"```python\n{answer}\n```"}


# ══════════════════════════════════════════════════════════════════════════════
# 2. CODE DEBUGGING  (~8%)
# ══════════════════════════════════════════════════════════════════════════════

BUG_TASKS = [
    (
        ["What is wrong with this Python code?\n\n```python\ndef divide(a, b):\n    return a / b\n\nresult = divide(10, 0)\nprint(result)\n```",
         "Find the bug:\n\n```python\ndef divide(a, b):\n    return a / b\nprint(divide(10, 0))\n```",
         "Why does this crash?\n\n```python\ndef divide(a, b):\n    return a / b\ndivide(10, 0)\n```"],
        "Bug: ZeroDivisionError — no guard against `b == 0`.\n\nFix:\n```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError(\"divisor cannot be zero\")\n    return a / b\n```"
    ),
    (
        ["What's wrong with this code?\n\n```python\ndef get_first(lst):\n    return lst[0]\n\nprint(get_first([]))\n```",
         "Find the bug:\n\n```python\ndef get_first(items):\n    return items[0]\nget_first([])\n```"],
        "Bug: IndexError — no check for empty list.\n\nFix:\n```python\ndef get_first(lst):\n    if not lst:\n        return None\n    return lst[0]\n```"
    ),
    (
        ["Why is this loop infinite?\n\n```python\ni = 0\nwhile i < 10:\n    print(i)\n```",
         "What's the bug?\n\n```python\ni = 0\nwhile i < 10:\n    print(i)\n# never stops\n```"],
        "Bug: `i` is never incremented — infinite loop.\n\nFix:\n```python\ni = 0\nwhile i < 10:\n    print(i)\n    i += 1\n```"
    ),
    (
        ["What's wrong with this Python function?\n\n```python\ndef append_to(item, lst=[]):\n    lst.append(item)\n    return lst\n\nprint(append_to(1))\nprint(append_to(2))\n```",
         "Find the bug — this prints [1] then [1, 2] instead of [1] and [2]:\n\n```python\ndef append_to(item, lst=[]):\n    lst.append(item)\n    return lst\n```"],
        "Bug: mutable default argument. The default `lst=[]` is created once and shared across all calls.\n\nFix:\n```python\ndef append_to(item, lst=None):\n    if lst is None:\n        lst = []\n    lst.append(item)\n    return lst\n```"
    ),
    (
        ["Fix this Python code that's supposed to read a file safely:\n\n```python\ndef read_file(path):\n    f = open(path)\n    data = f.read()\n    return data\n```",
         "What's wrong with this file reading code?\n\n```python\ndef read_file(path):\n    f = open(path)\n    return f.read()\n```"],
        "Bug: file handle is never closed — resource leak.\n\nFix:\n```python\ndef read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()\n```\nUsing `with` guarantees the file is closed even if an exception occurs."
    ),
    (
        ["Why does this return wrong results?\n\n```python\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # prints 0\n```",
         "Find the bug in this factorial function:\n\n```python\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n```"],
        "Bug: base case returns 0 instead of 1. `factorial(0)` should be 1, so the product chain doesn't collapse to 0.\n\nFix:\n```python\ndef factorial(n: int) -> int:\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n```"
    ),
    (
        ["What's wrong with this dictionary iteration?\n\n```python\nd = {'a': 1, 'b': 2, 'c': 3}\nfor k in d:\n    if d[k] == 2:\n        del d[k]\n```",
         "Fix this code that tries to delete dict entries while iterating:\n\n```python\nfor k in d:\n    if d[k] == 2:\n        del d[k]\n```"],
        "Bug: RuntimeError — you cannot delete keys from a dict while iterating over it.\n\nFix:\n```python\nd = {k: v for k, v in d.items() if v != 2}\n# or:\nkeys_to_delete = [k for k, v in d.items() if v == 2]\nfor k in keys_to_delete:\n    del d[k]\n```"
    ),
    (
        ["Why does this Python class have a bug?\n\n```python\nclass Counter:\n    count = 0\n    def increment(self):\n        self.count += 1\n\na = Counter()\nb = Counter()\na.increment()\nprint(b.count)  # prints 1, not 0\n```",
         "Find the issue with shared class state:\n\n```python\nclass Counter:\n    count = 0\n    def increment(self):\n        self.count += 1\n```"],
        "Bug: `count` is a class variable, shared between all instances. `a.increment()` creates an instance variable `a.count=1` but `b.count` still reads the class variable.\n\nFix:\n```python\nclass Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1\n```"
    ),
    (
        ["What's wrong with this bash script?\n\n```bash\n#!/bin/bash\nif [ $1 = \"hello\" ]\nthen\n    echo \"hi\"\nfi\n```",
         "Fix this bash conditional:\n\n```bash\nif [ $1 = \"hello\" ]; then echo \"hi\"; fi\n```"],
        'Bug: unquoted `$1` causes a syntax error if the argument is empty or contains spaces.\n\nFix:\n```bash\n#!/bin/bash\nif [ "$1" = "hello" ]; then\n    echo "hi"\nfi\n```\nAlways quote variables in shell conditionals.'
    ),
    (
        ["Why does this bash loop skip files with spaces in their names?\n\n```bash\nfor f in $(ls /tmp); do\n    echo $f\ndone\n```",
         "Fix this bash loop that breaks on filenames with spaces:\n\n```bash\nfor f in $(ls /tmp); do echo $f; done\n```"],
        "Bug: `$(ls)` word-splits on spaces, breaking multi-word filenames.\n\nFix:\n```bash\nfor f in /tmp/*; do\n    echo \"$f\"\ndone\n```\nNever parse `ls`. Use glob patterns directly and quote `\"$f\"`."
    ),
    (
        ["Find the off-by-one error:\n\n```python\ndef last_n(lst, n):\n    return lst[len(lst) - n : len(lst) + 1]\n\nprint(last_n([1,2,3,4,5], 3))  # should be [3,4,5]\n```",
         "What's wrong? Should return last n elements but returns extra:\n\n```python\ndef last_n(lst, n):\n    return lst[len(lst)-n : len(lst)+1]\n```"],
        "Bug: end index is `len(lst)+1` which goes out of bounds (Python silently returns to end anyway, but the intent is wrong — use `len(lst)` or just omit the end).\n\nFix:\n```python\ndef last_n(lst: list, n: int) -> list:\n    return lst[-n:]\n```"
    ),
    (
        ["Why does this SQL injection vulnerability exist and how do you fix it?\n\n```python\ndef get_user(username):\n    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n    return db.execute(query)\n```",
         "Find the security bug:\n\n```python\nquery = f\"SELECT * FROM users WHERE name = '{username}'\"\ndb.execute(query)\n```"],
        "Bug: SQL injection. If `username = \"' OR '1'='1\"`, the query returns all users.\n\nFix — use parameterised queries:\n```python\ndef get_user(username: str):\n    return db.execute(\n        \"SELECT * FROM users WHERE name = ?\",\n        (username,)\n    )\n```\nNever interpolate user input into SQL strings."
    ),
]

def make_code_debug():
    variants, answer = rng.choice(BUG_TASKS)
    return {"prompt": rng.choice(variants), "response": answer}


# ══════════════════════════════════════════════════════════════════════════════
# 3. BASH SCRIPTING  (~10%)  — diverse phrasing, correct syntax
# ══════════════════════════════════════════════════════════════════════════════

BASH_TASKS = [
    # Each entry: (list_of_prompt_variants, answer)
    (
        ["Find all files larger than {size}MB under {path} and delete them.",
         "One-liner: delete every file bigger than {size}MB inside {path}.",
         "How do I remove large files (over {size}MB) from {path}?",
         "Bash command to clean up files over {size}M in {path}.",
         "Delete files exceeding {size} megabytes recursively under {path}."],
        "find {path} -type f -size +{size}M -delete"
    ),
    (
        ["Count the number of lines in {path}.",
         "How many lines does {path} have?",
         "Print the line count of {path}.",
         "bash: line count for file {path}."],
        "wc -l {path}"
    ),
    (
        ["Show the 10 largest files under {path}.",
         "List the biggest files in {path} sorted by size.",
         "What are the top 10 files by size under {path}?",
         "Find and rank the largest files in {path}."],
        "du -ah {path} | sort -rh | head -10"
    ),
    (
        ["Kill all processes matching the name {proc}.",
         "Force-kill every {proc} process.",
         "How do I kill all instances of {proc} at once?",
         "Terminate every running {proc}."],
        "pkill -9 {proc}"
    ),
    (
        ["Show all open TCP connections with process names.",
         "List every TCP socket and which process owns it.",
         "What processes are using network connections?",
         "Print active TCP connections with PIDs."],
        "ss -tnp"
    ),
    (
        ["Tail /var/log/syslog and follow new output.",
         "Stream new lines from syslog as they arrive.",
         "Watch /var/log/syslog in real time.",
         "Follow syslog output live."],
        "tail -n 100 -f /var/log/syslog"
    ),
    (
        ["Find all files modified in the last 24 hours under /etc.",
         "Which files in /etc changed in the past day?",
         "List recently modified config files under /etc.",
         "Show files touched in the last 24h inside /etc."],
        "find /etc -mtime -1 -type f"
    ),
    (
        ["Show disk usage of each subdirectory under {path}, sorted by size.",
         "Which subdirectories of {path} are using the most space?",
         "List disk usage per folder under {path} biggest first.",
         "Breakdown disk space usage under {path}."],
        "du -sh {path}/* | sort -rh"
    ),
    (
        ["Check which process is listening on port {port}.",
         "What is using port {port}?",
         "Find the PID listening on port {port}.",
         "Which service is bound to port {port}?"],
        "ss -tlnp 'sport = :{port}'"
    ),
    (
        ["Archive and compress {path} into /tmp/backup.tar.gz.",
         "Create a gzip-compressed tar of {path}.",
         "Tar up {path} and save as /tmp/backup.tar.gz.",
         "How do I back up {path} with tar?"],
        "tar -czf /tmp/backup.tar.gz {path}"
    ),
    (
        ["Show the last 50 kernel messages.",
         "Print recent kernel log output.",
         "What has the kernel logged recently?",
         "Display the last 50 lines of dmesg."],
        "dmesg | tail -50"
    ),
    (
        ["Find all SUID binaries on the system.",
         "List every setuid executable.",
         "Which binaries have the SUID bit set?",
         "Audit the system for SUID files."],
        "find / -perm -4000 -type f 2>/dev/null"
    ),
    (
        ["Check if port {port} is open on {host}.",
         "Test connectivity to {host}:{port}.",
         "Is {host} listening on port {port}?",
         "nc: check reachability of {host} port {port}."],
        "nc -zv -w3 {host} {port}"
    ),
    (
        ["List all zombie processes.",
         "Which processes are in zombie state?",
         "Find Z-state processes on the system.",
         "Show any zombie PIDs."],
        "ps aux | awk '$8==\"Z\"'"
    ),
    (
        ["Show the top 5 CPU-consuming processes.",
         "Which processes are using the most CPU?",
         "List the 5 biggest CPU hogs right now.",
         "Print top processes sorted by CPU usage."],
        "ps aux --sort=-%cpu | head -6"
    ),
    (
        ["Count failed SSH login attempts in auth.log.",
         "How many SSH brute-force attempts are in auth.log?",
         "Find the number of failed password attempts from auth.log.",
         "Count 'Failed password' lines in /var/log/auth.log."],
        "grep -c 'Failed password' /var/log/auth.log"
    ),
    (
        ["Show environment variables of process {pid}.",
         "What env vars does PID {pid} have?",
         "Print the environment of process {pid}.",
         "Dump /proc/{pid}/environ in readable form."],
        "cat /proc/{pid}/environ | tr '\\0' '\\n'"
    ),
    (
        ["Show all failed systemd services.",
         "Which systemd units have failed?",
         "List services in failed state.",
         "What is broken in systemd right now?"],
        "systemctl --failed"
    ),
    (
        ["Print the 5 most recently modified files under {path}.",
         "What files under {path} were changed most recently?",
         "Find the newest files in {path}.",
         "Sort files in {path} by modification time, newest first."],
        "find {path} -type f -printf '%T@ %p\\n' | sort -rn | head -5 | awk '{print $2}'"
    ),
    (
        ["Recursively change ownership of {path} to {user}.",
         "Set {user} as owner of everything under {path}.",
         "chown {path} and all its contents to {user}.",
         "Transfer ownership of {path} to user {user}."],
        "chown -R {user}:{user} {path}"
    ),
    (
        ["Check for processes with deleted exe binaries.",
         "Find processes whose executable has been deleted (possible rootkit indicator).",
         "Which running processes have a missing binary on disk?",
         "Detect in-memory-only processes."],
        "ls -la /proc/*/exe 2>/dev/null | grep '(deleted)'"
    ),
    (
        ["Show memory map of process {pid}.",
         "What memory regions does PID {pid} have?",
         "Print the VMAs of process {pid}.",
         "Display /proc/{pid}/maps."],
        "cat /proc/{pid}/maps"
    ),
    (
        ["Watch network interface {iface} packet rates.",
         "Monitor packets on {iface} in real time.",
         "Show live traffic stats for interface {iface}.",
         "Poll {iface} statistics every second."],
        "watch -n 1 'ip -s link show {iface}'"
    ),
    (
        ["Show routing table.",
         "Print all routes on this machine.",
         "What are the IP routes configured?",
         "Display the kernel routing table."],
        "ip route show"
    ),
    (
        ["Write a bash script that checks if {service} is running and starts it if not.",
         "Bash: restart {service} if it's down.",
         "Auto-heal script: ensure {service} is always running.",
         "Shell script: start {service} when it stops."],
        "#!/usr/bin/env bash\nif ! systemctl is-active --quiet {service}; then\n    echo '{service} is down, starting...'\n    systemctl start {service}\nfi"
    ),
    (
        ["Write a bash script that emails an alert if disk at {path} exceeds 90%.",
         "Alert via email when {path} disk usage goes above 90%.",
         "Bash disk-usage alert script for {path}.",
         "Script: send email if {path} is more than 90% full."],
        "#!/usr/bin/env bash\nUSAGE=$(df {path} | awk 'NR==2{{print $5}}' | tr -d '%')\nif [ \"$USAGE\" -gt 90 ]; then\n    echo \"Disk at {path} is ${{USAGE}}% full\" | mail -s 'Disk Alert' admin@example.com\nfi"
    ),
    (
        ["Write a bash function that retries a command up to 5 times.",
         "Bash retry wrapper: run a command again on failure, up to 5 attempts.",
         "Implement a retry loop in bash.",
         "Bash: re-run a failing command with back-off."],
        "retry() {\n    local max=5 delay=2 cmd=\"$@\"\n    for i in $(seq 1 $max); do\n        \"$@\" && return 0\n        echo \"Attempt $i/$max failed. Retrying in ${delay}s...\"\n        sleep $delay\n    done\n    echo \"All $max attempts failed.\"; return 1\n}"
    ),
    (
        ["Print the top 5 IPs hitting an nginx access log.",
         "Find the most frequent client IPs in nginx access.log.",
         "Which IPs are sending the most requests to nginx?",
         "Rank IPs by request count from nginx log."],
        "awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head -5"
    ),
    (
        ["Kill any process using more than 80% CPU.",
         "Auto-kill CPU hogs above 80%.",
         "Script: terminate processes exceeding 80% CPU.",
         "Bash: kill -9 anything over 80% CPU."],
        "ps aux | awk 'NR>1 && $3>80 {print $2}' | xargs -r kill -9"
    ),
    (
        ["Show open file handles for process {pid}.",
         "List all FDs held by PID {pid}.",
         "What files does process {pid} have open?",
         "Print /proc/{pid}/fd contents."],
        "ls -la /proc/{pid}/fd"
    ),
]

def make_bash_cmd():
    variants, answer_tmpl = rng.choice(BASH_TASKS)
    kw = dict(pid=rand_pid(), size=rng.choice([10,50,100,200,500]),
              path=rand_path("dir"), user=rng.choice(USERS),
              host=rand_ip(), port=rand_port(), service=rng.choice(SERVICES),
              proc=rng.choice(PROC_NAMES), iface=rng.choice(IFACES))
    try:
        prompt  = rng.choice(variants).format(**kw)
        answer  = answer_tmpl.format(**kw)
    except KeyError:
        prompt  = rng.choice(variants)
        answer  = answer_tmpl
    return {"prompt": prompt, "response": f"```bash\n{answer}\n```"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. C / SYSTEMS CODING  (~5%)
# ══════════════════════════════════════════════════════════════════════════════

C_TASKS = [
    (
        ["Write a C function to safely copy a string with a max length.",
         "Implement strncpy replacement in C that always null-terminates.",
         "C: safe_strcpy(dst, src, dst_size) — never overflows, always null-terminates."],
        '''#include <stddef.h>

void safe_strcpy(char *dst, const char *src, size_t dst_size) {
    if (dst_size == 0) return;
    size_t i;
    for (i = 0; i < dst_size - 1 && src[i]; i++)
        dst[i] = src[i];
    dst[i] = \'\\0\';
}'''
    ),
    (
        ["Write a C function that checks if a pointer is NULL before dereferencing.",
         "C: safe_deref(ptr) returns the int value or -1 if ptr is NULL.",
         "Implement null-safe pointer dereference in C."],
        '''#include <stdio.h>

int safe_deref(const int *ptr) {
    if (ptr == NULL) {
        fprintf(stderr, "NULL pointer dereference avoided\\n");
        return -1;
    }
    return *ptr;
}'''
    ),
    (
        ["Write a C ring buffer implementation with push and pop.",
         "Implement a fixed-size circular buffer in C.",
         "C: ring_push() and ring_pop() for a circular byte buffer."],
        '''#define RING_SIZE 64

typedef struct {
    uint8_t buf[RING_SIZE];
    volatile uint32_t head;
    volatile uint32_t tail;
} RingBuf;

int ring_push(RingBuf *r, uint8_t b) {
    uint32_t next = (r->head + 1) % RING_SIZE;
    if (next == r->tail) return -1;   /* full */
    r->buf[r->head] = b;
    r->head = next;
    return 0;
}

int ring_pop(RingBuf *r, uint8_t *out) {
    if (r->tail == r->head) return -1; /* empty */
    *out = r->buf[r->tail];
    r->tail = (r->tail + 1) % RING_SIZE;
    return 0;
}'''
    ),
    (
        ["Write a C function to open a file, read it fully, and return the contents.",
         "C: read_file(path) allocates and returns the file contents as a char*.",
         "Implement a file slurp function in C."],
        '''#include <stdio.h>
#include <stdlib.h>

char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    rewind(f);
    char *buf = malloc(len + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, len, f);
    buf[len] = \'\\0\';
    fclose(f);
    if (len_out) *len_out = len;
    return buf;   /* caller must free() */
}'''
    ),
    (
        ["Write a C linked list with insert and delete.",
         "Implement a singly-linked list in C with insert_head and delete_val.",
         "C: linked list node struct, insert at head, delete by value."],
        '''#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node *next;
} Node;

Node *insert_head(Node *head, int val) {
    Node *n = malloc(sizeof(Node));
    n->val = val;
    n->next = head;
    return n;
}

Node *delete_val(Node *head, int val) {
    Node dummy = {0, head}, *prev = &dummy;
    while (prev->next) {
        if (prev->next->val == val) {
            Node *del = prev->next;
            prev->next = del->next;
            free(del);
            break;
        }
        prev = prev->next;
    }
    return dummy.next;
}'''
    ),
    (
        ["Write a C function that counts bits set to 1 in an integer.",
         "Implement popcount in C without using __builtin_popcount.",
         "C: count the number of set bits in a 32-bit integer."],
        '''int popcount(uint32_t n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

/* Brian Kernighan\'s faster version:
int popcount(uint32_t n) {
    int count = 0;
    while (n) { n &= n - 1; count++; }
    return count;
} */'''
    ),
    (
        ["Write a C function that converts a string to an integer safely.",
         "Implement atoi in C with error detection.",
         "C: str_to_int(s, out) returns 0 on success, -1 on invalid input."],
        '''#include <limits.h>

int str_to_int(const char *s, long *out) {
    if (!s || !*s) return -1;
    char *end;
    long val = strtol(s, &end, 10);
    if (*end != \'\\0\') return -1;       /* trailing garbage */
    if (val > INT_MAX || val < INT_MIN) return -1;  /* overflow */
    *out = val;
    return 0;
}'''
    ),
]

def make_c_coding():
    variants, answer = rng.choice(C_TASKS)
    return {"prompt": rng.choice(variants), "response": f"```c\n{answer}\n```"}


# ══════════════════════════════════════════════════════════════════════════════
# 5. CODE EXPLANATION  (~4%)
# ══════════════════════════════════════════════════════════════════════════════

EXPLAIN_TASKS = [
    (
        ["Explain what this Python code does:\n\n```python\nresult = [x**2 for x in range(10) if x % 2 == 0]\n```",
         "What does this one-liner produce?\n\n```python\n[x**2 for x in range(10) if x % 2 == 0]\n```"],
        "This is a list comprehension that builds a list of squares of even numbers from 0 to 9.\n\n- `range(10)` → 0,1,2,...,9\n- `if x % 2 == 0` → keep only even numbers: 0,2,4,6,8\n- `x**2` → square each\n\nResult: `[0, 4, 16, 36, 64]`"
    ),
    (
        ["What does this bash command do?\n\n```bash\nfind /var/log -mtime +7 -name '*.log' -exec gzip {} \\;\n```",
         "Explain:\n\n```bash\nfind /var/log -mtime +7 -name '*.log' -exec gzip {} \\;\n```"],
        "Finds all `.log` files under `/var/log` that were last modified more than 7 days ago, and gzip-compresses each one.\n\n- `-mtime +7` → modified more than 7 days ago\n- `-name '*.log'` → only .log files\n- `-exec gzip {} \\;` → run `gzip` on each found file"
    ),
    (
        ["What does this Python decorator do?\n\n```python\ndef singleton(cls):\n    instances = {}\n    def get(*args, **kw):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kw)\n        return instances[cls]\n    return get\n```",
         "Explain the singleton decorator:\n\n```python\ndef singleton(cls):\n    instances = {}\n    def get(*args, **kw):\n        if cls not in instances:\n            instances[cls] = cls(*args, **kw)\n        return instances[cls]\n    return get\n```"],
        "This is a Singleton decorator. It ensures only one instance of a class is ever created.\n\n- `instances` is a dict capturing one instance per class\n- On first call: creates the instance and stores it\n- On subsequent calls: returns the cached instance\n\nUsage: `@singleton` above a class definition. Every `MyClass()` call returns the same object."
    ),
    (
        ["What does this awk command do?\n\n```bash\nawk '$3 > 50 {print $1, $3}' data.txt\n```",
         "Explain:\n\n```bash\nawk '$3 > 50 {print $1, $3}' data.txt\n```"],
        "Prints the 1st and 3rd fields (columns) of every line in `data.txt` where the 3rd field is greater than 50.\n\n- `$3 > 50` → filter condition (3rd whitespace-separated field)\n- `{print $1, $3}` → action: print columns 1 and 3 for matching lines"
    ),
    (
        ["What does this Python code do?\n\n```python\nwith open('file.txt') as f:\n    data = f.read()\nlines = data.split('\\n')\nresult = {i: l for i, l in enumerate(lines) if l.strip()}\n```",
         "Explain step by step:\n\n```python\nwith open('file.txt') as f:\n    data = f.read()\nlines = data.split('\\n')\nresult = {i: l for i, l in enumerate(lines) if l.strip()}\n```"],
        "1. Opens `file.txt` and reads its entire contents into `data`.\n2. Splits `data` by newlines into a list `lines`.\n3. Builds a dict mapping line number → line content, but only for non-empty lines (`if l.strip()` skips blank lines).\n\nResult: `{0: 'first line', 2: 'third line', ...}` (line numbers of non-blank lines)."
    ),
]

def make_code_explain():
    variants, answer = rng.choice(EXPLAIN_TASKS)
    return {"prompt": rng.choice(variants), "response": answer}


# ══════════════════════════════════════════════════════════════════════════════
# 6. KERNEL FAULT DIAGNOSIS  (~10%)
# ══════════════════════════════════════════════════════════════════════════════

FAULT_NAMES = {14:"Page Fault",13:"General Protection Fault",8:"Double Fault",
               0:"Divide by Zero",6:"Invalid Opcode",11:"Segment Not Present",12:"Stack Segment Fault"}
ERR_DESCS = {0x0:"read, page not present, kernel",0x2:"write, page not present, kernel",
             0x4:"read, page not present, user",0x6:"write, page not present, user",
             0x3:"read, protection violation, kernel",0x7:"write, protection violation, user"}

def make_kernel_fault():
    vector = rng.choices([14,13,8,0,6,11,12],weights=[30,20,5,10,8,12,15])[0]
    err  = rng.choice(list(ERR_DESCS.keys()))
    rip  = rand_addr(); cr2 = hex(rng.choice([0x0,0x8,0x10,0x18,rng.randint(0x1,0x80)]))
    tid  = rand_pid() % 512; rax = hex(rng.randint(0,0xFFFFFFFF)); rsp = rand_addr()
    syscalls = rng.sample([f"SYS_OPEN {rand_path()} 0 -> {rng.randint(3,255)}",
        f"SYS_READ {rng.randint(3,255)} 4096 -> 4096",f"SYS_MMAP 0x0 4096 3 -> {rand_addr()}",
        f"SYS_WRITE {rng.randint(3,255)} 128 -> 128",f"SYS_CLOSE {rng.randint(3,255)} -> 0",
        f"SYS_FUTEX {rand_addr()} 0 -> 0"],k=rng.randint(2,4))
    logs = rng.sample(["[VFS] opened "+rand_path(),f"[SCHED] thread {tid} running",
        f"[VMM] mapped {rand_addr()}->{rand_addr()}","[IRQ] timer tick","[MM] oom adj"],k=rng.randint(2,3))
    prompt = (f"[FAULT] vector={vector} ({FAULT_NAMES[vector]}) err=0x{err:04X} rip={rip} cr2={cr2} tid={tid} cwd=/bin\n\n"
              f"[REGISTERS] rax={rax} rdi={cr2} rsi=0x100 rsp={rsp}\n\n"
              f"[RECENT_SYSCALLS]\n"+"".join(f"  {s}\n" for s in syscalls)+"\n"
              f"[LOG]\n"+"".join(f"  {l}\n" for l in logs)+"\n"
              f"[QUERY] Diagnose this fault and suggest a corrective action.")
    is_null = int(cr2,16)<0x100; ed = ERR_DESCS.get(err,"unknown")
    if vector==14:
        ft = "Null pointer dereference" if is_null else "Invalid memory access"
        rc = (f"CR2={cr2} is near-null ({ed})." if is_null else f"CR2={cr2} unmapped ({ed}).")+" RIP="+rip+"."
        ac = ("Add NULL check before dereference; inspect for use-after-free." if is_null
              else "Validate pointer bounds; check stack canaries.")
    elif vector==13:
        ft="General Protection Fault"; rc=f"Privilege violation at RIP={rip}. err={hex(err)}={ed}."; ac="Verify segment selectors and privilege rings."
    elif vector==0:
        ft="Divide by Zero"; rc=f"Division by zero at RIP={rip}. RAX={rax} was dividend."; ac="Add divisor != 0 guard before the division."
    elif vector==6:
        ft="Invalid Opcode"; rc=f"Undefined instruction at RIP={rip}. Possible memory corruption."; ac="Verify binary integrity; check for heap/stack corruption."
    elif vector==8:
        ft="Double Fault"; rc=f"Exception during exception handling. Stack likely exhausted. RSP={rsp}."; ac="Increase kernel stack; check for interrupt handler recursion."
    else:
        ft="Segment Fault"; rc=f"Segment fault at RIP={rip}. RSP={rsp}."; ac="Validate segment selectors in context switch code."
    return {"prompt":prompt,"response":f"Fault type: {ft}.\nRoot cause: {rc}\nerr=0x{err:04X} = {ed}.\nCorrective action: {ac}"}


# ══════════════════════════════════════════════════════════════════════════════
# 7. TOOL CALLS — single-step  (~16%)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_TASKS = [
    (["Read {path}.", "Show me the contents of {path}.", "Open and display {path}.", "Cat {path}."],
     '{{"tool":"read_file","args":{{"path":"{path}"}}}}'),
    (["List {path}.", "What's in {path}?", "Show directory {path}.", "ls {path}."],
     '{{"tool":"list_dir","args":{{"path":"{path}"}}}}'),
    (["Kill process {pid} with SIGKILL.", "Force-kill PID {pid}.", "Terminate {pid} immediately.", "SIGKILL {pid}."],
     '{{"tool":"kill_process","args":{{"pid":{pid},"signal":9}}}}'),
    (["Send SIGTERM to process {pid}.", "Gracefully stop PID {pid}.", "Ask {pid} to exit.", "Soft-kill {pid}."],
     '{{"tool":"kill_process","args":{{"pid":{pid},"signal":15}}}}'),
    (["Get info on process {pid}.", "Inspect PID {pid}.", "What is process {pid} doing?", "Show {pid} details."],
     '{{"tool":"get_proc_info","args":{{"pid":{pid}}}}}'),
    (["Run: {cmd}", "Execute: {cmd}", "Shell command: {cmd}", "Run shell: {cmd}"],
     '{{"tool":"exec_shell","args":{{"cmd":"{cmd}"}}}}'),
    (["Write '{content}' to {path}.", "Save '{content}' into {path}.", "Create {path} with content '{content}'."],
     '{{"tool":"write_file","args":{{"path":"{path}","content":"{content}"}}}}'),
    (["Connect to {host}:{port}.", "Test {host} on port {port}.", "Open TCP connection to {host}:{port}."],
     '{{"tool":"net_connect","args":{{"host":"{host}","port":{port}}}}}'),
    (["Read the last {n} lines of syslog.", "Show {n} syslog lines.", "Get {n} entries from system log."],
     '{{"tool":"read_syslog","args":{{"lines":{n}}}}}'),
    (["Start {service}.", "Bring up {service}.", "Launch {service} service.", "systemctl start {service}."],
     '{{"tool":"service_ctl","args":{{"action":"start","service":"{service}"}}}}'),
    (["Stop {service}.", "Shut down {service}.", "systemctl stop {service}.", "Halt {service}."],
     '{{"tool":"service_ctl","args":{{"action":"stop","service":"{service}"}}}}'),
    (["Restart {service}.", "Reload {service}.", "Bounce {service}.", "systemctl restart {service}."],
     '{{"tool":"service_ctl","args":{{"action":"restart","service":"{service}"}}}}'),
    (["Check status of {service}.", "Is {service} running?", "systemctl status {service}.", "How is {service} doing?"],
     '{{"tool":"service_ctl","args":{{"action":"status","service":"{service}"}}}}'),
    (["Check disk at {path}.", "How full is {path}?", "Disk usage of {path}.", "df {path}."],
     '{{"tool":"get_disk_usage","args":{{"path":"{path}"}}}}'),
    (["Open files of PID {pid}.", "What has process {pid} open?", "List FDs for {pid}."],
     '{{"tool":"list_dir","args":{{"path":"/proc/{pid}/fd"}}}}'),
]

SHELL_CMDS = ["df -h","free -m","ps aux","vmstat 1 3","iostat -xz 1 1",
              f"lsof -p {rand_pid()}","dmesg | tail -30","uptime","who"]

def make_tool_single():
    variants, resp_tmpl = rng.choice(TOOL_TASKS)
    pid=rand_pid(); path=rand_path(); service=rng.choice(SERVICES)
    host=rand_ip(); port=rand_port(); n=rng.choice([50,100,200])
    content=rng.choice(["1","enabled","debug=true","max_conn=100"])
    cmd=rng.choice(SHELL_CMDS)
    kw=dict(pid=pid,path=path,service=service,host=host,port=port,n=n,content=content,cmd=cmd)
    try:
        return {"prompt":TOOLS_HEADER+"\n\n"+rng.choice(variants).format(**kw),
                "response":resp_tmpl.format(**kw)}
    except KeyError:
        return {"prompt":TOOLS_HEADER+"\n\n"+rng.choice(variants),"response":resp_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 8. TOOL CALLS — multi-step  (~8%)
# ══════════════════════════════════════════════════════════════════════════════

MULTI_TASKS = [
    ("PID {pid} is at 95% CPU. Investigate then terminate.",
     [("get_proc_info",'{{"pid":{pid}}}',"Inspect process {pid}."),
      ("kill_process",'{{"pid":{pid},"signal":9}}',"Kill process {pid} with SIGKILL.")]),
    ("Disk at /var/log nearly full. Check then clean old logs.",
     [("get_disk_usage",'{{"path":"/var/log"}}',"Check /var/log disk usage."),
      ("exec_shell",'{{"cmd":"find /var/log -name \'*.gz\' -mtime +30 -delete"}}',"Delete old compressed logs.")]),
    ("{service} failed. Check status then restart.",
     [("service_ctl",'{{"action":"status","service":"{service}"}}',"Check {service} status."),
      ("service_ctl",'{{"action":"restart","service":"{service}"}}',"Restart {service}.")]),
    ("Memory low. Find biggest process then soft-kill it.",
     [("exec_shell",'{{"cmd":"ps aux --sort=-%mem | head -5"}}',"List top memory consumers."),
      ("kill_process",'{{"pid":{pid},"signal":15}}',"Send SIGTERM to process {pid}.")]),
    ("PID {pid} in D state. Read wait channel then check dmesg.",
     [("read_file",'{{"path":"/proc/{pid}/wchan"}}',"Read what kernel fn {pid} is blocked on."),
      ("exec_shell",'{{"cmd":"dmesg | tail -30"}}',"Check recent kernel messages.")]),
    ("Config at {path} may have changed. Read it then restart {service}.",
     [("read_file",'{{"path":"{path}"}}',"Read current config."),
      ("service_ctl",'{{"action":"restart","service":"{service}"}}',"Restart {service}.")]),
    ("OOM killer fired. Read syslog, check memory, check disk.",
     [("read_syslog",'{{"lines":200}}',"Find OOM events in syslog."),
      ("exec_shell",'{{"cmd":"free -m"}}',"Check current memory."),
      ("get_disk_usage",'{{"path":"/"}}',"Check disk usage.")]),
]

def make_tool_multi():
    task = rng.choice(MULTI_TASKS)
    prompt_tmpl, steps = task
    pid=rand_pid(); service=rng.choice(SERVICES); path=rand_path()
    kw=dict(pid=pid,service=service,path=path)
    prompt = TOOLS_HEADER+"\n\n"+prompt_tmpl.format(**kw)
    lines=[]
    for i,(tool,args_t,reason_t) in enumerate(steps,1):
        lines.append(f"Step {i}: {reason_t.format(**kw)}")
        lines.append(f'Tool call: {{"tool":"{tool}","args":{args_t.format(**kw)}}}')
    return {"prompt":prompt,"response":"\n".join(lines)}


# ══════════════════════════════════════════════════════════════════════════════
# 9. SYSADMIN Q&A  (~8%)
# ══════════════════════════════════════════════════════════════════════════════

SYSADMIN_QA = [
    ("What does the OOM killer do?",
     "When the system runs out of memory and swap, the OOM killer selects and kills the process with the "
     "highest OOM score (based on memory use, runtime, priority) via SIGKILL. Protect critical processes "
     "with `echo -1000 > /proc/<pid>/oom_score_adj`."),
    ("What is a zombie process and how do you remove it?",
     "A zombie finished executing but its parent hasn't called wait(). It holds no resources beyond a "
     "process table entry. Fix: send SIGCHLD to the parent, or kill the parent so init adopts and reaps it. "
     "You cannot SIGKILL a zombie directly."),
    ("Difference between SIGTERM and SIGKILL?",
     "SIGTERM (15): requests graceful shutdown — process can catch it, flush buffers, clean up. "
     "SIGKILL (9): unconditional kernel-level kill, no cleanup possible. Always try SIGTERM first."),
    ("What is copy-on-write (COW) in fork()?",
     "fork() shares parent memory pages with the child marked read-only. On first write by either process, "
     "a page fault fires and the kernel copies only that page. Makes fork() fast — only modified pages "
     "are duplicated."),
    ("How does the Linux CFS scheduler work?",
     "CFS tracks each task's virtual runtime (vruntime) — CPU time received normalised by priority weight. "
     "Always runs the task with lowest vruntime (stored in a red-black tree). Nice values adjust weight; "
     "real-time tasks (SCHED_FIFO/RR) preempt all normal tasks."),
    ("What is inode exhaustion?",
     "'No space left on device' despite free disk space means inode exhaustion. Diagnose: `df -i`. "
     "Cause: millions of tiny files. Fix: delete them, or reformat with more inodes: `mkfs.ext4 -N <count>`."),
    ("What is cgroups and what does it control?",
     "Control groups limit and account for resource use by process groups. "
     "Resources: CPU (cpu.quota), memory (memory.limit_in_bytes), I/O (blkio.weight), network (tc). "
     "Docker and systemd use cgroups internally. v2 hierarchy at /sys/fs/cgroup/."),
    ("What is eBPF?",
     "eBPF is a kernel VM that runs sandboxed programs in kernel context without kernel modules. "
     "Used for: network filtering (XDP), performance tracing (bpftrace), security enforcement (seccomp-BPF). "
     "Programs are JIT-compiled and verified safe before loading. Tools: bcc, bpftrace, Cilium."),
    ("What is the difference between a hard link and a symbolic link?",
     "Hard link: directory entry pointing to the same inode — survives original deletion. Can't span "
     "filesystems or link directories. Symbolic link: file containing a path — breaks if target is deleted. "
     "Can cross filesystems and point to directories."),
    ("What is TCP TIME_WAIT and why does it exist?",
     "TIME_WAIT holds a closed TCP connection's state for 2×MSL (~60s) so delayed packets from the old "
     "connection don't corrupt a new one on the same port. High TIME_WAIT counts are normal under heavy "
     "connection churn; mitigate with SO_REUSEADDR."),
    ("What is huge pages and when should I use them?",
     "Huge pages (2MB vs 4KB normal) reduce TLB pressure for large working sets. Use for databases, JVMs, "
     "high-performance networking. Set via /proc/sys/vm/nr_hugepages. Disable transparent huge pages (THP) "
     "for latency-sensitive apps: `echo never > /sys/kernel/mm/transparent_hugepage/enabled`."),
    ("What does vm.swappiness control?",
     "swappiness (0–100) controls how aggressively the kernel swaps anonymous pages vs reclaiming file cache. "
     "0 = avoid swap, prefer reclaiming file cache. 100 = aggressively swap. Default is 60. "
     "Databases often set it to 1-10 to avoid unexpected swap latency."),
]

def make_sysadmin_qa():
    q,a = rng.choice(SYSADMIN_QA)
    return {"prompt":q,"response":a}


# ══════════════════════════════════════════════════════════════════════════════
# 10. LOG ANALYSIS  (~5%)
# ══════════════════════════════════════════════════════════════════════════════

LOG_SCENARIOS = [
    (lambda kw: (
        f"Jun 12 03:14:2{i} host sshd[{kw['pid']}]: Failed password for root from {kw['ip']} port {kw['port']+i} ssh2\n"
        for i in range(4)),
     "SSH brute-force from {ip}: 4 failures in 3s targeting root. "
     "Action: block {ip} with iptables, install fail2ban, set PermitRootLogin no, disable PasswordAuthentication."),
    (lambda kw: [
        f"kernel: EXT4-fs error (device sda1): bad block bitmap checksum\n",
        f"kernel: EXT4-fs (sda1): delayed block allocation failed with error -5\n",
        f"kernel: EXT4-fs (sda1): This should not happen!! Data will be lost\n"],
     "Critical EXT4 corruption on sda1, error -5=EIO. Actions: (1) backup NOW, "
     "(2) remount ro: `mount -o remount,ro /dev/sda1`, (3) smartctl -a /dev/sda, "
     "(4) fsck from live environment, (5) replace if SMART shows bad sectors."),
    (lambda kw: [f"kernel: {kw['proc']}[{kw['pid']}]: segfault at {kw['addr']} ip {kw['rip']} sp {kw['rsp']} error 4 in libc.so.6\n"],
     "Process {proc} ({pid}) crashed: read from unmapped address {addr} inside libc — "
     "NULL/dangling pointer passed to a libc function. Debug: gdb or valgrind, check NULL guards before libc calls."),
    (lambda kw: [f"kernel: possible SYN flooding on port {kw['port']}. Sending cookies.\n"]*3,
     "SYN flood on port {port}. Kernel auto-enabled SYN cookies. "
     "Also: `sysctl -w net.ipv4.tcp_max_syn_backlog=4096`, rate-limit SYNs with iptables."),
    (lambda kw: [f"nginx: [error] connect() failed (111: Connection refused) while connecting to upstream, "
                 f"client: {kw['ip']}, upstream: \"http://127.0.0.1:{kw['port']}/\"\n"],
     "Nginx can't reach upstream at 127.0.0.1:{port} — nothing is listening on that port. "
     "Fix: check `ss -tlnp | grep {port}`, start the backend, verify nginx upstream config."),
]

def make_log_analysis():
    scenario = rng.choice(LOG_SCENARIOS)
    log_gen, resp_tmpl = scenario
    kw=dict(pid=rand_pid(),ip=rand_ip(),port=rand_port(),proc=rng.choice(PROC_NAMES),
            addr=rand_addr(),rip=rand_addr(),rsp=rand_addr())
    raw = log_gen(kw)
    log = "".join(raw) if not isinstance(raw,str) else raw
    try:
        response = resp_tmpl.format(**kw)
    except KeyError:
        response = resp_tmpl
    return {"prompt":f"Analyze this log:\n\n{log}\nWhat is happening and what should be done?",
            "response":response}


# ══════════════════════════════════════════════════════════════════════════════
# 11. PROCESS DEBUGGING  (~5%)
# ══════════════════════════════════════════════════════════════════════════════

PROC_SCENARIOS = [
    ("Process {pid} ({name}) has been in D state for {mins} minutes.",
     "D state = blocked on kernel I/O (NFS hang, failing disk, deadlocked driver). "
     "Check: `cat /proc/{pid}/wchan` for the blocking function. `dmesg | tail -20` for I/O errors. "
     "Cannot SIGKILL a D-state process — fix the underlying I/O."),
    ("OOM killer log: Kill process {pid} ({name}) score {score}.",
     "OOM killed {name} ({pid}) — score {score} made it highest-priority target. "
     "Actions: `free -m` for current state, `ps aux --sort=-%mem | head` for hogs, "
     "add RAM/swap, set oom_score_adj=-1000 on critical processes."),
    ("Process {pid} shows {virt}GB VIRT but only {rss}MB RSS.",
     "Normal. VIRT = total reserved address space (mmaps, shared libs, unrealised heap). "
     "RSS = physical RAM actually used. Gap is lazy-allocated / COW pages not yet faulted in. "
     "Watch RSS growth over time for actual leaks; VIRT alone is not a problem."),
    ("strace shows process {pid} making thousands of stat() calls per second on the same path.",
     "The process is busy-polling a path with stat() — a hot spin loop. Wastes CPU. "
     "Fix: switch to inotify (inotify_add_watch) for event-driven file change notifications "
     "instead of polling."),
    ("ps shows PID {pid} in Z (zombie) state, parent {ppid}.",
     "Zombie: {pid} finished but parent {ppid} hasn't called wait(). "
     "Send SIGCHLD to {ppid}: `kill -CHLD {ppid}`. If that fails, kill {ppid} "
     "so init adopts and reaps the zombie. Cannot SIGKILL the zombie itself."),
]

def make_proc_debug():
    tmpl_q,tmpl_a = rng.choice(PROC_SCENARIOS)
    kw=dict(pid=rand_pid(),ppid=rand_pid(),name=rng.choice(PROC_NAMES),
            mins=rng.randint(2,60),score=rng.randint(100,999),
            virt=rng.randint(2,50),rss=rng.randint(50,500))
    return {"prompt":"Diagnose: "+tmpl_q.format(**kw),"response":tmpl_a.format(**kw)}


# ══════════════════════════════════════════════════════════════════════════════
# 12. SECURITY  (~4%)
# ══════════════════════════════════════════════════════════════════════════════

SECURITY_QA = [
    ("Non-root process {pid} ({name}) read /etc/shadow. How serious?",
     "Critical. /etc/shadow holds password hashes. Indicates privilege escalation or SUID misconfiguration. "
     "Kill {pid}, check for SUID: `stat $(readlink -f /proc/{pid}/exe)`, restore permissions: "
     "`chmod 640 /etc/shadow && chown root:shadow /etc/shadow`, rotate all passwords."),
    ("Unknown process on port 4444, PID {pid}.",
     "Port 4444 is common for Metasploit reverse shells. Identify: `ls -la /proc/{pid}/exe`, "
     "check connections: `ss -tnp | grep {pid}`, kill: `kill -9 {pid}`, block: "
     "`iptables -A INPUT --dport 4444 -j DROP`, audit for persistence (crontabs, systemd, ~/.bashrc)."),
    ("A user ran chmod 777 /etc/passwd. What now?",
     "Critical. Any user can now add root-equivalent accounts. Fix immediately: "
     "`chmod 644 /etc/passwd && chown root:root /etc/passwd`. Check for modifications: "
     "`diff /etc/passwd /etc/passwd-`. Audit: `journalctl _COMM=chmod`."),
    ("dmesg: apparmor=DENIED operation=exec target=/bin/sh pid={pid}",
     "AppArmor blocked shell exec from {pid} — could be shell injection attack blocked, or false positive. "
     "Identify: `ls -la /proc/{pid}/exe`. If attack: audit app for command injection. "
     "If false positive: `aa-logprof` to update profile. Never blindly disable enforcement."),
    ("Find all world-writable files outside /tmp.",
     "`find / -not -path '/tmp/*' -not -path '/proc/*' -perm -0002 -type f 2>/dev/null`\n"
     "World-writable files outside /tmp are a security risk. For each: verify intent, remove write bit with `chmod o-w`."),
]

def make_security():
    q_tmpl,a_tmpl = rng.choice(SECURITY_QA)
    kw=dict(pid=rand_pid(),name=rng.choice(PROC_NAMES))
    try:
        return {"prompt":q_tmpl.format(**kw),"response":a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt":q_tmpl,"response":a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 13. NETWORK  (~4%)
# ══════════════════════════════════════════════════════════════════════════════

NETWORK_QA = [
    ("Connection to {host}:{port} is timing out. How do I diagnose?",
     "1. `ping {host}` — is host reachable? 2. `nc -zv -w3 {host} {port}` — is port open? "
     "3. `traceroute {host}` — where does path fail? 4. `ss -tnp | grep {port}` — local firewall? "
     "5. On server: `ss -tlnp | grep {port}` — is service listening?"),
    ("How do I capture packets on {iface} for port {port}?",
     "`tcpdump -i {iface} -n 'port {port}' -w /tmp/cap.pcap`\n"
     "Then analyse with Wireshark or `tcpdump -r /tmp/cap.pcap`."),
    ("What is SNAT vs DNAT?",
     "SNAT (Source NAT): rewrites source IP of outgoing packets — used for internet masquerading. "
     "Applied in POSTROUTING chain. "
     "DNAT (Destination NAT): rewrites destination IP of incoming packets — used for port forwarding. "
     "Applied in PREROUTING chain."),
    ("How do I add a static route to {host} via {gw}?",
     "Temporary: `ip route add {host}/32 via {gw}`\n"
     "Verify: `ip route show | grep {host}`\n"
     "Persistent: add to /etc/netplan or /etc/network/interfaces depending on distro."),
    ("Why do I have many CLOSE_WAIT connections?",
     "CLOSE_WAIT means the remote side sent FIN but the local application hasn't called close(). "
     "Many CLOSE_WAIT = the app is not closing sockets properly — file descriptor leak. "
     "Profile the application for missing close()/shutdown() calls."),
]

def make_network():
    q_tmpl,a_tmpl = rng.choice(NETWORK_QA)
    kw=dict(host=rand_ip(),port=rand_port(),iface=rng.choice(IFACES),gw=rand_ip())
    try:
        return {"prompt":q_tmpl.format(**kw),"response":a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt":q_tmpl,"response":a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 14. SYSTEMD  (~3%)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEMD_QA = [
    ("How do I create a systemd service for {name}?",
     f"Create /etc/systemd/system/{{name}}.service:\n\n"
     "[Unit]\nDescription={name}\nAfter=network.target\n\n"
     "[Service]\nExecStart=/usr/bin/{name}\nRestart=always\nRestartSec=5\n\n"
     "[Install]\nWantedBy=multi-user.target\n\n"
     "Then: `systemctl daemon-reload && systemctl enable --now {name}`"),
    ("{service} keeps failing with exit code 1. How do I debug?",
     "1. `journalctl -u {service} -n 100` — read error output. "
     "2. `systemctl status {service}` — see last lines and exit code. "
     "3. Run ExecStart manually as the service user to see stderr. "
     "4. Check config file syntax and that dependencies are available."),
    ("How do I run a command daily with systemd instead of cron?",
     "Create two files:\n\n/etc/systemd/system/mytask.service (Type=oneshot, ExecStart=...)\n"
     "/etc/systemd/system/mytask.timer:\n[Timer]\nOnCalendar=daily\nPersistent=true\n"
     "[Install]\nWantedBy=timers.target\n\nThen: `systemctl enable --now mytask.timer`"),
    ("How do I see all logs for {service} since last reboot?",
     "`journalctl -u {service} -b`\n"
     "Add `-f` to follow, `-p err` for errors only, `--since '1 hour ago'` to time-limit."),
]

def make_systemd():
    q_tmpl,a_tmpl = rng.choice(SYSTEMD_QA)
    kw=dict(service=rng.choice(SERVICES),name=rng.choice(PROC_NAMES))
    try:
        return {"prompt":q_tmpl.format(**kw),"response":a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt":q_tmpl,"response":a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# 15. DOCKER  (~2%)
# ══════════════════════════════════════════════════════════════════════════════

DOCKER_QA = [
    ("Container {name} exited with code 137. Why?",
     "Exit 137 = 128+9 = SIGKILL. Either OOM-killed (check `docker inspect {name} --format={{{{.State.OOMKilled}}}}`) "
     "or manually killed. Fix: increase memory limit `--memory=2g`, or check app for leaks."),
    ("How do I exec a shell in container {name}?",
     "`docker exec -it {name} /bin/bash`\nIf bash is absent: `docker exec -it {name} /bin/sh`"),
    ("How do I limit container {name} to 1 CPU and 512MB memory?",
     "`docker run --cpus=1.0 --memory=512m <image>`\nVerify: `docker stats {name}`"),
    ("How do I see what a container is doing without exec-ing into it?",
     "`docker logs --tail 100 -f {name}` — live log output.\n"
     "`docker top {name}` — running processes.\n"
     "`docker stats {name}` — CPU/memory/network usage."),
    ("What is the difference between COPY and ADD in a Dockerfile?",
     "COPY: copies files from build context — simple and predictable. Always prefer COPY.\n"
     "ADD: same as COPY plus auto-extracts tar archives and can fetch URLs. "
     "Avoid ADD unless you specifically need tar extraction."),
]

CONTAINER_NAMES=["web","api","db","worker","nginx","redis","app","proxy"]

def make_docker():
    q_tmpl,a_tmpl = rng.choice(DOCKER_QA)
    kw=dict(name=rng.choice(CONTAINER_NAMES))
    try:
        return {"prompt":q_tmpl.format(**kw),"response":a_tmpl.format(**kw)}
    except KeyError:
        return {"prompt":q_tmpl,"response":a_tmpl}


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

GENERATORS = [
    (make_python_coding, 18),
    (make_bash_cmd,      10),
    (make_code_debug,     8),
    (make_tool_single,   16),
    (make_tool_multi,     8),
    (make_kernel_fault,  10),
    (make_sysadmin_qa,    8),
    (make_log_analysis,   5),
    (make_proc_debug,     5),
    (make_c_coding,       5),
    (make_code_explain,   4),
    (make_security,       4),
    (make_network,        4),
    (make_systemd,        3),
    (make_docker,         2),
]
_gens, _weights = zip(*GENERATORS)


def generate_dataset(n, seed=42):
    local = random.Random(seed)
    out = []
    for _ in range(n):
        gen = local.choices(_gens, weights=_weights, k=1)[0]
        try:
            out.append(gen())
        except Exception:
            out.append(make_python_coding())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  type=int, default=80000)
    parser.add_argument("--val",    type=int, default=8000)
    parser.add_argument("--outdir", default="training/data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for split, n, seed, fname in [
        ("train", args.train, 42,  "v4_train.jsonl"),
        ("val",   args.val,   99,  "v4_val.jsonl"),
    ]:
        print(f"Generating {n} {split} samples...")
        data = generate_dataset(n, seed)
        path = os.path.join(args.outdir, fname)
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        mb = os.path.getsize(path) // 1024 // 1024
        print(f"  → {path}  ({mb} MB)")

    print("\nCategory weights:")
    total = sum(_weights)
    for g, w in GENERATORS:
        est = int(args.train * w / total)
        print(f"  {g.__name__:<22} {100*w/total:5.1f}%  (~{est:,})")
    print(f"\nTotal: {args.train:,} train / {args.val:,} val")

if __name__ == "__main__":
    main()
