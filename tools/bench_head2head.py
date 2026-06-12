#!/usr/bin/env python3
"""
Head-to-head benchmark: qwen3:0.6b vs MominoMoE-v2
Both models run in parallel via Ollama API. Results printed side by side.
"""
import requests, threading, time, textwrap, sys

OLLAMA = "http://localhost:11434"

MODEL_A = "qwen3:0.6b"
MODEL_B = "hf.co/Momin-Aldahdouh/MominoMoE-v2:Q4_K_M"

TOOLS_SCHEMA = """Available tools (call as JSON on a single line):
{"tool": "read_file", "args": {"path": "<str>"}}
{"tool": "exec_shell", "args": {"cmd": "<str>"}}
{"tool": "kill_process", "args": {"pid": <int>, "signal": <int>}}
{"tool": "list_dir", "args": {"path": "<str>"}}
{"tool": "get_proc_info", "args": {"pid": <int>}}
{"tool": "write_file", "args": {"path": "<str>", "content": "<str>"}}
Respond with ONLY a JSON tool call, nothing else."""

PROMPTS = [
    {
        "label": "Kernel fault diagnosis",
        "text": (
            "[FAULT] vector=14 (Page Fault) err=0x0006 rip=0x0000000000401234 "
            "cr2=0x0000000000000008 tid=3 cwd=/bin\n\n"
            "[REGISTERS] rax=0x0 rdi=0x8 rsi=0x100 rsp=0x7fff00100ff8\n\n"
            "[RECENT_SYSCALLS]\n  SYS_OPEN /bin/sh 0 -> 3\n  SYS_READ 3 4096 -> 4096\n\n"
            "[LOG]\n  [VFS] opened /bin/sh\n  [SCHED] thread 3 running\n\n"
            "[QUERY] Diagnose this fault and suggest a corrective action."
        ),
    },
    {
        "label": "Tool call — read a config file",
        "text": f"{TOOLS_SCHEMA}\n\nRead the file /etc/os-release.",
    },
    {
        "label": "Tool call — kill a hung process",
        "text": f"{TOOLS_SCHEMA}\n\nProcess 1847 is consuming 100% CPU and is unresponsive. Terminate it with SIGKILL.",
    },
    {
        "label": "Tool call — inspect process then exec shell",
        "text": f"{TOOLS_SCHEMA}\n\nCheck the state of process 42, then list the contents of /proc/42/fd.",
    },
    {
        "label": "Shell command generation",
        "text": "Write a single bash one-liner that finds all files larger than 100MB under /var/log and deletes them.",
    },
    {
        "label": "General knowledge",
        "text": "Explain what a kernel page fault is in one paragraph.",
    },
]

W = 52  # column width per model


SYSTEM_PROMPT = (
    "You are a kernel fault diagnostician for MominOS, an x86-64 OS. "
    "For kernel fault prompts, diagnose the fault type, root cause, and corrective action. "
    "For other questions, answer concisely. /no_think"
)


def query(model, prompt, result, idx):
    """Stream from Ollama chat endpoint, collect tokens, measure timing."""
    import json
    url = f"{OLLAMA}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "stream": True,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 250},
    }
    first_token_time = None
    tokens = 0
    text = ""
    t0 = time.perf_counter()
    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                    text += tok
                    tokens += 1
                if chunk.get("done"):
                    break
    except Exception as e:
        text = f"[ERROR: {e}]"
    elapsed = time.perf_counter() - t0
    result[idx] = {
        "text": text,
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft": first_token_time or elapsed,
        "tps": tokens / elapsed if elapsed > 0 else 0,
    }


def wrap(text, width):
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width) or [""])
    return lines


def print_side_by_side(label_a, text_a, label_b, text_b):
    lines_a = wrap(text_a, W)
    lines_b = wrap(text_b, W)
    n = max(len(lines_a), len(lines_b))
    lines_a += [""] * (n - len(lines_a))
    lines_b += [""] * (n - len(lines_b))
    print(f"  {'─'*W}   {'─'*W}")
    print(f"  {label_a:<{W}}   {label_b:<{W}}")
    print(f"  {'─'*W}   {'─'*W}")
    for a, b in zip(lines_a, lines_b):
        print(f"  {a:<{W}}   {b:<{W}}")
    print(f"  {'─'*W}   {'─'*W}")


def run_round(prompt_label, prompt_text):
    print(f"\n{'='*((W*2)+6)}")
    print(f"  PROMPT: {prompt_label}")
    print(f"{'='*((W*2)+6)}")
    print(f"  {prompt_text[:120]}{'...' if len(prompt_text)>120 else ''}\n")

    results = [None, None]
    t_a = threading.Thread(target=query, args=(MODEL_A, prompt_text, results, 0))
    t_b = threading.Thread(target=query, args=(MODEL_B, prompt_text, results, 1))

    print("  [Running both models in parallel...]")
    wall_start = time.perf_counter()
    t_a.start(); t_b.start()
    t_a.join();  t_b.join()
    wall = time.perf_counter() - wall_start
    print(f"  Wall time: {wall:.1f}s\n")

    a, b = results[0], results[1]

    short_a = MODEL_A.split("/")[-1]
    short_b = "MominoMoE-v2"

    print_side_by_side(
        f"{short_a}  ({a['tps']:.1f} tok/s, ttft={a['ttft']:.2f}s)",
        a["text"].strip(),
        f"{short_b}  ({b['tps']:.1f} tok/s, ttft={b['ttft']:.2f}s)",
        b["text"].strip(),
    )

    print(f"\n  STATS")
    print(f"  {'Metric':<22} {'qwen3:0.6b':>20}   {'MominoMoE-v2':>20}")
    print(f"  {'-'*22} {'-'*20}   {'-'*20}")
    print(f"  {'Time to first token':<22} {a['ttft']:>19.2f}s   {b['ttft']:>19.2f}s")
    print(f"  {'Total time':<22} {a['elapsed']:>19.1f}s   {b['elapsed']:>19.1f}s")
    print(f"  {'Tokens generated':<22} {a['tokens']:>20}   {b['tokens']:>20}")
    print(f"  {'Throughput (tok/s)':<22} {a['tps']:>20.1f}   {b['tps']:>20.1f}")


def main():
    print(f"\n{'='*((W*2)+6)}")
    print(f"  HEAD-TO-HEAD BENCHMARK")
    print(f"  {MODEL_A}  vs  MominoMoE-v2")
    print(f"{'='*((W*2)+6)}")

    # Verify both models are available
    for m in [MODEL_A, MODEL_B]:
        try:
            r = requests.post(f"{OLLAMA}/api/generate",
                json={"model": m, "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
                timeout=30)
            r.raise_for_status()
            print(f"  OK: {m}")
        except Exception as e:
            print(f"  FAIL: {m} — {e}")
            sys.exit(1)

    for p in PROMPTS:
        run_round(p["label"], p["text"])

    print(f"\n{'='*((W*2)+6)}")
    print("  Done.")
    print(f"{'='*((W*2)+6)}\n")


if __name__ == "__main__":
    main()
