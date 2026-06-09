#!/usr/bin/env python3
"""
Phase 1: Dataset Download & Conversion for MominoMoE OS Copilot Agent Training.

Downloads:
  - Agent-Ark/Toucan-1.5M (SFT config): 119,287 tool-agent trajectories
  - Solaris99/AgentBank configs: intercode_bash, alfworld, apps, hotpotqa,
    humaneval, webarena, webshop

Converts each to MominoMoE SFT JSONL format: {"prompt": "...", "response": "..."}
Splits data 90/5/5 train/val/test.
Saves to: agent_train.jsonl, agent_val.jsonl, agent_test.jsonl
"""

import json
import os
import sys
import random
from typing import List, Dict, Optional
from collections import defaultdict

from datasets import load_dataset

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR  # Save alongside the script

# ── Dataset configs ──────────────────────────────────────────────────────

DATASETS = [
    # (hf_dataset, hf_config, split, expected_samples)
    ("Agent-Ark/Toucan-1.5M", "SFT", "train", 119287),
    ("Solaris99/AgentBank", "intercode_bash", "train", 200),
    ("Solaris99/AgentBank", "alfworld", "train", 3321),
    ("Solaris99/AgentBank", "apps", "train", 4408),
    ("Solaris99/AgentBank", "hotpotqa", "train", 4273),
    ("Solaris99/AgentBank", "humaneval", "train", 132),
    ("Solaris99/AgentBank", "webarena", "train", 658),
    ("Solaris99/AgentBank", "webshop", "train", 4958),
]

OUTPUT_FILES = {
    "train": os.path.join(OUTPUT_DIR, "agent_train.jsonl"),
    "val": os.path.join(OUTPUT_DIR, "agent_val.jsonl"),
    "test": os.path.join(OUTPUT_DIR, "agent_test.jsonl"),
}


# ── Conversion Functions ─────────────────────────────────────────────────

def convert_toucan_messages(messages_str: str, sample_id: str = "") -> Optional[Dict]:
    """Convert Toucan's JSON messages string to prompt+response format.
    
    Toucan messages have roles: user, assistant, tool_call, tool_response.
    We reconstruct the full conversation history as prompt (everything except
    the final assistant message), and the final assistant message as response.
    
    For SFT format:
      - prompt = user's first message + all assistant/tool_call/tool_response
        history (excluding the last assistant message)
      - response = the last assistant message
    """
    try:
        messages = json.loads(messages_str)
    except json.JSONDecodeError:
        return None
    
    if not isinstance(messages, list) or len(messages) == 0:
        return None
    
    # Find the last assistant message index
    last_assistant_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    
    if last_assistant_idx < 0:
        # No assistant message — skip
        return None
    
    # Build conversation history (all messages up to and including last assistant)
    # But we separate: prompt = everything before the last assistant message,
    # formatted as conversation; response = the last assistant content
    
    # Build a nicely formatted conversation history
    history_parts = []
    for i in range(last_assistant_idx):
        msg = messages[i]
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")
        
        if role == "user":
            history_parts.append(f"<|user|> {content}")
        elif role == "assistant":
            history_parts.append(f"<|assistant|> {content}")
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn_name = tc.get("function", {}).get("name", "unknown")
                        fn_args = tc.get("function", {}).get("arguments", {})
                        if isinstance(fn_args, str):
                            try:
                                fn_args = json.loads(fn_args)
                            except json.JSONDecodeError:
                                pass
                        history_parts.append(
                            f"<|tool_call|> {fn_name}({json.dumps(fn_args)})"
                        )
        elif role == "tool_call":
            name = msg.get("name", "unknown")
            args = msg.get("args", msg.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            history_parts.append(
                f"<|tool_call|> {name}({json.dumps(args)})"
            )
        elif role == "tool_response" or role == "tool":
            tool_call_id_info = f" (call: {tool_call_id})" if tool_call_id else ""
            history_parts.append(f"<|tool_response|{tool_call_id_info}> {content}")
        else:
            history_parts.append(f"<|{role}|> {content}")
    
    # The last assistant message is the response
    final_assistant = messages[last_assistant_idx]
    response_content = final_assistant.get("content", "")
    
    # If response empty but has tool_calls, use tool_calls as response
    if not response_content.strip():
        tool_calls = final_assistant.get("tool_calls", [])
        if tool_calls:
            tc_parts = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn_name = tc.get("function", {}).get("name", "unknown")
                    fn_args = tc.get("function", {}).get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except json.JSONDecodeError:
                            pass
                    tc_parts.append(
                        f"<|tool_call|> {fn_name}({json.dumps(fn_args)})"
                    )
            response_content = "\n".join(tc_parts)
    
    prompt = "\n".join(history_parts)
    
    if not prompt.strip() or not response_content.strip():
        return None
    
    return {
        "prompt": prompt,
        "response": response_content,
        "source": "Toucan-1.5M_SFT",
        "uuid": sample_id,
    }


def convert_agentbank_conversation(sample: Dict, source_name: str) -> Optional[Dict]:
    """Convert an AgentBank sample to prompt+response format.
    
    AgentBank has various formats. We try to handle them flexibly.
    """
    prompt = None
    response = None
    
    # Try 'conversations' field (most common AgentBank format)
    conversations = sample.get("conversations")
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            conversations = None
    
    if isinstance(conversations, list) and len(conversations) > 0:
        # Find last assistant turn
        last_asst = None
        history = []
        for msg in conversations:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", ""))
                content = msg.get("content", msg.get("value", ""))
                if role in ("assistant", "gpt", "bot", "agent"):
                    last_asst = content
                    # Don't add to history yet
                else:
                    if role in ("user", "human", "system"):
                        history.append(f"<|user|> {content}")
                    elif role == "tool":
                        history.append(f"<|tool_response|> {content}")
                    elif role == "tool_call":
                        history.append(f"<|tool_call|> {content}")
                    else:
                        history.append(f"<|{role}|> {content}")
        
        if last_asst:
            prompt = "\n".join(history)
            response = last_asst
    
    # Try 'input' / 'output' fields
    if prompt is None:
        inp = sample.get("input", sample.get("instruction", ""))
        out = sample.get("output", sample.get("response", ""))
        if inp and out:
            prompt = f"<|user|> {inp}"
            response = out
    
    # Try 'query' / 'answer' fields  
    if prompt is None:
        q = sample.get("query", sample.get("question", ""))
        a = sample.get("answer", sample.get("completion", ""))
        if q and a:
            prompt = f"<|user|> {q}"
            response = a
    
    # Try 'prompt' / 'completion' fields
    if prompt is None:
        p = sample.get("prompt", "")
        c = sample.get("completion", sample.get("chosen", ""))
        if p and c:
            prompt = f"<|user|> {p}"
            response = c
    
    # For bash-specific, try 'nl' / 'bash' fields
    if prompt is None:
        nl = sample.get("nl", sample.get("natural_language", sample.get("instruction", "")))
        bash = sample.get("bash", sample.get("command", sample.get("code", "")))
        if nl and bash:
            prompt = f"<|user|> Convert this to a bash command: {nl}"
            response = bash
    
    if prompt and response:
        return {
            "prompt": prompt.strip(),
            "response": response.strip(),
            "source": source_name,
        }
    
    return None


# ── Main Pipeline ────────────────────────────────────────────────────────

def process_datasets() -> List[Dict]:
    """Process all datasets and return list of converted samples."""
    all_samples = []
    source_counts = defaultdict(int)
    
    for hf_name, config, split, expected in DATASETS:
        print(f"\n{'='*60}")
        print(f"Loading: {hf_name} / {config} (expected ~{expected} samples)")
        print(f"{'='*60}")
        
        try:
            ds = load_dataset(hf_name, config, split=split, streaming=True)
        except Exception as e:
            print(f"  ERROR loading dataset: {e}")
            continue
        
        count = 0
        errors = 0
        
        for i, sample in enumerate(ds):
            # For Toucan, messages is a JSON string
            if hf_name == "Agent-Ark/Toucan-1.5M":
                messages = sample.get("messages", "")
                result = convert_toucan_messages(messages, sample_id=sample.get("uuid", ""))
            else:
                # AgentBank
                source_name = f"AgentBank/{config}"
                result = convert_agentbank_conversation(sample, source_name)
            
            if result:
                all_samples.append(result)
                source_counts[result["source"]] += 1
                count += 1
            else:
                errors += 1
            
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i+1} samples... ({count} converted, {errors} errors)")
        
        print(f"  Done: {count} converted, {errors} skipped out of {i+1} total")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_samples)} samples converted")
    print(f"{'='*60}")
    for source, cnt in sorted(source_counts.items()):
        print(f"  {source}: {cnt}")
    
    return all_samples


def split_and_save(samples: List[Dict]):
    """Split 90/5/5 and save to JSONL files."""
    print(f"\n{'='*60}")
    print(f"Splitting {len(samples)} samples into train/val/test (90/5/5)")
    print(f"{'='*60}")
    
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    
    splits = {
        "train": train,
        "val": val,
        "test": test,
    }
    
    for split_name, data in splits.items():
        out_path = OUTPUT_FILES[split_name]
        with open(out_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  {split_name}: {len(data)} samples -> {out_path}")
    
    # Print source distribution for each split
    for split_name, data in splits.items():
        source_dist = defaultdict(int)
        for item in data:
            source_dist.get(item.get("source", "unknown"), 0)
            source_dist[item.get("source", "unknown")] += 1
        print(f"\n  {split_name} source distribution:")
        for src, cnt in sorted(source_dist.items()):
            print(f"    {src}: {cnt}")


def verify_outputs():
    """Verify the output JSONL files."""
    print(f"\n{'='*60}")
    print("Verifying output files...")
    print(f"{'='*60}")
    
    for split_name, out_path in OUTPUT_FILES.items():
        if not os.path.exists(out_path):
            print(f"  MISSING: {out_path}")
            continue
        
        with open(out_path) as f:
            lines = f.readlines()
        
        valid = 0
        has_prompt = 0
        has_response = 0
        empty_prompt = 0
        empty_response = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                valid += 1
                if "prompt" in obj:
                    has_prompt += 1
                    if not obj["prompt"].strip():
                        empty_prompt += 1
                if "response" in obj:
                    has_response += 1
                    if not obj["response"].strip():
                        empty_response += 1
            except json.JSONDecodeError:
                pass
        
        print(f"\n  {split_name}: {out_path}")
        print(f"    Total lines: {len(lines)}")
        print(f"    Valid JSON: {valid}")
        print(f"    Has 'prompt' key: {has_prompt}")
        print(f"    Has 'response' key: {has_response}")
        print(f"    Empty prompts: {empty_prompt}")
        print(f"    Empty responses: {empty_response}")
        
        if valid > 0:
            print(f"    Sample entry:")
            obj = json.loads(lines[0].strip())
            print(f"      prompt (first 100 chars): {obj.get('prompt', '')[:100]}")
            print(f"      response (first 100 chars): {obj.get('response', '')[:100]}")


def main():
    print("=" * 60)
    print("MominoMoE OS Copilot Agent — Dataset Preparation")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all datasets
    samples = process_datasets()
    
    if len(samples) == 0:
        print("ERROR: No samples converted!")
        sys.exit(1)
    
    # Split and save
    split_and_save(samples)
    
    # Verify
    verify_outputs()
    
    print(f"\n{'='*60}")
    print("Done! Summary:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Train: {OUTPUT_FILES['train']}")
    print(f"  Val:   {OUTPUT_FILES['val']}")
    print(f"  Test:  {OUTPUT_FILES['test']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()