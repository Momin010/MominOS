#!/usr/bin/env python3
"""Dataset classes for MominoMoE training: Pretrain, SFT, and DPO.

Each dataset outputs tokenized tensors ready for model.forward().
All datasets use causal masking conventions:
  - PretrainDataset: causal mask for all positions
  - SFTDataset: response-only mask (ignore prompt tokens in loss)
  - DPODataset: pairwise (chosen, rejected) with masks
"""

import json
import os
import random
from typing import List, Dict, Optional, Callable, Union, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


# ── Special Token Helpers (can be overridden) ────────────────────────────

@dataclass
class ChatTokens:
    """Container for chat template special tokens."""
    bos: int = 1
    eos: int = 2
    pad: int = 0
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    system: str = "<|system|>"
    end_turn: str = "<|end|>"


DEFAULT_CHAT_TOKENS = ChatTokens()


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file, skipping empty lines."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create standard causal attention mask (lower triangular)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.view(1, 1, seq_len, seq_len)


# ── PretrainDataset ──────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """Dataset for causal language model pretraining.
    
    Loads tokenized text chunks and creates causal LM inputs/targets.
    Each sample: input_ids, attention_mask, labels (shifted for CLM loss).
    """
    
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 2048,
        tokenizer_vocab_size: int = 32000,
        shuffle_chunks: bool = True,
        seed: int = 42,
        memmap: bool = False,
    ):
        """
        Args:
            data_path: Path to tokenized .bin or .npy file, or directory of files
            max_seq_len: Maximum sequence length for each sample
            tokenizer_vocab_size: Vocab size for validation
            shuffle_chunks: Whether to shuffle chunk order
            seed: Random seed for reproducibility
            memmap: Use memory mapping for large datasets
        """
        self.max_seq_len = max_seq_len
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.rng = random.Random(seed)
        
        # Load tokens
        if os.path.isdir(data_path):
            self._load_from_dir(data_path, memmap)
        else:
            self._load_from_file(data_path, memmap)
        
        # Create chunks
        self._create_chunks()
        if shuffle_chunks:
            indices = list(range(len(self.chunks)))
            self.rng.shuffle(indices)
            self.chunks = [self.chunks[i] for i in indices]
    
    def _load_from_file(self, path: str, memmap: bool):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npy":
            self.tokens = torch.from_numpy(
                __import__("numpy").load(path, mmap_mode="r" if memmap else None)
            ).long()
        elif ext == ".bin":
            arr = __import__("numpy").fromfile(path, dtype="uint16")
            self.tokens = torch.from_numpy(arr).long()
        else:
            # Assume text file — tokenize on the fly
            raise NotImplementedError(
                f"PretrainDataset does not support {ext} files directly. "
                "Please provide pre-tokenized .npy or .bin files."
            )
        print(f"Loaded {len(self.tokens)} tokens from {path}")
    
    def _load_from_dir(self, dir_path: str, memmap: bool):
        """Load all token files from a directory."""
        all_tokens = []
        for fname in sorted(os.listdir(dir_path)):
            fpath = os.path.join(dir_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".npy", ".bin"):
                self._load_from_file(fpath, memmap)
                all_tokens.append(self.tokens)
        if all_tokens:
            self.tokens = torch.cat(all_tokens)
        else:
            raise ValueError(f"No token files found in {dir_path}")
    
    def _create_chunks(self):
        """Split tokens into max_seq_len chunks."""
        total = len(self.tokens)
        n_chunks = total // self.max_seq_len
        self.chunks = []
        for i in range(n_chunks):
            chunk = self.tokens[i * self.max_seq_len : (i + 1) * self.max_seq_len]
            self.chunks.append(chunk)
        # Drop remainder (too short)
        print(f"Created {len(self.chunks)} chunks of length {self.max_seq_len}")
    
    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.chunks[idx]
        input_ids = tokens.clone()
        labels = tokens.clone()
        attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ── SFTDataset ───────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning with response-only loss masking.
    
    Expects JSONL format: {"prompt": "...", "response": "..."}
    or pre-tokenized: {"input_ids": [...], "labels": [...]}.
    
    For text-format, labels for prompt tokens are set to -100 (ignored in loss).
    Only response tokens contribute to the loss.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_encode_fn: Callable[[str], List[int]],
        max_seq_len: int = 2048,
        chat_tokens: ChatTokens = DEFAULT_CHAT_TOKENS,
        response_only_loss: bool = True,
    ):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer_encode_fn: Function to encode text to token IDs
            max_seq_len: Maximum sequence length
            chat_tokens: Chat token configuration
            response_only_loss: Mask prompt tokens with -100
        """
        self.data_path = data_path
        self.encode = tokenizer_encode_fn
        self.max_seq_len = max_seq_len
        self.chat_tokens = chat_tokens
        self.response_only_loss = response_only_loss
        
        # Load samples
        self.samples = load_jsonl(data_path)
        print(f"Loaded {len(self.samples)} SFT samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _format_prompt_response(self, sample: Dict) -> str:
        """Format prompt and response with chat tokens."""
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        # Format with response marker
        formatted = (
            f"{prompt}\n"
            f"{self.chat_tokens.assistant} {response}"
        )
        return formatted
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        if "input_ids" in sample and "labels" in sample:
            # Pre-tokenized format
            input_ids = torch.tensor(sample["input_ids"][:self.max_seq_len], dtype=torch.long)
            labels = torch.tensor(sample["labels"][:self.max_seq_len], dtype=torch.long)
        else:
            # Text format — tokenize on the fly
            formatted = self._format_prompt_response(sample)
            tokens = self.encode(formatted)[:self.max_seq_len]
            input_ids = torch.tensor(tokens, dtype=torch.long)
            
            if self.response_only_loss:
                # Build labels: -100 for prompt, token ids for response
                prompt_text = sample.get("prompt", "")
                prompt_tokens = len(self.encode(prompt_text))
                labels = input_ids.clone()
                # Mask prompt portion
                mask_end = min(prompt_tokens, len(labels))
                labels[:mask_end] = -100
            else:
                labels = input_ids.clone()
        
        seq_len = input_ids.size(0)
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        # Pad to max_seq_len
        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ── DPODataset ───────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization training.
    
    Expects JSONL format: {"prompt": "...", "chosen": "...", "rejected": "..."}
    or pre-tokenized: {"prompt_ids": [...], "chosen_ids": [...], "rejected_ids": [...]}.
    
    Each sample yields a dict with:
      - prompt_input_ids, prompt_attention_mask
      - chosen_input_ids, chosen_labels, chosen_attention_mask
      - rejected_input_ids, rejected_labels, rejected_attention_mask
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer_encode_fn: Callable[[str], List[int]],
        max_prompt_len: int = 512,
        max_response_len: int = 512,
        chat_tokens: ChatTokens = DEFAULT_CHAT_TOKENS,
    ):
        """
        Args:
            data_path: Path to JSONL file with DPO pairs
            tokenizer_encode_fn: Function to encode text to token IDs
            max_prompt_len: Max tokens for the prompt
            max_response_len: Max tokens for each response
            chat_tokens: Chat token configuration
        """
        self.data_path = data_path
        self.encode = tokenizer_encode_fn
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.chat_tokens = chat_tokens
        
        self.samples = load_jsonl(data_path)
        print(f"Loaded {len(self.samples)} DPO samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _format_sequence(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        """Tokenize prompt and response separately.
        
        Returns:
            (prompt_ids, full_ids) where full_ids = prompt + response + eos
        """
        prompt_ids = self.encode(prompt)[:self.max_prompt_len]
        resp_ids = self.encode(response)[:self.max_response_len]
        full_ids = prompt_ids + resp_ids + [self.chat_tokens.eos]
        return prompt_ids, full_ids
    
    def _pad_sequence(
        self, ids: List[int], max_len: int, pad_val: int = 0
    ) -> torch.Tensor:
        """Pad/truncate sequence to max_len."""
        if len(ids) >= max_len:
            return torch.tensor(ids[:max_len], dtype=torch.long)
        padded = ids + [pad_val] * (max_len - len(ids))
        return torch.tensor(padded, dtype=torch.long)
    
    def _build_labels(
        self, full_ids: List[int], prompt_len: int, max_len: int
    ) -> torch.Tensor:
        """Build labels with -100 masking for prompt tokens."""
        labels = torch.tensor(full_ids[:max_len], dtype=torch.long)
        # Mask prompt and padding
        prompt_end = min(prompt_len, max_len)
        labels[:prompt_end] = -100
        return labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        prompt_ids = self.encode(prompt)[:self.max_prompt_len]
        chosen_ids = self.encode(chosen)[:self.max_response_len]
        rejected_ids = self.encode(rejected)[:self.max_response_len]
        
        prompt_len = len(prompt_ids)
        max_chosen_len = min(prompt_len + len(chosen_ids) + 1, self.max_prompt_len + self.max_response_len)
        max_rej_len = min(prompt_len + len(rejected_ids) + 1, self.max_prompt_len + self.max_response_len)
        
        # Build chosen sequence
        chosen_full = prompt_ids + chosen_ids + [self.chat_tokens.eos]
        rejected_full = prompt_ids + rejected_ids + [self.chat_tokens.eos]
        
        # Pad/truncate
        chosen_ids_t = self._pad_sequence(chosen_full, max_chosen_len)
        rej_ids_t = self._pad_sequence(rejected_full, max_rej_len)
        prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)
        
        # Build labels
        chosen_labels = self._build_labels(chosen_full, prompt_len, max_chosen_len)
        rej_labels = self._build_labels(rejected_full, prompt_len, max_rej_len)
        
        # Attention masks
        chosen_mask = (chosen_ids_t != 0).long()
        rej_mask = (rej_ids_t != 0).long()
        prompt_mask = torch.ones(len(prompt_ids), dtype=torch.long)
        
        return {
            "prompt_input_ids": prompt_ids_t,
            "prompt_attention_mask": prompt_mask,
            "chosen_input_ids": chosen_ids_t,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rej_ids_t,
            "rejected_labels": rej_labels,
            "rejected_attention_mask": rej_mask,
        }


# ── Collation Functions ─────────────────────────────────────────────────

def pretrain_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate PretrainDataset samples into batched tensors."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def sft_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate SFTDataset samples into batched tensors (already padded)."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def dpo_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate DPODataset samples into batched tensors."""
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([b[key] for b in batch])
    return result


def get_collate_fn(dataset_type: str):
    """Get the appropriate collate function for dataset type."""
    mapping = {
        "pretrain": pretrain_collate,
        "sft": sft_collate,
        "dpo": dpo_collate,
    }
    return mapping.get(dataset_type, sft_collate)