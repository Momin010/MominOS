#!/usr/bin/env python3
"""Byte-level tokenizer for MominoMoE SFT training.

Since MominoMoE-1.2B has random weights (no pretrained tokenizer),
we use a simple byte-level encoding where each byte is mapped to a
unique token ID. This allows the model to learn character-level
patterns during SFT.

Token ID mapping:
- 0: PAD
- 1: BOS (beginning of sequence)
- 2: EOS (end of sequence)
- 3-258: Raw byte values 0-255
- 259+: Unused (reserved for future special tokens)

Usage:
    from byte_tokenizer import ByteTokenizer
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)
"""

from typing import List, Optional


class ByteTokenizer:
    """Simple byte-level tokenizer for MominoMoE training."""

    SPECIAL_TOKENS = {
        "pad": 0,
        "bos": 1,
        "eos": 2,
    }

    # Byte values 0-255 are mapped to IDs 3-258
    BYTE_OFFSET = 3
    VOCAB_SIZE = 32000  # Matches MominoMoEConfig

    def __init__(self):
        self.pad_id = self.SPECIAL_TOKENS["pad"]
        self.bos_id = self.SPECIAL_TOKENS["bos"]
        self.eos_id = self.SPECIAL_TOKENS["eos"]

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Encode text to token IDs using byte-level mapping.

        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token
            max_length: Maximum sequence length (truncates if exceeded)

        Returns:
            List of token IDs
        """
        # Convert each byte to token ID (byte + offset)
        ids = [b + self.BYTE_OFFSET for b in text.encode("utf-8")]

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_length and len(ids) > max_length:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special: Filter out special tokens (pad/bos/eos)

        Returns:
            Decoded text string
        """
        byte_values = []
        for tid in ids:
            if skip_special and tid in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if self.BYTE_OFFSET <= tid < self.BYTE_OFFSET + 256:
                byte_values.append(tid - self.BYTE_OFFSET)
            # Skip out-of-range IDs (unused tokens)

        try:
            return bytes(byte_values).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def __call__(self, text: str, **kwargs) -> List[int]:
        """Alias for encode."""
        return self.encode(text, **kwargs)


# Global singleton for convenience
tokenizer = ByteTokenizer()
encode = tokenizer.encode
decode = tokenizer.decode