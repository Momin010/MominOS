"""Validation harnesses for MominoMoE.

Provides:
  - golden_harness.py: Dump all intermediate tensors to .npz for validation
  - compare_logits.py: Compare golden .npz against C engine outputs
"""

from .golden_harness import GoldenHarness
from .compare_logits import CompareLogits

__all__ = ["GoldenHarness", "CompareLogits"]