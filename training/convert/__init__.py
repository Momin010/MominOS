"""Model conversion and export tools for MominoMoE.

Provides:
  - quant.py: Q8_0 block quantization (block size 32)
  - convert_model.py: PyTorch → MOM1 binary converter
  - export_config.py: C header generator (model_config.h)
"""

from ..train_moe.quant import quantize_q8_0, dequantize_q8_0

__all__ = ["quantize_q8_0", "dequantize_q8_0"]