"""MominOS Training Package.

Top-level package for the MominoMoE training pipeline.
Subpackages:
  - train_moe: Model architecture (config, layers, model)
  - convert: Model export (Q8_0 quantization, MOM1 binary, C headers)
  - eval: Validation harnesses (golden intermediates, logit comparison)
  - data: Synthetic data generation (templates, prompts, quality filtering)
  - training: Training infrastructure (datasets, trainers, utilities)
"""

__version__ = "0.1.0"