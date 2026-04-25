"""Vendored GenD (yermandy) deepfake detector.

Source: https://github.com/yermandy/GenD (License: MIT)
Paper: Deepfake Detection that Generalizes Across Benchmarks (WACV 2026).
Weights: https://huggingface.co/yermandy/GenD_CLIP_L_14
"""

from .modeling_gend import GenD, GenDConfig

__all__ = ["GenD", "GenDConfig"]
