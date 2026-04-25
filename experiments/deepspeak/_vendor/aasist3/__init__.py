"""Vendored AASIST3 (MTUCI) speech deepfake detector.

Source: https://github.com/mtuciru/AASIST3 (License: CC-BY-NC-4.0, non-commercial)
Paper: AASIST3 — KAN-Enhanced AASIST Speech Deepfake Detection using SSL
Features and Additional Regularization for the ASVspoof 2024 Challenge.
Weights: https://huggingface.co/MTUCI/AASIST3
"""

from .full_model import aasist3

__all__ = ["aasist3"]
