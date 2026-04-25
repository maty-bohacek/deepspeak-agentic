"""Vendored GenConViT model code.

Source: https://github.com/erprogs/GenConViT (License: GPL-3.0)
Paper: Deepfake Video Detection Using Generative Convolutional Vision
Transformer (Deressa et al.).
Weights: https://huggingface.co/Deressa/GenConViT
"""

from .config import load_config
from .genconvit_ed import GenConViTED
from .genconvit_vae import GenConViTVAE

__all__ = ["GenConViTED", "GenConViTVAE", "load_config"]
