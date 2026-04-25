"""Vendored GenConViT wrapper — accepts explicit weight paths.

Upstream used ``weight/<name>.pth`` relative to cwd; we take absolute paths
so it works from any working directory.

Checkpoints were trained with ``timm==0.6.5``; we remap the Swin-Tiny
embedder keys to the current timm layout on load.
"""

import re
import warnings

import torch
import torch.nn as nn

from .genconvit_ed import GenConViTED
from .genconvit_vae import GenConViTVAE


_SWIN_DOWNSAMPLE_RE = re.compile(r"^(.*layers\.)(\d+)(\.downsample\..*)$")
_SWIN_HEAD_RE = re.compile(r"^(.*)\.head\.(weight|bias)$")


def _remap_old_timm_swin(state: dict) -> dict:
    """Remap timm 0.6.5 Swin keys to the current timm layout.

    Two changes between 0.6.5 and current timm Swin:
      - ``downsample`` moved from the end of stages {0,1,2} to the start of
        stages {1,2,3}. Shift the stage index by +1.
      - Classifier ``head`` became a ``ClassifierHead`` module: rename
        ``head.{weight,bias}`` → ``head.fc.{weight,bias}``.

    Keys that don't match either pattern pass through unchanged, so the
    ConvNeXt backbone keys (which use ``stages.*`` and already have
    ``head.fc.*``) and the top-level ``fc``/``fc2`` layers are unaffected.
    """
    remapped = {}
    for k, v in state.items():
        m = _SWIN_DOWNSAMPLE_RE.match(k)
        if m:
            prefix, idx, rest = m.groups()
            k = f"{prefix}{int(idx) + 1}{rest}"
        m = _SWIN_HEAD_RE.match(k)
        if m:
            k = f"{m.group(1)}.head.fc.{m.group(2)}"
        remapped[k] = v
    return remapped


def _load_state_dict(model: nn.Module, ckpt_path: str, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=torch.device(device), weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _remap_old_timm_swin(state)
    # ``relative_position_index`` is registered as a buffer in 0.6.5 but
    # dropped from state_dict in current timm; allow non-strict load so
    # that residual delta surfaces as a warning rather than a hard error.
    missing, unexpected = model.load_state_dict(state, strict=False)
    leftover_missing = [k for k in missing if "relative_position_index" not in k]
    leftover_unexpected = [k for k in unexpected if "relative_position_index" not in k]
    if leftover_missing or leftover_unexpected:
        warnings.warn(
            f"GenConViT load: missing={leftover_missing[:5]}"
            f"{' …' if len(leftover_missing) > 5 else ''}, "
            f"unexpected={leftover_unexpected[:5]}"
            f"{' …' if len(leftover_unexpected) > 5 else ''}"
        )


class GenConViT(nn.Module):
    """GenConViT detector — supports 'ed', 'vae', or 'genconvit' (both)."""

    def __init__(self, config, ed_weights=None, vae_weights=None, net="ed", fp16=False):
        super().__init__()
        self.net = net
        self.fp16 = fp16

        if net == "ed":
            self.model_ed = GenConViTED(config)
            _load_state_dict(self.model_ed, ed_weights)
            self.model_ed.eval()
            if fp16:
                self.model_ed.half()

        elif net == "vae":
            self.model_vae = GenConViTVAE(config)
            _load_state_dict(self.model_vae, vae_weights)
            self.model_vae.eval()
            if fp16:
                self.model_vae.half()

        else:  # both
            self.model_ed = GenConViTED(config)
            self.model_vae = GenConViTVAE(config)
            _load_state_dict(self.model_ed, ed_weights)
            _load_state_dict(self.model_vae, vae_weights)
            self.model_ed.eval()
            self.model_vae.eval()
            if fp16:
                self.model_ed.half()
                self.model_vae.half()

    def forward(self, x):
        if self.net == "ed":
            return self.model_ed(x)
        if self.net == "vae":
            out, _ = self.model_vae(x)
            return out
        x1 = self.model_ed(x)
        x2, _ = self.model_vae(x)
        return torch.cat((x1, x2), dim=0)
