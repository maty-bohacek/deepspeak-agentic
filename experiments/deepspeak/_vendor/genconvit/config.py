"""GenConViT config (vendored).

Upstream loaded this from a YAML file; we inline it to avoid file lookups.
"""

GENCONVIT_CONFIG = {
    "model": {
        "backbone": "convnext_tiny",
        "embedder": "swin_tiny_patch4_window7_224",
        "latent_dims": 12544,
    },
    "batch_size": 32,
    "epoch": 1,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_classes": 2,
    "img_size": 224,
}


def load_config():
    return GENCONVIT_CONFIG
