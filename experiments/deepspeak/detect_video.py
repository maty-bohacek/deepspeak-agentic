"""Video deepfake detection.

Detectors (all operate on evenly-sampled frames from the per-side combined clip):

    GenConViT-ED    — Deressa et al., ``Deressa/GenConViT`` (AE-based)
    GenConViT-VAE   — Deressa et al., ``Deressa/GenConViT`` (VAE-based)
    CLIP-GDD        — yermandy, ``yermandy/deepfake-detection`` (CLIP ViT-L/14,
                       distributed as TorchScript)
    GenD-CLIP-L14   — yermandy, ``yermandy/GenD_CLIP_L_14`` (WACV 2026)

Each detector emits ``{"score": P(fake), "label": "real"|"fake", "n_frames": int}``.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from deepspeak.config import (
    SPLIT_VIDEO_DIR, SPEAKER_CLIP_DIR, DETECTION_DIR,
    CSV_PATH, ensure_dirs,
)
from deepspeak.detection_metrics import (
    compute_all_metrics, plot_roc_curve, plot_metrics_comparison,
    save_detection_results, compute_metrics_by_engine, plot_by_engine,
    attach_engine,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FRAMES = 10


# ── Frame extraction ──────────────────────────────────────────────────

def extract_frames(video_path: Path, n_frames: int = N_FRAMES) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video using ffmpeg."""
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "fps=1,select='not(mod(n\\,1))'",
            "-frames:v", str(n_frames * 3),
            "-q:v", "2",
            f"{tmpdir}/frame_%04d.jpg",
        ]
        subprocess.run(cmd, capture_output=True)

        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))
        if not frame_files:
            return []

        indices = np.linspace(0, len(frame_files) - 1, min(n_frames, len(frame_files)), dtype=int)
        for idx in indices:
            try:
                frames.append(Image.open(frame_files[idx]).convert("RGB"))
            except Exception:
                continue
    return frames


def _score_to_result(fake_scores: list[float]) -> dict:
    if not fake_scores:
        return {"score": 0.5, "label": "unknown", "n_frames": 0}
    avg = float(np.mean(fake_scores))
    return {
        "score": avg,
        "label": "fake" if avg > 0.5 else "real",
        "n_frames": len(fake_scores),
        "frame_scores": [float(s) for s in fake_scores],
    }


# ── Detector 1 & 2: GenConViT ED and VAE ──────────────────────────────

_genconvit_models: dict[str, torch.nn.Module] = {}
_genconvit_transform = None


def _genconvit_transform_fn():
    """Standard 224×224 ImageNet transform used by GenConViT."""
    global _genconvit_transform
    if _genconvit_transform is None:
        from torchvision import transforms
        _genconvit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return _genconvit_transform


def _load_genconvit(net: str) -> torch.nn.Module:
    """Lazy-load the GenConViT model (downloads the .pth once from HF)."""
    if net in _genconvit_models:
        return _genconvit_models[net]

    from huggingface_hub import hf_hub_download
    from deepspeak._vendor.genconvit import load_config
    from deepspeak._vendor.genconvit.genconvit import GenConViT

    fname = "genconvit_ed_inference.pth" if net == "ed" else "genconvit_vae_inference.pth"
    weights_path = hf_hub_download(repo_id="Deressa/GenConViT", filename=fname)

    config = load_config()
    kwargs = dict(config=config, net=net, fp16=False)
    if net == "ed":
        kwargs["ed_weights"] = weights_path
    else:
        kwargs["vae_weights"] = weights_path

    model = GenConViT(**kwargs).to(DEVICE).eval()
    _genconvit_models[net] = model
    return model


def _genconvit_predict(video_path: Path, net: str) -> dict:
    frames = extract_frames(video_path)
    if not frames:
        return {"score": 0.5, "label": "unknown", "n_frames": 0}

    model = _load_genconvit(net)
    tfm = _genconvit_transform_fn()
    batch = torch.stack([tfm(f) for f in frames]).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
    # Upstream ``pred_func.real_or_fake`` XORs the argmax with 1, so class 0 = fake.
    # It also uses sigmoid rather than softmax — we mirror that here.
    probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
    return _score_to_result([float(p) for p in probs])


def detect_genconvit_ed(video_path: Path) -> dict:
    """GenConViT ED (AE variant). Weights: Deressa/GenConViT."""
    return _genconvit_predict(video_path, "ed")


def detect_genconvit_vae(video_path: Path) -> dict:
    """GenConViT VAE variant. Weights: Deressa/GenConViT."""
    return _genconvit_predict(video_path, "vae")


# ── Detector 3: CLIP-GDD (yermandy/deepfake-detection) ────────────────

_clip_gdd_model = None
_clip_gdd_processor = None


def _load_clip_gdd():
    """Lazy-load the TorchScript CLIP-GDD model."""
    global _clip_gdd_model, _clip_gdd_processor
    if _clip_gdd_model is not None:
        return _clip_gdd_model, _clip_gdd_processor

    from huggingface_hub import hf_hub_download
    from transformers import CLIPProcessor

    model_path = hf_hub_download(
        repo_id="yermandy/deepfake-detection", filename="model.torchscript",
    )
    _clip_gdd_model = torch.jit.load(model_path, map_location=DEVICE).eval().to(DEVICE)
    _clip_gdd_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return _clip_gdd_model, _clip_gdd_processor


def detect_clip_gdd(video_path: Path) -> dict:
    """CLIP-GDD (yermandy/deepfake-detection). Output: [p_real, p_fake]."""
    frames = extract_frames(video_path)
    if not frames:
        return {"score": 0.5, "label": "unknown", "n_frames": 0}

    model, processor = _load_clip_gdd()
    pixel_values = torch.stack([
        processor(images=f, return_tensors="pt")["pixel_values"][0] for f in frames
    ]).to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values)
    probs = logits.softmax(dim=1)[:, 1].float().cpu().numpy()
    return _score_to_result([float(p) for p in probs])


# ── Detector 4: GenD CLIP L/14 (yermandy/GenD_CLIP_L_14) ──────────────

_gend_model = None


def _load_gend():
    global _gend_model
    if _gend_model is not None:
        return _gend_model

    from deepspeak._vendor.gend.modeling_gend import GenD, GenDConfig
    from huggingface_hub import hf_hub_download

    repo_id = "yermandy/GenD_CLIP_L_14"

    # Newer `transformers` raises when `from_pretrained` is called while a
    # meta-device context is active, which is exactly what the outer
    # `PreTrainedModel.from_pretrained` sets up before invoking `__init__`.
    # `GenD.__init__` then calls `CLIPModel.from_pretrained` internally,
    # triggering that error. Construct the module graph directly and load
    # the published state dict afterwards to sidestep the nesting.
    config = GenDConfig.from_pretrained(repo_id)
    model = GenD(config)

    try:
        from safetensors.torch import load_file
        weights_path = hf_hub_download(repo_id, "model.safetensors")
        state = load_file(weights_path)
    except Exception:
        weights_path = hf_hub_download(repo_id, "pytorch_model.bin")
        state = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state, strict=False)
    _gend_model = model.to(DEVICE).eval()
    return _gend_model


def detect_gend_clip_l14(video_path: Path) -> dict:
    """GenD CLIP L/14 (yermandy/GenD_CLIP_L_14). Output: [p_real, p_fake]."""
    frames = extract_frames(video_path)
    if not frames:
        return {"score": 0.5, "label": "unknown", "n_frames": 0}

    model = _load_gend()
    batch = torch.stack([model.feature_extractor.preprocess(f) for f in frames]).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)
    probs = logits.softmax(dim=-1)[:, 1].cpu().numpy()
    return _score_to_result([float(p) for p in probs])


# ── Main detection pipeline ────────────────────────────────────────────

DETECTORS = {
    "GenConViT-ED":  detect_genconvit_ed,
    "GenConViT-VAE": detect_genconvit_vae,
    "CLIP-GDD":      detect_clip_gdd,
    "GenD-CLIP-L14": detect_gend_clip_l14,
}


def run_video_detection(df: pd.DataFrame):
    """Run all video detectors on all speaker clips and compute metrics."""
    ensure_dirs()

    cache_path = DETECTION_DIR / "video_raw_predictions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            all_predictions = json.load(f)
    else:
        all_predictions = {}

    id_to_engine = {
        row["final_conversation_id"]: row.get("engine")
        for _, row in df.iterrows()
    }

    for det_name, det_func in DETECTORS.items():
        if det_name in all_predictions:
            print(f"  {det_name}: using cached predictions")
            continue

        print(f"\n  Running {det_name}...")
        predictions = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {det_name}"):
            vid = row["final_conversation_id"]
            human_pos = row.get("human_position", "right")
            agent_pos = "left" if human_pos == "right" else "right"
            engine = row.get("engine")

            for side in ["left", "right"]:
                clip_path = SPEAKER_CLIP_DIR / vid / f"{side}_combined.mp4"
                if not clip_path.exists():
                    clip_path = SPLIT_VIDEO_DIR / side / f"{vid}.mp4"
                if not clip_path.exists():
                    continue

                is_fake = 1 if side == agent_pos else 0

                try:
                    result = det_func(clip_path)
                except Exception as e:
                    print(f"    Error on {vid}/{side}: {e}")
                    continue

                predictions.append({
                    "video_id": vid,
                    "side": side,
                    "is_fake": is_fake,
                    "score": result["score"],
                    "pred_label": result["label"],
                    "n_frames": result["n_frames"],
                    "engine": str(engine) if engine is not None and engine == engine else None,
                })

        all_predictions[det_name] = predictions

        with open(cache_path, "w") as f:
            json.dump(all_predictions, f, indent=2)

    results = {}
    for det_name in DETECTORS:
        preds = all_predictions.get(det_name, [])
        if not preds:
            continue
        attach_engine(preds, id_to_engine)
        y_true = [p["is_fake"] for p in preds]
        y_scores = [p["score"] for p in preds]
        metrics = compute_all_metrics(y_true, y_scores)
        by_engine = compute_metrics_by_engine(preds)
        results[det_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_scores": y_scores,
            "predictions": preds,
            "by_engine": by_engine,
        }
        print(f"\n  {det_name}: AUROC={metrics['auroc']:.3f}, "
              f"EER={metrics['eer']:.3f}, Acc={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1_macro']:.3f}")
        for eng, ed in by_engine.items():
            em = ed["metrics"]
            print(f"      engine={eng}: AUROC={em['auroc']:.3f} "
                  f"EER={em['eer']:.3f} Acc={em['accuracy']:.3f} "
                  f"F1={em['f1_macro']:.3f} (n={ed['n']})")

    save_detection_results(results, "video")
    if results:
        plot_roc_curve(results, "video")
        plot_metrics_comparison(results, "video")
        plot_by_engine(results, "video")

    return results


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_video_detection(df)
