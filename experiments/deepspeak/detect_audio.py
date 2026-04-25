"""Audio deepfake / synthetic speech detection.

Detectors:

    wav2vec-xlsr        — Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
                          (id2label = {0: real, 1: fake})
    AASIST3             — MTUCI/AASIST3 (KAN-AASIST, ASVspoof 2024)
    DF_Arena_500M       — Speech-Arena-2025/DF_Arena_500M_V_1
    DF_Arena_1B         — Speech-Arena-2025/DF_Arena_1B_V_1

Each detector emits ``{"score": P(fake), "label": "real"|"fake"}``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm

from deepspeak.config import (
    AUDIO_DIR, SPEAKER_CLIP_DIR, DETECTION_DIR, CSV_PATH,
    SAMPLE_RATE, ensure_dirs,
)
from deepspeak.detection_metrics import (
    compute_all_metrics, plot_roc_curve, plot_metrics_comparison,
    save_detection_results, compute_metrics_by_engine, plot_by_engine,
    attach_engine,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000  # all four detectors expect 16 kHz mono


# ── Audio loading ──────────────────────────────────────────────────────

def load_audio(audio_path: Path, max_duration: float = 30.0) -> np.ndarray | None:
    """Load audio, resample to 16 kHz mono, cap at ``max_duration`` seconds."""
    try:
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True, duration=max_duration)
        if len(y) < 1600:  # < 0.1 s
            return None
        return y
    except Exception:
        return None


def _label(score: float) -> str:
    return "fake" if score > 0.5 else "real"


# ── Detector 1: wav2vec-xlsr (Gustking) ───────────────────────────────

_wav2vec_model = None
_wav2vec_fe = None


def _load_wav2vec_xlsr():
    global _wav2vec_model, _wav2vec_fe
    if _wav2vec_model is not None:
        return _wav2vec_model, _wav2vec_fe

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
    _wav2vec_fe = AutoFeatureExtractor.from_pretrained(name)
    _wav2vec_model = AutoModelForAudioClassification.from_pretrained(name).to(DEVICE).eval()
    return _wav2vec_model, _wav2vec_fe


def detect_wav2vec_xlsr(audio_path: Path) -> dict:
    """wav2vec-xlsr: Wav2Vec2ForSequenceClassification, id2label={0:real,1:fake}."""
    y = load_audio(audio_path)
    if y is None:
        return {"score": 0.5, "label": "unknown"}

    model, fe = _load_wav2vec_xlsr()

    chunk = SR * 10  # 10 s
    fake_probs = []
    for start in range(0, len(y), chunk):
        seg = y[start:start + chunk]
        if len(seg) < SR:  # drop stragglers < 1 s
            continue
        inputs = fe(seg, sampling_rate=SR, return_tensors="pt")
        input_values = inputs.input_values.to(DEVICE)
        attention_mask = inputs.attention_mask.to(DEVICE) if "attention_mask" in inputs else None
        with torch.no_grad():
            logits = model(
                input_values,
                attention_mask=attention_mask,
            ).logits
        probs = torch.softmax(logits, dim=-1)[0]
        fake_probs.append(float(probs[1]))

    if not fake_probs:
        return {"score": 0.5, "label": "unknown"}
    avg = float(np.mean(fake_probs))
    return {"score": avg, "label": _label(avg), "chunk_scores": fake_probs}


# ── Detector 2: AASIST3 (MTUCI) ───────────────────────────────────────

_aasist3_model = None
AASIST3_NB_SAMP = 64600  # ~4 s at 16 kHz


def _load_aasist3():
    global _aasist3_model
    if _aasist3_model is not None:
        return _aasist3_model

    from deepspeak._vendor.aasist3 import aasist3 as aasist3_cls
    _aasist3_model = aasist3_cls.from_pretrained("MTUCI/AASIST3").to(DEVICE).eval()
    return _aasist3_model


def detect_aasist3(audio_path: Path) -> dict:
    """AASIST3: fixed 64 600-sample windows, class 1 = spoof."""
    y = load_audio(audio_path)
    if y is None:
        return {"score": 0.5, "label": "unknown"}

    model = _load_aasist3()

    # Slide non-overlapping 4 s windows; pad last if short.
    windows: list[np.ndarray] = []
    for start in range(0, len(y), AASIST3_NB_SAMP):
        seg = y[start:start + AASIST3_NB_SAMP]
        if len(seg) < AASIST3_NB_SAMP:
            if len(seg) < SR:  # < 1 s of material — discard
                continue
            seg = np.pad(seg, (0, AASIST3_NB_SAMP - len(seg)))
        windows.append(seg)

    if not windows:
        return {"score": 0.5, "label": "unknown"}

    x = torch.from_numpy(np.stack(windows)).float().to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    avg = float(np.mean(probs))
    return {"score": avg, "label": _label(avg), "chunk_scores": [float(p) for p in probs]}


# ── Detectors 3 & 4: DF_Arena 500M / 1B ───────────────────────────────

_df_arena_pipes: dict[str, object] = {}


def _load_df_arena(repo_id: str):
    """Lazy-load DF_Arena anti-spoofing pipeline (trust_remote_code)."""
    if repo_id in _df_arena_pipes:
        return _df_arena_pipes[repo_id]

    from transformers import pipeline
    pipe = pipeline(
        "antispoofing",
        model=repo_id,
        trust_remote_code=True,
        device=0 if DEVICE == "cuda" else -1,
    )
    _df_arena_pipes[repo_id] = pipe
    return pipe


def _detect_df_arena(audio_path: Path, repo_id: str) -> dict:
    y = load_audio(audio_path)
    if y is None:
        return {"score": 0.5, "label": "unknown"}

    pipe = _load_df_arena(repo_id)
    # Chunk long audio into 10 s windows to match wav2vec-xlsr handling;
    # pipeline accepts a numpy array of raw samples at 16 kHz.
    chunk = SR * 10
    fake_probs = []
    for start in range(0, len(y), chunk):
        seg = y[start:start + chunk]
        if len(seg) < SR:
            continue
        try:
            result = pipe(seg)
        except Exception:
            continue
        # result: {'label': 'spoof'|'bonafide', 'all_scores': {'spoof': p, 'bonafide': 1-p}, ...}
        all_scores = result.get("all_scores") if isinstance(result, dict) else None
        if isinstance(all_scores, dict) and "spoof" in all_scores:
            fake_probs.append(float(all_scores["spoof"]))
        elif isinstance(result, dict) and "score" in result:
            score = float(result["score"])
            fake_probs.append(score if result.get("label") == "spoof" else 1.0 - score)

    if not fake_probs:
        return {"score": 0.5, "label": "unknown"}
    avg = float(np.mean(fake_probs))
    return {"score": avg, "label": _label(avg), "chunk_scores": fake_probs}


def detect_df_arena_500m(audio_path: Path) -> dict:
    """DF_Arena 500M V1 (Speech-Arena-2025)."""
    return _detect_df_arena(audio_path, "Speech-Arena-2025/DF_Arena_500M_V_1")


def detect_df_arena_1b(audio_path: Path) -> dict:
    """DF_Arena 1B V1 (Speech-Arena-2025)."""
    return _detect_df_arena(audio_path, "Speech-Arena-2025/DF_Arena_1B_V_1")


# ── Main detection pipeline ────────────────────────────────────────────

DETECTORS = {
    "wav2vec-xlsr":  detect_wav2vec_xlsr,
    "AASIST3":       detect_aasist3,
    "DF_Arena_500M": detect_df_arena_500m,
    "DF_Arena_1B":   detect_df_arena_1b,
}


def get_speaker_audio_path(video_id: str, side: str) -> Path | None:
    """Per-side masked audio if present, else the full-mix audio."""
    path = AUDIO_DIR / f"{video_id}_{side}.wav"
    if path.exists():
        return path
    path = AUDIO_DIR / f"{video_id}.wav"
    if path.exists():
        return path
    return None


def run_audio_detection(df: pd.DataFrame):
    """Run all audio detectors and compute metrics."""
    ensure_dirs()

    cache_path = DETECTION_DIR / "audio_raw_predictions.json"
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
                audio_path = get_speaker_audio_path(vid, side)
                if audio_path is None:
                    continue

                is_fake = 1 if side == agent_pos else 0

                try:
                    result = det_func(audio_path)
                except Exception as e:
                    print(f"    Error on {vid}/{side}: {e}")
                    continue

                predictions.append({
                    "video_id": vid,
                    "side": side,
                    "is_fake": is_fake,
                    "score": result["score"],
                    "pred_label": result["label"],
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

    save_detection_results(results, "audio")
    if results:
        plot_roc_curve(results, "audio")
        plot_metrics_comparison(results, "audio")
        plot_by_engine(results, "audio")

    return results


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_audio_detection(df)
