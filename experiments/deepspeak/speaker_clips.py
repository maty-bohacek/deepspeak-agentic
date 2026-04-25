"""Per-side selective-audio muxing with pyannote neural diarization.

Pipeline
--------
We use two off-the-shelf components and nothing bespoke:

  * **pyannote.audio 3.1 speaker-diarization** on the full (trimmed)
    audio, constrained to exactly two speakers. pyannote produces
    turn-level segments whose boundaries respect natural speech
    structure — it doesn't cut inside a word or a sentence.
  * **MediaPipe FaceMesh** (via ``lip_motion.py``) to match each
    pyannote cluster to a video side: for each cluster we aggregate
    lip motion on the left half vs. the right half across all of
    that cluster's turns; the side with greater aggregate motion is
    declared that cluster's side. This is a 2-way classification
    over minutes of video, so it is very robust — camera distance,
    blur, and occasional missed detections barely affect it.

Post-processing for clean audio
-------------------------------
  * Adjacent same-speaker turns separated by a gap ≤ 1.5 s are merged,
    so a natural breath inside a sentence doesn't break the audio.
  * Each (possibly merged) turn is padded by 80 ms at the start and
    200 ms at the end so we never clip the leading consonant or the
    trailing release of a word.
  * A 40 ms cosine fade is applied at every mask boundary for a
    click-free result.

Output
------
Per video, we emit the familiar files:

  * ``data/audio/<vid>_left.wav``, ``<vid>_right.wav``
    — full duration, containing only that side's speaker.
  * ``data/speaker_clips/<vid>/{left,right}_combined.mp4``
    — split-screen half (video stream copied) with the masked audio.
  * ``data/speaker_clips/<vid>/side_timeline.json``
    — diagnostic JSON with the merged turn list and diagnostics.
  * ``data/speaker_clips/<vid>/speaker_map.json``
    — ``{left: HUMAN|AGENT, right: HUMAN|AGENT}`` from the CSV.
"""

from __future__ import annotations

import json
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# pyannote.audio emits a very long torchcodec warning at import time even
# when the caller pre-loads audio (our path). Silence just that one.
warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly.*",
    category=UserWarning,
)

from deepspeak.config import (
    AUDIO_DIR,
    CSV_PATH,
    FLIP_SIDE_IDS,
    HF_TOKEN,
    SPEAKER_CLIP_DIR,
    SPLIT_VIDEO_DIR,
    ensure_dirs,
)
from deepspeak.lip_motion import compute_lip_signals


# ── Tuning knobs ───────────────────────────────────────────────────────
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
NUM_SPEAKERS = 2

MERGE_GAP_SEC = 1.5          # merge consecutive same-speaker turns across gaps ≤ this
PAD_START_SEC = 0.08         # extend each turn this much to its left
PAD_END_SEC = 0.20           # extend each turn this much to its right
FADE_MS = 40                 # cosine fade at every mask boundary


# ── Muxing (ffmpeg) ────────────────────────────────────────────────────

def _mux_video_with_audio(
    video_src: Path, audio_src: Path, dst: Path
) -> Path | None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_src),
        "-i", str(audio_src),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return dst


# ── pyannote pipeline ─────────────────────────────────────────────────

_PYANNOTE_PIPELINE = None


def _get_pyannote_pipeline():
    """Load pyannote.audio 3.1 once per process, on GPU if available."""
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is not None:
        return _PYANNOTE_PIPELINE

    token = os.environ.get("HF_TOKEN") or HF_TOKEN
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "pyannote.audio is not installed. Add it to requirements and "
            "`pip install pyannote.audio`."
        ) from e

    # pyannote.audio 3.1+ uses `token=`; older versions used `use_auth_token=`.
    # Try the modern call first and fall back if someone is on an older build.
    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, token=token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=token)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {PYANNOTE_MODEL}. Two things must be true:\n"
            f"  1) your HF_TOKEN must be valid;\n"
            f"  2) the logged-in account must have accepted the model terms at\n"
            f"     https://hf.co/pyannote/speaker-diarization-3.1 and\n"
            f"     https://hf.co/pyannote/segmentation-3.0.\n"
            f"Underlying error: {e}"
        ) from e

    # Move to GPU if available.
    try:
        import torch
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:
        pass

    _PYANNOTE_PIPELINE = pipeline
    return pipeline


def _load_audio_for_pyannote(audio_path: Path) -> dict:
    """Pre-load audio as an in-memory tensor dict.

    pyannote's default IO backend goes through torchcodec, which often
    fails on machines without the right CUDA / FFmpeg combo. Passing the
    audio as ``{"waveform": tensor, "sample_rate": sr}`` sidesteps that
    entirely — pyannote just consumes the tensor.
    """
    import soundfile as sf
    import torch

    y, sr = sf.read(str(audio_path), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    waveform = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)  # (1, time)
    return {"waveform": waveform, "sample_rate": int(sr)}


def run_diarization(audio_path: Path) -> list[dict]:
    """Return sorted list of {start, end, speaker} turns for one video."""
    pipeline = _get_pyannote_pipeline()
    audio_in = _load_audio_for_pyannote(audio_path)
    diar = pipeline(audio_in, num_speakers=NUM_SPEAKERS)
    # pyannote.audio 4.x wraps the Annotation in a DiarizeOutput; older
    # versions returned the Annotation directly. Unwrap if needed.
    annotation = getattr(diar, "speaker_diarization", diar)
    turns: list[dict] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    turns.sort(key=lambda s: s["start"])
    return turns


# ── Cluster → side mapping ────────────────────────────────────────────

def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    win = min(win, x.size)
    k = np.ones(win, dtype=np.float32) / float(win)
    m = np.convolve(x.astype(np.float32), k, mode="same")
    m2 = np.convolve((x * x).astype(np.float32), k, mode="same")
    var = np.maximum(m2 - m * m, 0.0)
    return np.sqrt(var).astype(np.float32)


def assign_clusters_to_sides(
    video_id: str,
    turns: list[dict],
) -> dict[str, str] | None:
    """For each pyannote cluster, decide whether it lives on the left or
    right side of the split-screen based on MediaPipe lip motion aggregated
    over all of its turns.

    Returns ``{cluster_label: "left"|"right"}``.
    """
    lip = compute_lip_signals(video_id)
    if lip is None:
        return None

    ts = np.asarray(lip["timestamps"], dtype=np.float32)
    l_mar = np.asarray(
        [0.0 if v is None else float(v) for v in lip["left_mar"]],
        dtype=np.float32,
    )
    r_mar = np.asarray(
        [0.0 if v is None else float(v) for v in lip["right_mar"]],
        dtype=np.float32,
    )

    # Rolling std of MAR over ~240 ms = articulation signal.
    # MAR is dimensionless so this works at any face size.
    fps = float(lip.get("sample_fps", 25.0)) or 25.0
    win = max(1, int(round(0.24 * fps)))
    l_activity = _rolling_std(l_mar, win)
    r_activity = _rolling_std(r_mar, win)

    # Per-cluster aggregate: sum of (mean lip activity × duration) on each side.
    votes: dict[str, dict[str, float]] = {}
    for t in turns:
        start, end = float(t["start"]), float(t["end"])
        if end <= start:
            continue
        mask = (ts >= start) & (ts <= end)
        if not mask.any():
            mask = (ts >= start - 0.5) & (ts <= end + 0.5)
        if not mask.any():
            continue
        l_mean = float(np.nanmean(l_activity[mask])) if l_activity[mask].size else 0.0
        r_mean = float(np.nanmean(r_activity[mask])) if r_activity[mask].size else 0.0
        dur = end - start
        v = votes.setdefault(t["speaker"], {"left": 0.0, "right": 0.0})
        v["left"] += l_mean * dur
        v["right"] += r_mean * dur

    if not votes:
        return None

    # Rank clusters by strength of left-preference; assign strongest first
    # and don't let two clusters claim the same side.
    preferences: list[tuple[str, float]] = []
    for spk, v in votes.items():
        total = v["left"] + v["right"]
        l_share = (v["left"] / total) if total > 0 else 0.5
        preferences.append((spk, l_share))
    preferences.sort(key=lambda p: abs(p[1] - 0.5), reverse=True)

    assignment: dict[str, str] = {}
    taken: set[str] = set()
    for spk, l_share in preferences:
        want = "left" if l_share >= 0.5 else "right"
        if want in taken:
            want = "right" if want == "left" else "left"
        assignment[spk] = want
        taken.add(want)
        if len(taken) == 2:
            break

    # Any remaining clusters (shouldn't happen with num_speakers=2) inherit
    # the side of the least-opposed cluster.
    for spk, _ in preferences:
        assignment.setdefault(spk, "left" if "right" in taken else "right")

    return assignment


# ── Turn merging / padding ────────────────────────────────────────────

def _merge_same_side(turns: list[dict], max_gap: float = MERGE_GAP_SEC) -> list[dict]:
    """Merge consecutive turns with the same ``side`` separated by ≤ max_gap."""
    if not turns:
        return []
    turns = sorted(turns, key=lambda s: s["start"])
    out: list[dict] = [dict(turns[0])]
    for seg in turns[1:]:
        last = out[-1]
        if seg.get("side") == last.get("side") and seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            out.append(dict(seg))
    return out


def _pad_turns(
    turns: list[dict],
    duration: float,
    start_pad: float = PAD_START_SEC,
    end_pad: float = PAD_END_SEC,
) -> list[dict]:
    padded = []
    for seg in turns:
        padded.append({
            **seg,
            "start": max(0.0, float(seg["start"]) - start_pad),
            "end": min(duration, float(seg["end"]) + end_pad),
        })
    # Overlaps inside one side after padding: merge them (no more fading
    # in the middle of continuous speech).
    padded.sort(key=lambda s: s["start"])
    out: list[dict] = []
    for seg in padded:
        if out and seg.get("side") == out[-1].get("side") and seg["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], seg["end"])
        else:
            out.append(dict(seg))
    return out


# ── Audio mask builder ────────────────────────────────────────────────

def _build_mask(
    turns: list[dict],
    side: str,
    n_samples: int,
    sr: int,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    """Full-duration [0,1] sample-level mask for one side, with cosine
    fades at every turn boundary."""
    mask = np.zeros(n_samples, dtype=np.float32)
    fade = max(1, int(sr * fade_ms / 1000))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float32)))

    for seg in turns:
        if seg.get("side") != side:
            continue
        s = int(max(0.0, float(seg["start"])) * sr)
        e = int(min(n_samples / sr, float(seg["end"])) * sr)
        if e <= s:
            continue
        mask[s:e] = 1.0
        fi_end = min(s + fade, e)
        if fi_end > s:
            mask[s:fi_end] = np.minimum(mask[s:fi_end], ramp[: fi_end - s])
        fo_start = max(s, e - fade)
        if e > fo_start:
            mask[fo_start:e] = np.minimum(mask[fo_start:e], ramp[: e - fo_start][::-1])
    return mask


# ── Per-video pipeline ────────────────────────────────────────────────

def build_selective_for_video(
    video_id: str,
    human_position: str = "right",
    force: bool = False,
) -> dict | None:
    """Produce per-side masked WAV + muxed MP4 for one video."""
    import soundfile as sf

    ensure_dirs()
    out_dir = SPEAKER_CLIP_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / "side_timeline.json"
    if marker.exists() and not force:
        with open(marker) as f:
            return json.load(f)

    full_audio = AUDIO_DIR / f"{video_id}.wav"
    if not full_audio.exists():
        print(f"  [{video_id}] full audio missing; skipping")
        return None

    # 1. Neural diarization.
    tqdm.write(f"  [{video_id}] pyannote diarization...")
    turns = run_diarization(full_audio)
    if not turns:
        print(f"  [{video_id}] pyannote returned no turns; skipping")
        return None
    tqdm.write(f"  [{video_id}]   {len(turns)} turns")

    # 2. Cluster → side via lip motion.
    tqdm.write(f"  [{video_id}] MediaPipe lip motion (cluster→side)...")
    cluster_to_side = assign_clusters_to_sides(video_id, turns)
    if not cluster_to_side:
        print(f"  [{video_id}] lip motion unavailable; cannot match clusters")
        return None
    tqdm.write(f"  [{video_id}]   clusters → {cluster_to_side}")

    # Manual override: invert the mapping for IDs known to be flipped.
    side_flipped = video_id in FLIP_SIDE_IDS
    if side_flipped:
        cluster_to_side = {
            k: ("right" if v == "left" else "left")
            for k, v in cluster_to_side.items()
        }
        tqdm.write(f"  [{video_id}]   FLIPPED via FLIP_SIDE_IDS → {cluster_to_side}")

    for seg in turns:
        seg["side"] = cluster_to_side.get(seg["speaker"])

    # 3. Merge + pad. Keep whole turns intact.
    y, sr = sf.read(str(full_audio), always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32, copy=False)
    duration = len(y) / sr

    merged = _merge_same_side(turns)
    padded = _pad_turns(merged, duration)

    # 4. Build masks, write per-side WAVs, mux.
    human_pos = human_position if human_position in ("left", "right") else "right"
    agent_pos = "left" if human_pos == "right" else "right"

    outputs: dict[str, dict] = {}
    for side in ("left", "right"):
        mask = _build_mask(padded, side, len(y), sr)
        out_wav = AUDIO_DIR / f"{video_id}_{side}.wav"
        sf.write(str(out_wav), (y * mask).astype(np.float32), sr, subtype="PCM_16")

        split_vid = SPLIT_VIDEO_DIR / side / f"{video_id}.mp4"
        combined = out_dir / f"{side}_combined.mp4"
        video_path = None
        if split_vid.exists():
            if combined.exists():
                combined.unlink()
            if _mux_video_with_audio(split_vid, out_wav, combined) is not None:
                video_path = str(combined)
        outputs[side] = {"audio": str(out_wav), "video": video_path}

    # 5. speaker_map.json — trivial from CSV.
    with open(out_dir / "speaker_map.json", "w") as f:
        json.dump(
            {
                "speaker_map": {
                    "left": "HUMAN" if human_pos == "left" else "AGENT",
                    "right": "HUMAN" if human_pos == "right" else "AGENT",
                },
                "human_position": human_pos,
                "agent_position": agent_pos,
                "mapping_method": "pyannote_diarization + mediapipe_side_match",
            },
            f,
            indent=2,
        )

    # 6. Diagnostic timeline.
    timeline = {
        "video_id": video_id,
        "method": "pyannote-3.1 + MediaPipe cluster→side + whole-turn muting",
        "human_position": human_pos,
        "agent_position": agent_pos,
        "cluster_to_side": cluster_to_side,
        "side_flipped_by_override": side_flipped,
        "turns_raw": turns,
        "turns_merged_padded": padded,
        "outputs": outputs,
    }
    with open(marker, "w") as f:
        json.dump(timeline, f, indent=2)

    return timeline


def create_all_speaker_clips(df: pd.DataFrame):
    ensure_dirs()
    # Warm the pyannote model once so each per-video call is fast.
    try:
        _get_pyannote_pipeline()
    except Exception as e:
        print(f"ERROR: pyannote init failed. {e}")
        return

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Selective audio (pyannote+MP)"
    ):
        vid = row["final_conversation_id"]
        human_pos = row.get("human_position", "right")
        if pd.isna(human_pos):
            human_pos = "right"
        try:
            build_selective_for_video(vid, human_position=str(human_pos))
        except Exception as e:
            print(f"  Selective audio failed for {vid}: {e}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    create_all_speaker_clips(df)
