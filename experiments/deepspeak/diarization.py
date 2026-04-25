"""Per-side Whisper transcription for split-screen two-speaker videos.

This module replaces the old diarization step. After ``speaker_clips.py``
produces per-side masked audio — each track contains only one person's
speech, silenced elsewhere via audio-visual active speaker detection —
we transcribe each track independently with faster-whisper.

Because the mask is spatial (left channel = left person, right channel
= right person) and the CSV's ``human_position`` column hard-codes which
side is the human, speaker identity and role are both known a priori.
No diarization / clustering is involved.

Output JSON schema is kept compatible with the old diarization JSON so
downstream analysis / detection modules don't need changes:

    {
        "video_id": ...,
        "human_position": "left"|"right",
        "agent_position": "left"|"right",
        "aligned_transcript": [                     # chronological
            {"start": ..., "end": ..., "text": ...,
             "speaker": "HUMAN"|"AGENT",
             "side": "left"|"right",
             "words": [...]},
            ...
        ],
        "diarization_segments": [                   # merged runs
            {"start": ..., "end": ..., "speaker": ...}, ...
        ],
        "speaker_texts": {"HUMAN": "...", "AGENT": "..."},
        "full_text": "..."
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from deepspeak.config import (
    AUDIO_DIR,
    CSV_PATH,
    DIARIZATION_DIR,
    SPEAKER_CLIP_DIR,
    WHISPER_MODEL,
    ensure_dirs,
)


# Load the faster-whisper model only once per process.
_WHISPER: dict = {}


def _get_whisper():
    from faster_whisper import WhisperModel

    if "model" not in _WHISPER:
        _WHISPER["model"] = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )
    return _WHISPER["model"]


def transcribe_side(
    audio_path: Path,
    speaker: str,
    side: str,
) -> list[dict]:
    """Transcribe one side's masked audio. Returns labeled segments.

    VAD filtering is enabled so the long silent stretches in the masked
    track (where the other person was speaking) are skipped cheaply and
    don't produce hallucinated tokens.
    """
    if not audio_path.exists():
        return []
    model = _get_whisper()
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="en",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )

    out: list[dict] = []
    for seg in segments:
        words: list[dict] = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": round(float(w.probability), 4),
                })
        out.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "speaker": speaker,
            "side": side,
            "words": words,
        })
    return out


def _merge_same_speaker(
    aligned: list[dict], max_gap: float = 1.0
) -> list[dict]:
    """Collapse consecutive same-speaker segments into diarization runs."""
    if not aligned:
        return []
    aligned = sorted(aligned, key=lambda s: s["start"])
    out: list[dict] = []
    cur = None
    for seg in aligned:
        if cur is None:
            cur = {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]}
            continue
        if seg["speaker"] == cur["speaker"] and seg["start"] - cur["end"] <= max_gap:
            cur["end"] = max(cur["end"], seg["end"])
        else:
            out.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"]}
    if cur is not None:
        out.append(cur)
    return out


def _roles_from_csv_position(human_position: str) -> tuple[str, str, dict[str, str]]:
    """Return (human_pos, agent_pos, {side: role})."""
    human_pos = human_position if human_position in ("left", "right") else "right"
    agent_pos = "left" if human_pos == "right" else "right"
    return human_pos, agent_pos, {human_pos: "HUMAN", agent_pos: "AGENT"}


def transcribe_video(video_id: str, human_position: str) -> dict | None:
    """Transcribe both per-side WAVs for one video and write the JSON."""
    ensure_dirs()
    out_path = DIARIZATION_DIR / f"{video_id}.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    human_pos, agent_pos, role = _roles_from_csv_position(human_position)

    left_wav = AUDIO_DIR / f"{video_id}_left.wav"
    right_wav = AUDIO_DIR / f"{video_id}_right.wav"
    if not (left_wav.exists() and right_wav.exists()):
        print(f"  [{video_id}] per-side audio missing; run --step clips first")
        return None

    left_segs = transcribe_side(left_wav, role["left"], "left")
    right_segs = transcribe_side(right_wav, role["right"], "right")

    aligned = sorted(left_segs + right_segs, key=lambda s: s["start"])
    speaker_texts = {
        "HUMAN": " ".join(s["text"] for s in aligned if s["speaker"] == "HUMAN"),
        "AGENT": " ".join(s["text"] for s in aligned if s["speaker"] == "AGENT"),
    }

    result = {
        "video_id": video_id,
        "human_position": human_pos,
        "agent_position": agent_pos,
        "raw_transcript": aligned,               # back-compat key
        "diarization_segments": _merge_same_speaker(aligned),
        "aligned_transcript": aligned,
        "speaker_texts": speaker_texts,
        "full_text": " ".join(s["text"] for s in aligned),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # speaker_map.json — used by analysis.py, detect_text.py, etc.
    map_dir = SPEAKER_CLIP_DIR / video_id
    map_dir.mkdir(parents=True, exist_ok=True)
    with open(map_dir / "speaker_map.json", "w") as f:
        json.dump(
            {
                "speaker_map": {"left": role["left"], "right": role["right"]},
                "human_position": human_pos,
                "agent_position": agent_pos,
                "mapping_method": "csv_position",
            },
            f,
            indent=2,
        )

    return result


def transcribe_all(df: pd.DataFrame) -> None:
    ensure_dirs()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing sides"):
        vid = row["final_conversation_id"]
        human_pos = row.get("human_position", "right")
        if pd.isna(human_pos):
            human_pos = "right"
        try:
            transcribe_video(vid, str(human_pos))
        except Exception as e:
            print(f"  Transcribe failed for {vid}: {e}")


# Back-compat aliases (run_pipeline.py previously imported these names).
process_all_diarization = transcribe_all
process_diarization = transcribe_video


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    transcribe_all(df)
