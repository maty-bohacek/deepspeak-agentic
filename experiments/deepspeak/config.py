"""Central configuration for the DeepSpeak Agentic toolkit."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"

RAW_VIDEO_DIR = DATA_DIR / "raw_videos"
TRIMMED_VIDEO_DIR = DATA_DIR / "trimmed_videos"
SPLIT_VIDEO_DIR = DATA_DIR / "split_videos"       # left/ and right/ subdirs
SPEAKER_CLIP_DIR = DATA_DIR / "speaker_clips"      # clips per speaker while talking
AUDIO_DIR = DATA_DIR / "audio"                     # extracted audio tracks
DIARIZATION_DIR = DATA_DIR / "diarization"         # diarization + transcript JSONs
LLM_ANALYSIS_DIR = DATA_DIR / "llm_analysis"       # keyword/summary JSONs

DETECTION_DIR = OUTPUT_DIR / "detection"
STATS_DIR = OUTPUT_DIR / "stats"
PLOTS_DIR = OUTPUT_DIR / "plots"

CSV_PATH = DATA_DIR / "ds3-annotations.csv"

# ── HuggingFace ────────────────────────────────────────────────────────
HF_DATASET_ID = "faridlab/deepspeak_agentic"
# Set HF_TOKEN in your environment (e.g. via `hf auth login` or `export HF_TOKEN=...`).
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
HF_CSV_URL = (
    "https://huggingface.co/datasets/faridlab/deepspeak_agentic"
    "/resolve/main/ds3-annotations.csv"
)
HF_VIDEO_URL_TEMPLATE = (
    "https://huggingface.co/datasets/faridlab/deepspeak_agentic"
    "/resolve/main/data/{video_id}.mp4"
)

# ── OpenAI ─────────────────────────────────────────────────────────────
# Set OPENAI_API_KEY in your environment.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ── Processing ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # for audio processing
WHISPER_MODEL = "base"       # faster-whisper model size
MAX_WORKERS = 4              # parallel downloads / processing

# ── Manual overrides ──────────────────────────────────────────────────
# Video IDs for which the automatic left/right audio-track assignment
# (pyannote clusters → side via MediaPipe lip motion) should be INVERTED.
# Add an ID here when you've confirmed the clips step produced a flipped
# mapping. Delete the corresponding ``data/speaker_clips/<id>/`` folder
# (and optionally ``data/audio/<id>_left.wav`` / ``_right.wav``) and
# re-run `python run_pipeline.py --step clips`.
FLIP_SIDE_IDS: set[str] = {
    "c5a21df7bcc1c4f8",
    "c6dec5108c03d4d1",
    "c8e369208a5a84c1",
    "c9c3ca57aaf4046f",
    "c62c7cdde8a0f4bf",
    "c301af20ccdea442",
    "c3779cb9ce8e740d",
    "c266309edfaa5455",
    "c114328695e55412",
    "cba05dd19a2ee498",
    "ce0c73336e9f6494",
    "ce6c3b7d30de7436",
    "ce9a3e79baf51426",
    "ce58cc0d44a7643a",
    "ce867503f00174d0",
    "cebcaff377efb400",
    "cf732262ef4234b0"
}


def ensure_dirs():
    """Create all output directories."""
    for d in [
        RAW_VIDEO_DIR, TRIMMED_VIDEO_DIR, SPLIT_VIDEO_DIR,
        SPLIT_VIDEO_DIR / "left", SPLIT_VIDEO_DIR / "right",
        SPEAKER_CLIP_DIR, AUDIO_DIR, DIARIZATION_DIR,
        LLM_ANALYSIS_DIR, DETECTION_DIR, STATS_DIR, PLOTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def ensure_engine_column(df):
    """Ensure ``df`` has an ``engine`` column (``tavus`` / ``heygen``).

    The annotations CSV lacks an explicit engine field — the deepfake engine
    is encoded in the ``recording_llm`` prefix (e.g. ``tavus-llama-4`` ->
    ``tavus``, ``heygen-gpt-4o`` -> ``heygen``). This derives the column
    in-place when it's missing or all-null so every downstream step
    (analysis, LLM, detection) can group by engine.
    """
    import pandas as pd

    df.columns = df.columns.str.strip()
    if "engine" in df.columns:
        if df["engine"].dtype == object:
            df["engine"] = df["engine"].astype(str).str.strip()
        if df["engine"].replace({"": None, "nan": None}).notna().any():
            return df

    def _infer(row):
        llm = row.get("recording_llm")
        if isinstance(llm, str):
            token = llm.split("-", 1)[0].strip().lower()
            if token and token != "nan":
                return token
        return None

    df["engine"] = df.apply(_infer, axis=1)
    return df
