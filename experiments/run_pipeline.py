#!/usr/bin/env python3
"""Main pipeline for the DeepSpeak Agentic toolkit.

Pipeline order:
    download  →  process  →  clips  →  transcribe  →  detect / analyze / llm

Usage:
    python run_pipeline.py                     # Run everything, in order
    python run_pipeline.py --step download
    python run_pipeline.py --step process      # Trim + split + extract full audio
    python run_pipeline.py --step clips        # Per-side audio-visual ASD muxing
                                               # (MediaPipe lip motion + VAD; emits
                                               #  full-duration videos where only the
                                               #  shown person's speech is audible)
    python run_pipeline.py --step transcribe   # Whisper on each per-side masked WAV
                                               # (HUMAN/AGENT from CSV human_position;
                                               #  no diarization / clustering)
    python run_pipeline.py --step diarize      # Alias for `transcribe` (back-compat)
    python run_pipeline.py --step detect       # All detection
    python run_pipeline.py --step detect-video
    python run_pipeline.py --step detect-audio
    python run_pipeline.py --step detect-text
    python run_pipeline.py --step analyze      # Dataset analysis
    python run_pipeline.py --step llm          # LLM-based analysis
    python run_pipeline.py --step summary      # Print summary of existing results
    python run_pipeline.py --limit 5           # Process only N videos
"""

import argparse
import sys
import pandas as pd

from deepspeak.config import CSV_PATH, ensure_dirs, ensure_engine_column


def main():
    parser = argparse.ArgumentParser(description="DeepSpeak Agentic Toolkit")
    parser.add_argument(
        "--step", type=str, default="all",
        choices=[
            "all", "download", "process", "clips",
            "transcribe", "diarize",
            "detect", "detect-video", "detect-audio", "detect-text",
            "analyze", "llm", "summary",
        ],
        help="Pipeline step to run",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos")
    args = parser.parse_args()

    ensure_dirs()
    step = args.step

    # ── Summary-only: doesn't need CSV or dataframe ───────────────────
    if step == "summary":
        from deepspeak.summary import run_summary
        run_summary()
        return

    # ── Download ───────────────────────────────────────────────────────
    if step in ("all", "download"):
        print("\n" + "=" * 60)
        print("STEP 1: DOWNLOAD")
        print("=" * 60)
        from deepspeak.download import download_csv, download_videos
        df = download_csv()
        download_videos(df, limit=args.limit)

    # Load CSV for subsequent steps
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        ensure_engine_column(df)
        if args.limit:
            df = df.head(args.limit)
    else:
        print("ERROR: CSV not found. Run --step download first.")
        sys.exit(1)

    # ── Video Processing ───────────────────────────────────────────────
    if step in ("all", "process"):
        print("\n" + "=" * 60)
        print("STEP 2: VIDEO PROCESSING (trim / split / extract audio)")
        print("=" * 60)
        from deepspeak.video_processing import process_all_videos
        process_all_videos(df)

    # ── Per-side selective-audio muxing ────────────────────────────────
    if step in ("all", "clips"):
        print("\n" + "=" * 60)
        print("STEP 3: PER-SIDE SELECTIVE-AUDIO MUXING (AV-ASD)")
        print("=" * 60)
        from deepspeak.speaker_clips import create_all_speaker_clips
        create_all_speaker_clips(df)

    # ── Transcription on per-side masked audio ────────────────────────
    if step in ("all", "transcribe", "diarize"):
        print("\n" + "=" * 60)
        print("STEP 4: PER-SIDE WHISPER TRANSCRIPTION")
        print("=" * 60)
        from deepspeak.diarization import transcribe_all
        transcribe_all(df)

    # ── Detection ──────────────────────────────────────────────────────
    if step in ("all", "detect", "detect-video"):
        print("\n" + "=" * 60)
        print("STEP 5a: VIDEO DEEPFAKE DETECTION")
        print("=" * 60)
        from deepspeak.detect_video import run_video_detection
        run_video_detection(df)

    if step in ("all", "detect", "detect-audio"):
        print("\n" + "=" * 60)
        print("STEP 5b: AUDIO DEEPFAKE DETECTION")
        print("=" * 60)
        from deepspeak.detect_audio import run_audio_detection
        run_audio_detection(df)

    if step in ("all", "detect", "detect-text"):
        print("\n" + "=" * 60)
        print("STEP 5c: TEXT AI DETECTION")
        print("=" * 60)
        from deepspeak.detect_text import run_text_detection
        run_text_detection(df)

    # ── Analysis ───────────────────────────────────────────────────────
    if step in ("all", "analyze"):
        print("\n" + "=" * 60)
        print("STEP 6: DATASET ANALYSIS")
        print("=" * 60)
        from deepspeak.analysis import run_analysis
        run_analysis(df)

    # ── LLM Analysis ──────────────────────────────────────────────────
    if step in ("all", "llm"):
        print("\n" + "=" * 60)
        print("STEP 7: LLM-BASED ANALYSIS")
        print("=" * 60)
        from deepspeak.llm_analysis import run_llm_analysis
        run_llm_analysis(df)

    # ── Summary ───────────────────────────────────────────────────────
    if step == "all":
        print("\n" + "=" * 60)
        print("STEP 8: RESULTS SUMMARY")
        print("=" * 60)
        from deepspeak.summary import run_summary
        run_summary()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
