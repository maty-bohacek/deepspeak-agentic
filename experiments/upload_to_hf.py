#!/usr/bin/env python3
"""Upload pipeline outputs to a HuggingFace dataset repo.

Authenticate first with::

    hf auth login

Repo layout produced on the Hub::

    ds3-annotations.csv            # annotations / metadata CSV
    videos/<id>.mp4                # original, full, unedited raw video
    <id>/left_video.mp4            # split-screen left half (re-encoded)
    <id>/left_audio.wav            # left-speaker-only masked audio
    <id>/right_video.mp4           # split-screen right half (re-encoded)
    <id>/right_audio.wav           # right-speaker-only masked audio

Source files on disk (produced by ``run_pipeline.py``)::

    data/ds3-annotations.csv              -> ds3-annotations.csv
    data/raw_videos/<id>.mp4              -> videos/<id>.mp4
    data/split_videos/left/<id>.mp4       -> <id>/left_video.mp4
    data/audio/<id>_left.wav              -> <id>/left_audio.wav
    data/split_videos/right/<id>.mp4      -> <id>/right_video.mp4
    data/audio/<id>_right.wav             -> <id>/right_audio.wav

Usage::

    python upload_to_hf.py --repo-id <user-or-org>/<dataset-name>
    python upload_to_hf.py --repo-id ... --limit 10            # subset
    python upload_to_hf.py --repo-id ... --dry-run             # plan only
    python upload_to_hf.py --repo-id ... --private             # create private
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from deepspeak.config import (
    AUDIO_DIR,
    CSV_PATH,
    RAW_VIDEO_DIR,
    SPLIT_VIDEO_DIR,
)


# ── Per-video file layout ─────────────────────────────────────────────

def _per_video_sources(video_id: str) -> dict[str, Path]:
    """Map staged repo-relative path -> source path on disk."""
    return {
        f"videos/{video_id}.mp4":       RAW_VIDEO_DIR / f"{video_id}.mp4",
        f"{video_id}/left_video.mp4":   SPLIT_VIDEO_DIR / "left" / f"{video_id}.mp4",
        f"{video_id}/left_audio.wav":   AUDIO_DIR / f"{video_id}_left.wav",
        f"{video_id}/right_video.mp4":  SPLIT_VIDEO_DIR / "right" / f"{video_id}.mp4",
        f"{video_id}/right_audio.wav":  AUDIO_DIR / f"{video_id}_right.wav",
    }


def _discover_video_ids() -> list[str]:
    """Every <id> for which we have a raw video on disk."""
    if not RAW_VIDEO_DIR.exists():
        return []
    return sorted(p.stem for p in RAW_VIDEO_DIR.glob("*.mp4") if p.is_file())


def _check_completeness(video_id: str) -> tuple[bool, list[str]]:
    """Return (complete, missing_relative_paths)."""
    missing = [rel for rel, src in _per_video_sources(video_id).items() if not src.exists()]
    return (not missing, missing)


# ── Staging via symlinks ──────────────────────────────────────────────

def _stage_with_symlinks(
    video_ids: list[str], stage_dir: Path, include_csv: bool
) -> tuple[int, list[tuple[str, list[str]]], bool]:
    """Materialise the upload tree under ``stage_dir`` using symlinks.

    Returns ``(num_videos_staged, [(video_id, missing_paths), ...],
    csv_staged)``. Skipped entries cover videos missing one or more
    required files.
    """
    skipped: list[tuple[str, list[str]]] = []
    staged = 0
    for vid in video_ids:
        complete, missing = _check_completeness(vid)
        if not complete:
            skipped.append((vid, missing))
            continue

        for rel, src in _per_video_sources(vid).items():
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Absolute symlink so it resolves regardless of the staging root.
            os.symlink(src.resolve(), dst)
        staged += 1

    csv_staged = False
    if include_csv and CSV_PATH.exists():
        os.symlink(CSV_PATH.resolve(), stage_dir / CSV_PATH.name)
        csv_staged = True
    return staged, skipped, csv_staged


# ── Upload ────────────────────────────────────────────────────────────

def _upload(stage_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    # ``upload_large_folder`` is designed for big datasets: it shards work
    # across processes, resumes on retry, and follows symlinks by default.
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage_dir),
        # Belt-and-braces: only ship the extensions we actually stage.
        allow_patterns=["*.mp4", "*.wav", "*.csv"],
    )


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--repo-id", required=True,
        help="Target dataset repo, e.g. 'username/deepspeak-agentic'.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only upload the first N videos (sorted by id).",
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        help="Only upload these specific video ids.",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create the repo as private if it doesn't exist yet.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be uploaded and exit (does not touch the Hub).",
    )
    parser.add_argument(
        "--stage-dir", type=Path, default=None,
        help="Where to materialise the symlink tree. Default: a tempdir that "
             "is cleaned up on exit.",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip uploading the ds3-annotations.csv file.",
    )
    args = parser.parse_args()

    # 1. Pick the set of video ids to consider.
    if args.ids:
        video_ids = sorted(set(args.ids))
    else:
        video_ids = _discover_video_ids()
    if not video_ids:
        print(f"ERROR: no raw videos found under {RAW_VIDEO_DIR}.")
        return 1
    if args.limit:
        video_ids = video_ids[: args.limit]

    print(f"Considering {len(video_ids)} video id(s).")

    # 2. Stage. Use a tempdir unless the user pinned a path.
    stage_ctx: tempfile.TemporaryDirectory | None = None
    if args.stage_dir:
        stage_dir = args.stage_dir
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True)
    else:
        stage_ctx = tempfile.TemporaryDirectory(prefix="deepspeak-hf-stage-")
        stage_dir = Path(stage_ctx.name)

    try:
        include_csv = not args.no_csv
        staged, skipped, csv_staged = _stage_with_symlinks(
            video_ids, stage_dir, include_csv=include_csv
        )

        if include_csv and not csv_staged:
            print(f"WARNING: CSV not found at {CSV_PATH}; continuing without it.")
        elif csv_staged:
            print(f"Staged CSV: {CSV_PATH.name}")

        if skipped:
            print(f"\nSkipped {len(skipped)} incomplete video(s):")
            for vid, missing in skipped[:20]:
                print(f"  - {vid}: missing {len(missing)} file(s)")
                for m in missing:
                    print(f"      {m}")
            if len(skipped) > 20:
                print(f"  ... and {len(skipped) - 20} more")

        if staged == 0:
            print("\nNothing to upload (no complete videos).")
            return 1

        print(f"\nStaged {staged} video(s) into {stage_dir}")

        if args.dry_run:
            # Show a small sample of the layout so the user can sanity-check.
            sample = sorted(p.relative_to(stage_dir) for p in stage_dir.rglob("*") if p.is_file() or p.is_symlink())[:10]
            print("Sample of staged paths:")
            for s in sample:
                print(f"  {s}")
            print("\n--dry-run: not uploading.")
            return 0

        print(f"\nUploading to dataset '{args.repo_id}' (private={args.private})...")
        _upload(stage_dir, args.repo_id, args.private)
        print("Upload complete.")
        return 0
    finally:
        if stage_ctx is not None:
            stage_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
