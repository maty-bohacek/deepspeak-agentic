"""Video processing: trim, split left/right, extract audio.

The split-screen halves produced here keep the original (trimmed) audio
track intact. Per-speaker audio masking is handled separately in
``speaker_clips.py`` — we no longer cut-and-restitch speaker segments.

All outputs re-encode rather than using ``-c copy`` so that the resulting
MP4 files have valid keyframes at every cut point, monotonic timestamps,
and ``+faststart`` headers. This avoids the "opens as audio only in
QuickTime" / "frame lag" failure mode seen with stream-copied cuts whose
start offsets don't land on a keyframe.
"""

import json
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from deepspeak.config import (
    RAW_VIDEO_DIR, TRIMMED_VIDEO_DIR, SPLIT_VIDEO_DIR,
    AUDIO_DIR, CSV_PATH, SAMPLE_RATE, ensure_dirs,
)

# Encoding parameters shared by the trim / split re-encodes.
_VIDEO_ENC = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p",
    "-profile:v", "high", "-level", "4.0",
]
_AUDIO_ENC = ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]
_MP4_FLAGS = ["-movflags", "+faststart"]

# Hard cap per ffmpeg invocation. Clips in this dataset are short; any
# single call taking longer is stuck on a malformed input.
FFMPEG_TIMEOUT_SEC = 600


def _run(
    cmd: list[str], timeout: float | None = FFMPEG_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command, hiding its chatter.

    ``-nostdin`` is injected for ffmpeg calls so the child never tries
    to read keystrokes from the inherited TTY. A ``timeout`` guarantees
    we can never get wedged on a single bad file.
    """
    if cmd and cmd[0] == "ffmpeg" and "-nostdin" not in cmd:
        cmd = [cmd[0], "-nostdin", *cmd[1:]]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def get_video_duration(path: Path) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(path),
    ]
    result = _run(cmd)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_video_dimensions(path: Path) -> tuple[int, int]:
    """Get video width and height via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(path),
    ]
    result = _run(cmd)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            return int(stream["width"]), int(stream["height"])
    raise ValueError(f"No video stream in {path}")


def get_video_fps(path: Path) -> float:
    """Return average FPS of the first video stream."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(path),
    ]
    result = _run(cmd)
    info = json.loads(result.stdout)
    for stream in info["streams"]:
        if stream["codec_type"] == "video":
            for key in ("avg_frame_rate", "r_frame_rate"):
                val = stream.get(key, "")
                if "/" in val:
                    num, den = val.split("/")
                    try:
                        num, den = float(num), float(den)
                        if den > 0:
                            return num / den
                    except ValueError:
                        pass
    return 30.0


def trim_video(video_id: str, start_sec: float, end_sec: float) -> Path:
    """Trim video to [start_sec, end_sec]. Returns output path.

    Re-encodes so the resulting MP4 starts with a keyframe and plays
    cleanly in any player.
    """
    src = RAW_VIDEO_DIR / f"{video_id}.mp4"
    dst = TRIMMED_VIDEO_DIR / f"{video_id}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if start_sec and start_sec > 0:
        # ``-ss`` after ``-i`` = accurate (decoded) seek, not keyframe seek.
        cmd += ["-ss", f"{start_sec}"]
    if end_sec and end_sec > 0:
        cmd += ["-to", f"{end_sec}"]
    cmd += _VIDEO_ENC + _AUDIO_ENC + _MP4_FLAGS + [str(dst)]
    try:
        result = _run(cmd)
    except subprocess.TimeoutExpired:
        _unlink(dst)
        raise RuntimeError(f"trim timed out for {video_id}")
    if result.returncode != 0:
        _unlink(dst)
        raise RuntimeError(f"trim failed for {video_id}: {result.stderr[-500:]}")
    return dst


def split_left_right(video_id: str) -> tuple[Path, Path]:
    """Split a video into left and right halves. Re-encodes both streams."""
    src = TRIMMED_VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        src = RAW_VIDEO_DIR / f"{video_id}.mp4"

    left_dst = SPLIT_VIDEO_DIR / "left" / f"{video_id}.mp4"
    right_dst = SPLIT_VIDEO_DIR / "right" / f"{video_id}.mp4"

    if left_dst.exists() and left_dst.stat().st_size > 0 \
            and right_dst.exists() and right_dst.stat().st_size > 0:
        return left_dst, right_dst

    w, h = get_video_dimensions(src)
    half_w = (w // 2) - ((w // 2) % 2)   # keep even for yuv420p
    height = h - (h % 2)

    for dst, x_off in [(left_dst, 0), (right_dst, w - half_w)]:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"crop={half_w}:{height}:{x_off}:0,setsar=1",
            *_VIDEO_ENC, *_AUDIO_ENC, *_MP4_FLAGS,
            str(dst),
        ]
        try:
            result = _run(cmd)
        except subprocess.TimeoutExpired:
            _unlink(dst)
            raise RuntimeError(f"split timed out for {video_id} ({dst.name})")
        if result.returncode != 0:
            _unlink(dst)
            raise RuntimeError(
                f"split failed for {video_id} ({dst.name}): {result.stderr[-500:]}"
            )

    return left_dst, right_dst


def extract_audio(video_id: str) -> Path:
    """Extract mono 16kHz WAV audio from video."""
    src = TRIMMED_VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        src = RAW_VIDEO_DIR / f"{video_id}.mp4"

    dst = AUDIO_DIR / f"{video_id}.wav"
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        str(dst),
    ]
    try:
        result = _run(cmd)
    except subprocess.TimeoutExpired:
        _unlink(dst)
        raise RuntimeError(f"audio extract timed out for {video_id}")
    if result.returncode != 0:
        _unlink(dst)
        raise RuntimeError(f"audio extract failed for {video_id}: {result.stderr[-500:]}")
    return dst


def process_all_videos(df: pd.DataFrame):
    """Run full video processing pipeline on all videos in the DataFrame."""
    ensure_dirs()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        vid = row["final_conversation_id"]
        raw_path = RAW_VIDEO_DIR / f"{vid}.mp4"
        if not raw_path.exists():
            continue

        # 1. Trim
        start = row.get("beginning_delay_seconds")
        end = row.get("end_delay_seconds")
        start = float(start) if pd.notna(start) else 0
        end = float(end) if pd.notna(end) else 0
        try:
            trim_video(vid, start, end)
        except Exception as e:
            print(f"  Trim failed for {vid}: {e}")
            import shutil
            shutil.copy2(raw_path, TRIMMED_VIDEO_DIR / f"{vid}.mp4")

        # 2. Split left/right
        try:
            split_left_right(vid)
        except Exception as e:
            print(f"  Split failed for {vid}: {e}")

        # 3. Extract audio
        try:
            extract_audio(vid)
        except Exception as e:
            print(f"  Audio extraction failed for {vid}: {e}")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    process_all_videos(df)
