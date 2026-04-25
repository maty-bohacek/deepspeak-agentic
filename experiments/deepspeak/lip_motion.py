"""Lip-motion signal extraction for split-screen two-speaker videos.

For every sampled frame, MediaPipe Face Mesh is run once on the full
split-screen image. Each detected face is assigned to the ``left`` or
``right`` half by the x-coordinate of its landmark centroid. For each
side we record:

* ``mar`` — Mouth Aspect Ratio = vertical lip gap / mouth width.
  This is a dimensionless, self-normalizing signal: a face 20 px tall
  and a face 200 px tall with the same mouth opening produce the same
  MAR, so camera distance, resolution, and blur that only affects
  absolute sizes drop out.

* ``detected`` — 1 if a face was found on that side in that frame,
  else 0. Downstream uses this to gate the decision: if one side never
  gets a detection (person too small / too blurry for MediaPipe), it
  contributes 0 to the speaker-score and the other side wins by default.

MAR is mostly robust to:
  * scale (both numerator and denominator scale together),
  * in-plane tilt (distances are Euclidean in the image plane),
  * slow head translation (doesn't change the ratio).

It is partially affected by head yaw/pitch, but taking the rolling
standard deviation of MAR over a short window (done in
``speaker_clips.py``) keeps the signal focused on fast articulation
and mostly ignores slow pose changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepspeak.config import (
    RAW_VIDEO_DIR,
    SPEAKER_CLIP_DIR,
    TRIMMED_VIDEO_DIR,
    ensure_dirs,
)


# MediaPipe FaceMesh landmark indices (468-point model, no iris).
_UPPER_INNER_LIP = 13   # upper inner lip midpoint
_LOWER_INNER_LIP = 14   # lower inner lip midpoint
_LEFT_MOUTH_CORNER = 61
_RIGHT_MOUTH_CORNER = 291

# Cache schema version — bump to force recompute of stale caches.
_CACHE_VERSION = 2


def _per_side_mar_series(
    video_path: Path,
    sample_fps: float = 25.0,
    max_faces: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (timestamps, left_mar, right_mar, left_detected, right_detected).

    MAR values are NaN where no face was found on that side; the detection
    arrays are {0, 1}.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (np.array([]),) * 5  # type: ignore[return-value]

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / sample_fps)), 1)

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    timestamps: list[float] = []
    l_mar: list[float] = []
    r_mar: list[float] = []
    l_det: list[int] = []
    r_det: list[int] = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                t = frame_idx / src_fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                lv = np.nan
                rv = np.nan
                ld = 0
                rd = 0
                if results.multi_face_landmarks:
                    for lm in results.multi_face_landmarks:
                        pts = lm.landmark
                        # Scale-invariant mouth aspect ratio:
                        #   MAR = | upper_inner - lower_inner | / | left_corner - right_corner |
                        up = pts[_UPPER_INNER_LIP]
                        lo = pts[_LOWER_INNER_LIP]
                        lc = pts[_LEFT_MOUTH_CORNER]
                        rc = pts[_RIGHT_MOUTH_CORNER]
                        mouth_w = float(np.hypot(rc.x - lc.x, rc.y - lc.y))
                        if mouth_w < 1e-6:
                            continue
                        lip_h = float(np.hypot(up.x - lo.x, up.y - lo.y))
                        mar = lip_h / mouth_w

                        cx = float(np.mean([p.x for p in pts]))
                        if cx < 0.5:
                            # If two faces end up on the same side (rare;
                            # close-up capture), keep the one with a more
                            # expressive mouth — that's more likely to be
                            # the active speaker anyway.
                            lv = mar if np.isnan(lv) else max(lv, mar)
                            ld = 1
                        else:
                            rv = mar if np.isnan(rv) else max(rv, mar)
                            rd = 1

                timestamps.append(t)
                l_mar.append(lv)
                r_mar.append(rv)
                l_det.append(ld)
                r_det.append(rd)
            frame_idx += 1
    finally:
        cap.release()
        face_mesh.close()

    return (
        np.asarray(timestamps, dtype=np.float32),
        np.asarray(l_mar, dtype=np.float32),
        np.asarray(r_mar, dtype=np.float32),
        np.asarray(l_det, dtype=np.uint8),
        np.asarray(r_det, dtype=np.uint8),
    )


def compute_lip_signals(
    video_id: str,
    sample_fps: float = 25.0,
    cache: bool = True,
) -> dict | None:
    """Compute and cache per-side scale-invariant lip signals for one video.

    Returns a dict with keys:
      * ``timestamps``
      * ``left_mar``, ``right_mar``  — MAR per frame (None where no face)
      * ``left_detected``, ``right_detected``  — 0/1 per frame
      * ``sample_fps``, ``version``

    The cache is invalidated automatically when either ``sample_fps`` or
    ``version`` drifts from the cached values.
    """
    ensure_dirs()

    cache_dir = SPEAKER_CLIP_DIR / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "lip_signals.json"

    if cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
        except Exception:
            cached = {}
        if (
            float(cached.get("sample_fps", 0)) == float(sample_fps)
            and int(cached.get("version", 0)) == _CACHE_VERSION
        ):
            return cached

    src = TRIMMED_VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        src = RAW_VIDEO_DIR / f"{video_id}.mp4"
    if not src.exists():
        return None

    ts, lm, rm, ld, rd = _per_side_mar_series(src, sample_fps=sample_fps)
    if ts.size == 0:
        return None

    def _nan_to_none(arr: np.ndarray) -> list:
        return [None if np.isnan(v) else float(v) for v in arr]

    result = {
        "video_id": video_id,
        "version": _CACHE_VERSION,
        "sample_fps": float(sample_fps),
        "timestamps": ts.tolist(),
        "left_mar": _nan_to_none(lm),
        "right_mar": _nan_to_none(rm),
        "left_detected": [int(v) for v in ld],
        "right_detected": [int(v) for v in rd],
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


if __name__ == "__main__":
    import sys

    vid = sys.argv[1]
    out = compute_lip_signals(vid)
    if out is None:
        print("no lip data")
    else:
        # Print a compact summary rather than dumping the full arrays.
        n = len(out["timestamps"])
        ldet = sum(out["left_detected"]) / max(1, n)
        rdet = sum(out["right_detected"]) / max(1, n)
        print(f"{vid}: {n} frames @ {out['sample_fps']} fps, "
              f"left detected {ldet:.2%}, right detected {rdet:.2%}")
