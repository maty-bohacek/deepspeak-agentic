"""Download dataset CSV and video files from HuggingFace."""

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from deepspeak.config import (
    HF_TOKEN, HF_CSV_URL, HF_VIDEO_URL_TEMPLATE,
    CSV_PATH, RAW_VIDEO_DIR, DATA_DIR, MAX_WORKERS, ensure_dirs,
)


def download_file(url: str, dest: Path, token: str | None = None, force: bool = False) -> bool:
    """Download a single file with retry logic.

    Retries only make sense for transient failures (network blips, 5xx,
    rate limits). A 4xx other than 429 is a permanent response from the
    server — retrying just wastes ~30s of backoff per missing file.
    """
    if not force and dest.exists() and dest.stat().st_size > 0:
        return True
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for attempt in range(4):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=120)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                print(f"  FAILED: {dest.name} ({status} {e.response.reason})")
                return False
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt+1}/4 for {dest.name}: {e} (wait {wait}s)")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt+1}/4 for {dest.name}: {e} (wait {wait}s)")
            time.sleep(wait)
    print(f"  FAILED: {dest.name}")
    return False


def download_csv() -> pd.DataFrame:
    """Download the annotations CSV and return as DataFrame.

    Always fetches the latest copy from HuggingFace so that upstream
    edits to the specifier are picked up on every pipeline run.
    """
    ensure_dirs()
    print("Downloading latest annotations CSV from HuggingFace...")
    download_file(HF_CSV_URL, CSV_PATH, HF_TOKEN, force=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} annotations from CSV")
    return df


def download_videos(df: pd.DataFrame, limit: int | None = None) -> list[str]:
    """Download video files. Returns list of successfully downloaded video IDs."""
    ensure_dirs()
    video_ids = df["final_conversation_id"].tolist()
    if limit:
        video_ids = video_ids[:limit]

    downloaded = []

    def _download_one(vid: str) -> tuple[str, bool]:
        url = HF_VIDEO_URL_TEMPLATE.format(video_id=vid)
        dest = RAW_VIDEO_DIR / f"{vid}.mp4"
        ok = download_file(url, dest, HF_TOKEN)
        return vid, ok

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_download_one, v): v for v in video_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            vid, ok = fut.result()
            if ok:
                downloaded.append(vid)

    print(f"Downloaded {len(downloaded)}/{len(video_ids)} videos")
    return downloaded


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = download_csv()
    download_videos(df, limit=limit)
