#!/usr/bin/env python3
"""Rename video IDs in an already-uploaded HuggingFace dataset.

Replaces every video ID with a random 12-character alphanumeric string.
Applied consistently across:

  * ``videos/<old>.mp4``   -> ``videos/<new>.mp4``
  * ``<old>/*``            -> ``<new>/*``
  * ``ds3-annotations.csv``: the ``final_conversation_id`` column is
    rewritten with the same mapping.

The mp4/wav files are moved via ``CommitOperationCopy`` + ``Delete``,
which for LFS-tracked files is a pointer update on the Hub — no re-upload
of the underlying blobs. Everything goes into a single atomic commit so
partial failures can't corrupt the dataset.

A local ``id_mapping.json`` is written so you keep a record of old->new.

Auth: ``hf auth login`` first.

Usage::

    python rename_hf_ids.py --repo-id <user>/<dataset>
    python rename_hf_ids.py --repo-id ... --dry-run
    python rename_hf_ids.py --repo-id ... --seed 42        # reproducible
"""

from __future__ import annotations

import argparse
import io
import json
import random
import string
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)


CSV_NAME = "ds3-annotations.csv"
ID_COLUMN = "final_conversation_id"
NEW_ID_LEN = 12
ALPHABET = string.ascii_letters + string.digits


def _discover_ids(files: list[str]) -> tuple[set[str], set[str]]:
    """Split the file listing into (video_file_ids, per_id_folder_ids)."""
    video_ids: set[str] = set()
    folder_ids: set[str] = set()
    for f in files:
        if f.startswith("videos/") and f.endswith(".mp4"):
            tail = f[len("videos/"):-len(".mp4")]
            if "/" not in tail:
                video_ids.add(tail)
        elif "/" in f and not f.startswith("videos/") and not f.startswith("."):
            folder_ids.add(f.split("/", 1)[0])
    return video_ids, folder_ids


def _generate_mapping(old_ids: set[str], rng: random.Random) -> dict[str, str]:
    """Assign each old id a fresh 12-char alphanumeric id.

    Guarantees new ids don't collide with other new ids or with any
    existing old id — otherwise a copy-then-delete step could wipe out a
    file we'd just renamed.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set(old_ids)
    for old in sorted(old_ids):
        while True:
            new = "".join(rng.choices(ALPHABET, k=NEW_ID_LEN))
            if new not in used:
                break
        mapping[old] = new
        used.add(new)
    return mapping


def _rename_path(path: str, mapping: dict[str, str]) -> str | None:
    """Return the new repo path for ``path``, or None if it isn't remapped."""
    if path.startswith("videos/") and path.endswith(".mp4"):
        tail = path[len("videos/"):-len(".mp4")]
        if "/" not in tail and tail in mapping:
            return f"videos/{mapping[tail]}.mp4"
        return None
    if "/" in path and not path.startswith("videos/") and not path.startswith("."):
        head, rest = path.split("/", 1)
        if head in mapping:
            return f"{mapping[head]}/{rest}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo-id", required=True, help="Dataset repo, e.g. 'user/ds'.")
    p.add_argument("--dry-run", action="store_true",
                   help="Plan and write mapping file, but don't commit to the Hub.")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed the RNG for reproducibility. Default: non-deterministic.")
    p.add_argument("--mapping-out", type=Path, default=Path("id_mapping.json"),
                   help="Where to write the old->new id map. Default: ./id_mapping.json")
    p.add_argument("--commit-message", type=str, default=None,
                   help="Override the auto-generated commit message.")
    args = p.parse_args()

    api = HfApi()

    # 1. Enumerate the repo and discover IDs.
    print(f"Listing files in '{args.repo_id}'...")
    files = api.list_repo_files(args.repo_id, repo_type="dataset")

    video_ids, folder_ids = _discover_ids(files)
    all_ids = video_ids | folder_ids
    if not all_ids:
        print("ERROR: no video IDs found on the repo.")
        return 1

    both = video_ids & folder_ids
    only_video = video_ids - folder_ids
    only_folder = folder_ids - video_ids
    print(f"  {len(both)} id(s) with both videos/<id>.mp4 and <id>/ folder")
    if only_video:
        print(f"  {len(only_video)} id(s) with only a videos/<id>.mp4 entry")
    if only_folder:
        print(f"  {len(only_folder)} id(s) with only a per-id folder")

    # 2. Generate the rename mapping.
    rng = random.Random(args.seed) if args.seed is not None else random.SystemRandom()
    mapping = _generate_mapping(all_ids, rng)

    args.mapping_out.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"\nWrote mapping ({len(mapping)} entries) -> {args.mapping_out}")
    for old, new in list(mapping.items())[:5]:
        print(f"  {old} -> {new}")
    if len(mapping) > 5:
        print(f"  ... and {len(mapping) - 5} more")

    # 3. Build copy + delete ops for every remappable file.
    ops: list = []
    for f in files:
        new_path = _rename_path(f, mapping)
        if new_path and new_path != f:
            ops.append(CommitOperationCopy(src_path_in_repo=f, path_in_repo=new_path))
            ops.append(CommitOperationDelete(path_in_repo=f))

    n_moves = sum(1 for o in ops if isinstance(o, CommitOperationCopy))

    # 4. Rewrite the CSV's id column with the same mapping.
    csv_updated = False
    if CSV_NAME in files:
        print(f"\nDownloading {CSV_NAME} to rewrite '{ID_COLUMN}'...")
        csv_local = hf_hub_download(args.repo_id, CSV_NAME, repo_type="dataset")
        df = pd.read_csv(csv_local)
        if ID_COLUMN not in df.columns:
            print(f"ERROR: CSV is missing column '{ID_COLUMN}'.")
            return 1
        before = df[ID_COLUMN].astype(str)
        df[ID_COLUMN] = before.map(lambda v: mapping.get(v, v))
        n_rows_changed = int((before != df[ID_COLUMN]).sum())
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        ops.append(CommitOperationAdd(path_in_repo=CSV_NAME, path_or_fileobj=buf))
        csv_updated = True
        print(f"  {n_rows_changed}/{len(df)} CSV rows will be updated.")
    else:
        print(f"\nNote: {CSV_NAME} not present on the repo; skipping CSV rewrite.")

    # 5. Summarise and maybe commit.
    print(
        f"\nPrepared {len(ops)} operations: "
        f"{n_moves} file move(s){' + 1 csv update' if csv_updated else ''}."
    )

    if args.dry_run:
        print("--dry-run: not committing. Mapping is saved locally.")
        return 0

    if not ops:
        print("Nothing to do.")
        return 0

    msg = args.commit_message or f"Anonymize video IDs ({len(mapping)} renamed)"
    print(f"\nCommitting to '{args.repo_id}' as one atomic change...")
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=msg,
    )
    print("Done. Keep id_mapping.json if you need to reconcile with local state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
