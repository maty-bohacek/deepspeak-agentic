#!/usr/bin/env python3
"""Emit the HeyGen x Tavus detection accuracy LaTeX table.

Reads the raw per-sample predictions from
``output/detection/{text,audio,video}_raw_predictions.json`` and re-computes
AUROC / EER / Acc / F1 per engine, resolving the engine for each video from
the CSV. If the CSV has no ``engine`` column, engine is inferred from the
prefix of ``recording_llm`` (e.g. ``tavus-llama-4`` -> ``tavus``).

Usage:
    python make_detection_table.py
    python make_detection_table.py > detection_table.tex
"""

import json
from pathlib import Path

import pandas as pd

from deepspeak.config import CSV_PATH, DETECTION_DIR
from deepspeak.detection_metrics import compute_all_metrics


ROWS = [
    ("text",  "Binoculars",    "text: Binoculars"),
    ("text",  "Desklib",       "text: Desklib"),
    ("text",  "DivEye",        "text: DivEye"),
    ("audio", "wav2vec-xlsr",  "audio: wav2vec-xlsr"),
    ("audio", "AASIST3",       "audio: AASIST3"),
    ("audio", "DF_Arena_500M", "audio: DF-Arena-500M"),
    ("audio", "DF_Arena_1B",   "audio: DF-Arena-1B"),
    ("video", "GenConViT-ED",  "video: GenConViT-ED"),
    ("video", "GenConViT-VAE", "video: GenConViT-VAE"),
    ("video", "CLIP-GDD",      "video: CLIP-GDD"),
    ("video", "GenD-CLIP-L14", "video: GenD-CLIP-L14"),
]

METRIC_KEYS = ("auroc", "eer", "accuracy", "f1_macro")
ENGINES = ("heygen", "tavus")


def _infer_engine(row: pd.Series) -> str | None:
    """Resolve engine, preferring an explicit column and falling back to
    the ``recording_llm`` prefix (e.g. ``tavus-llama-4`` -> ``tavus``)."""
    eng = row.get("engine")
    if isinstance(eng, str):
        eng = eng.strip().lower()
        if eng and eng != "nan":
            return eng
    llm = row.get("recording_llm")
    if isinstance(llm, str):
        token = llm.split("-", 1)[0].strip().lower()
        if token and token != "nan":
            return token
    return None


def _build_engine_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names ("engine " -> "engine") and string
    # values so prefix matching is stable.
    df.columns = df.columns.str.strip()
    for col in ("engine", "recording_llm", "final_conversation_id"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        vid = row.get("final_conversation_id")
        if not isinstance(vid, str) or vid in ("", "nan"):
            continue
        eng = _infer_engine(row)
        if eng is not None:
            mapping[vid] = eng
    return mapping


def _load_raw(modality: str) -> dict:
    path = DETECTION_DIR / f"{modality}_raw_predictions.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "---"
    if f != f:  # NaN
        return "---"
    return f"{f:.2f}"


def _cells_for(preds: list[dict], id_to_engine: dict[str, str], wanted_engine: str) -> list[str]:
    kept = []
    for p in preds:
        vid = p.get("video_id")
        eng = (p.get("engine") or id_to_engine.get(vid) or "").lower()
        if wanted_engine in eng:
            kept.append(p)
    if not kept:
        return ["---"] * len(METRIC_KEYS)
    y_true = [p["is_fake"] for p in kept]
    y_scores = [p["score"] for p in kept]
    if len(set(y_true)) < 2:
        return ["---"] * len(METRIC_KEYS)
    m = compute_all_metrics(y_true, y_scores)
    return [_fmt(m.get(k)) for k in METRIC_KEYS]


def _row(modality: str, det_key: str, label: str, raw: dict, id_to_engine: dict) -> str:
    preds = raw.get(modality, {}).get(det_key, [])
    cells = _cells_for(preds, id_to_engine, "heygen") + _cells_for(preds, id_to_engine, "tavus")
    return f"    {label:<22} & " + " & ".join(cells) + r" \\"


def build_table() -> str:
    id_to_engine = _build_engine_map(CSV_PATH)
    raw = {m: _load_raw(m) for m in ("text", "audio", "video")}

    groups: dict[str, list[str]] = {}
    for modality, det_key, label in ROWS:
        groups.setdefault(modality, []).append(
            _row(modality, det_key, label, raw, id_to_engine)
        )

    body: list[str] = []
    for i, modality in enumerate(("text", "audio", "video")):
        if i > 0:
            body.append(r"    \midrule")
        body.extend(groups.get(modality, []))

    preamble = r"""\begin{table*}[t]
  \centering
  \caption{Text, audio, and video forensic detection accuracy for HeyGen and Tavus platforms, averaged over different LLMs and voice generators.}
  \label{tab:text_detection}
  \small
  \begin{tabular}{lcccccccc}
    \toprule
    & \multicolumn{4}{c}{\textbf{HeyGen}} & \multicolumn{4}{c}{\textbf{Tavus}} \\
    \cmidrule(lr){2-5} \cmidrule(lr){6-9}
    Detector & AUROC\,$\uparrow$ & EER\,$\downarrow$ & Acc.\,$\uparrow$ & F1\,$\uparrow$
             & AUROC\,$\uparrow$ & EER\,$\downarrow$ & Acc.\,$\uparrow$ & F1\,$\uparrow$ \\
    \midrule"""

    postamble = r"""    \bottomrule
  \end{tabular}
\end{table*}"""

    return "\n".join([preamble, *body, postamble])


if __name__ == "__main__":
    print(build_table())
