"""Human-readable summary of pipeline results.

Reads the JSON artifacts produced by the `analyze`, `detect`, and `llm`
steps and prints a compact report. Also writes ``output/stats/summary.txt``.

Usage:
    python run_pipeline.py --step summary
    python -m deepspeak.summary
"""

import json
from pathlib import Path

from deepspeak.config import (
    DETECTION_DIR, STATS_DIR, PLOTS_DIR, LLM_ANALYSIS_DIR, ensure_dirs,
)


MODALITIES = ("video", "audio", "text")


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_dist(d: dict, top: int = 5) -> str:
    if not d:
        return "—"
    items = sorted(d.items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{k}={v}" for k, v in items)


def _dataset_block_lines(
    ds: dict,
    diar_agg: dict,
    indent: str = "  ",
    include_engines_row: bool = True,
) -> list[str]:
    """Render the body of a dataset block for a given (ds, diar) pair."""
    lines: list[str] = []
    lines.append(f"{indent}Conversations:   {ds.get('total_conversations', 0)}")
    lines.append(
        f"{indent}Total duration:  {ds.get('total_duration_hours', 0)} h  "
        f"(avg {ds.get('avg_duration_seconds', 0):.1f}s, "
        f"median {ds.get('median_duration_seconds', 0):.1f}s)"
    )
    lines.append(f"{indent}Total size:      {ds.get('total_size_mb', 0):.1f} MB")
    lines.append(f"{indent}Gender:          {_fmt_dist(ds.get('gender_distribution', {}))}")
    lines.append(f"{indent}LLMs:            {_fmt_dist(ds.get('llm_distribution', {}))}")
    lines.append(f"{indent}Voices:          {_fmt_dist(ds.get('voice_distribution', {}))}")
    lines.append(f"{indent}Scenarios:       {_fmt_dist(ds.get('scenario_distribution', {}))}")
    if include_engines_row:
        engine_dist = ds.get("engine_distribution", {})
        if engine_dist:
            lines.append(f"{indent}Engines:         {_fmt_dist(engine_dist)}")
    lines.append(
        f"{indent}Mobile/Desktop:  {ds.get('mobile_count', 0)} / {ds.get('desktop_count', 0)}"
    )
    if diar_agg:
        lines.append(
            f"{indent}Speaking (avg):  human {diar_agg.get('human_speaking_time_mean', 0):.1f}s  "
            f"agent {diar_agg.get('agent_speaking_time_mean', 0):.1f}s  "
            f"(human share {diar_agg.get('speaking_ratio_human_mean', 0):.1%})"
        )
        lines.append(
            f"{indent}Turns (avg):     total {diar_agg.get('total_turns_mean', 0):.1f}  "
            f"human {diar_agg.get('human_turns_mean', 0):.1f}  "
            f"agent {diar_agg.get('agent_turns_mean', 0):.1f}"
        )
        lines.append(
            f"{indent}Words (total):   human {diar_agg.get('human_word_count_total', 0)}  "
            f"agent {diar_agg.get('agent_word_count_total', 0)}"
        )
        if diar_agg.get("agent_latency_n"):
            overlap = diar_agg.get("agent_latency_n_overlap", 0) or 0
            suffix = (
                f"(n={diar_agg.get('agent_latency_n', 0)}, overlap={overlap} excl.)"
                if overlap else f"(n={diar_agg.get('agent_latency_n', 0)})"
            )
            lines.append(
                f"{indent}Agent latency:   mean {diar_agg.get('agent_latency_mean', 0):.3f}s  "
                f"median {diar_agg.get('agent_latency_median', 0):.3f}s  "
                f"min {diar_agg.get('agent_latency_min', 0):.3f}s  "
                f"max {diar_agg.get('agent_latency_max', 0):.3f}s  "
                f"{suffix}"
            )
    return lines


def _dataset_lines(stats: dict) -> list[str]:
    ds = stats.get("dataset", {}) if stats else {}
    diar = stats.get("diarization", {}).get("aggregate", {}) if stats else {}
    lines = ["── Dataset ──"]
    if not ds:
        lines.append("  (no dataset_statistics.json found — run --step analyze)")
        return lines

    lines.extend(_dataset_block_lines(ds, diar))

    # ── Per-engine breakdown (full stats, not just diarization) ───────
    per_engine_ds = ds.get("per_engine", {}) or {}
    per_engine_diar = (
        (stats or {}).get("diarization", {}).get("per_engine", {}) if stats else {}
    )
    engines = sorted(set(per_engine_ds.keys()) | set(per_engine_diar.keys()))
    for eng in engines:
        eng_ds = per_engine_ds.get(eng, {})
        eng_diar = per_engine_diar.get(eng, {})
        if not eng_ds and not eng_diar:
            continue
        lines.append("")
        lines.append(f"  ── Engine: {eng} ──")
        lines.extend(
            _dataset_block_lines(eng_ds, eng_diar, indent="    ", include_engines_row=False)
        )
    return lines


def _detection_lines() -> list[str]:
    lines = ["── Detection ──"]
    found = False
    for modality in MODALITIES:
        data = _load_json(DETECTION_DIR / f"{modality}_results.json")
        if not data:
            continue
        found = True
        lines.append(f"  [{modality}] — overall")
        header = f"    {'detector':<30} {'AUROC':>7} {'EER':>7} {'Acc':>7} {'F1':>7} {'N':>5}"
        lines.append(header)
        rows = []
        for name, payload in data.items():
            m = payload.get("metrics", {}) if isinstance(payload, dict) else {}
            n = len(payload.get("y_true", [])) if isinstance(payload, dict) else 0
            rows.append((
                name,
                m.get("auroc", float("nan")),
                m.get("eer", float("nan")),
                m.get("accuracy", float("nan")),
                m.get("f1_macro", float("nan")),
                n,
            ))
        rows.sort(key=lambda r: (r[1] if r[1] == r[1] else -1), reverse=True)
        for name, auroc, eer, acc, f1, n in rows:
            lines.append(
                f"    {name[:30]:<30} {auroc:>7.3f} {eer:>7.3f} {acc:>7.3f} {f1:>7.3f} {n:>5}"
            )

        # Per-engine breakdown
        engines: set[str] = set()
        for payload in data.values():
            if isinstance(payload, dict):
                engines.update((payload.get("by_engine") or {}).keys())
        for engine in sorted(engines):
            lines.append(f"  [{modality}] — engine={engine}")
            lines.append(header)
            eng_rows = []
            for name, payload in data.items():
                ed = (payload.get("by_engine") or {}).get(engine) if isinstance(payload, dict) else None
                if not ed:
                    continue
                m = ed.get("metrics", {})
                n = ed.get("n", len(ed.get("y_true", [])))
                eng_rows.append((
                    name,
                    m.get("auroc", float("nan")),
                    m.get("eer", float("nan")),
                    m.get("accuracy", float("nan")),
                    m.get("f1_macro", float("nan")),
                    n,
                ))
            eng_rows.sort(key=lambda r: (r[1] if r[1] == r[1] else -1), reverse=True)
            for name, auroc, eer, acc, f1, n in eng_rows:
                lines.append(
                    f"    {name[:30]:<30} {auroc:>7.3f} {eer:>7.3f} {acc:>7.3f} {f1:>7.3f} {n:>5}"
                )
    if not found:
        lines.append("  (no detection results found — run --step detect)")
    return lines


def _llm_block_lines(llm: dict, indent: str = "  ") -> list[str]:
    lines: list[str] = []
    lines.append(f"{indent}Analyzed:        {llm.get('total_analyzed', 0)} conversations")
    lines.append(f"{indent}Avg quality:     {llm.get('avg_quality', 0):.2f} / 5")
    top_topics = llm.get("top_topics", [])[:10]
    if top_topics:
        lines.append(
            f"{indent}Top topics:      "
            + ", ".join(f"{t}({c})" for t, c in top_topics)
        )
    lines.append(f"{indent}Conv. types:     {_fmt_dist(llm.get('conversation_types', {}))}")
    lines.append(f"{indent}Human sentiment: {_fmt_dist(llm.get('human_sentiment', {}))}")
    lines.append(f"{indent}Agent sentiment: {_fmt_dist(llm.get('agent_sentiment', {}))}")
    return lines


def _llm_lines() -> list[str]:
    lines = ["── LLM analysis ──"]
    llm = _load_json(LLM_ANALYSIS_DIR / "llm_stats.json")
    if not llm:
        lines.append("  (no llm_stats.json found — run --step llm)")
        return lines
    lines.extend(_llm_block_lines(llm))

    per_engine = llm.get("per_engine", {}) or {}
    for eng in sorted(per_engine.keys()):
        eng_stats = per_engine[eng]
        if not eng_stats or not eng_stats.get("total_analyzed"):
            continue
        lines.append("")
        lines.append(f"  ── Engine: {eng} ──")
        lines.extend(_llm_block_lines(eng_stats, indent="    "))
    return lines


def _artifact_lines() -> list[str]:
    lines = ["── Artifacts ──"]
    plots = sorted(PLOTS_DIR.glob("*.png")) if PLOTS_DIR.exists() else []
    lines.append(f"  Plots:           {len(plots)} PNGs in {PLOTS_DIR}")
    lines.append(f"  Detection JSON:  {DETECTION_DIR}")
    lines.append(f"  Stats JSON:      {STATS_DIR}")
    return lines


def build_summary() -> str:
    stats = _load_json(STATS_DIR / "dataset_statistics.json")
    sections = [
        "=" * 60,
        "DEEPSPEAK PIPELINE — RESULTS SUMMARY",
        "=" * 60,
        *_dataset_lines(stats),
        "",
        *_detection_lines(),
        "",
        *_llm_lines(),
        "",
        *_artifact_lines(),
        "=" * 60,
    ]
    return "\n".join(sections)


def run_summary():
    ensure_dirs()
    report = build_summary()
    print(report)
    out_path = STATS_DIR / "summary.txt"
    out_path.write_text(report + "\n")
    print(f"\n  Saved: {out_path}")
    return report


if __name__ == "__main__":
    run_summary()
