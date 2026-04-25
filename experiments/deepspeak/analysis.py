"""Dataset analysis and statistics reporting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from deepspeak.config import (
    CSV_PATH, DIARIZATION_DIR, SPEAKER_CLIP_DIR,
    STATS_DIR, PLOTS_DIR, ensure_dirs, ensure_engine_column,
)


def _compute_stats_for_subset(df: pd.DataFrame, include_engine_dist: bool = True) -> dict:
    """Compute the same comprehensive stats fields for any subset of ``df``.

    Used both for the overall dataset and for per-engine subsets.
    """
    stats: dict = {}
    if len(df) == 0:
        return {"total_conversations": 0}

    stats["total_conversations"] = len(df)
    stats["total_duration_seconds"] = float(df["video_duration_seconds"].sum())
    stats["total_duration_hours"] = round(stats["total_duration_seconds"] / 3600, 2)
    stats["avg_duration_seconds"] = float(df["video_duration_seconds"].mean())
    stats["median_duration_seconds"] = float(df["video_duration_seconds"].median())
    stats["std_duration_seconds"] = float(df["video_duration_seconds"].std())
    stats["min_duration_seconds"] = float(df["video_duration_seconds"].min())
    stats["max_duration_seconds"] = float(df["video_duration_seconds"].max())

    stats["total_size_mb"] = float(df["file_size_mb"].sum())
    stats["avg_size_mb"] = float(df["file_size_mb"].mean())

    stats["human_position_distribution"] = df["human_position"].value_counts().to_dict()

    stats["gender_distribution"] = df["recording_gender"].value_counts().to_dict()
    stats["llm_distribution"] = df["recording_llm"].value_counts().to_dict()
    stats["voice_distribution"] = df["recording_voice"].value_counts().to_dict()
    stats["scenario_distribution"] = df["recording_scenario"].value_counts().to_dict()
    if include_engine_dist:
        if "engine" in df.columns:
            stats["engine_distribution"] = df["engine"].value_counts(dropna=False).to_dict()
        else:
            stats["engine_distribution"] = {}

    mobile = df["mobile"].fillna(False).astype(bool)
    stats["mobile_count"] = int(mobile.sum())
    stats["desktop_count"] = int((~mobile).sum())

    virtual_bg = df["virtual_background"].fillna(False).astype(bool)
    stats["virtual_background_count"] = int(virtual_bg.sum())
    stats["no_virtual_background_count"] = int((~virtual_bg).sum())

    stats["user_drop_count"] = int(df["user_drop"].fillna(False).astype(bool).sum())

    delay_cols = ["beginning_delay_seconds", "end_delay_seconds"]
    for col in delay_cols:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) > 0:
            stats[f"{col}_count"] = len(valid)
            stats[f"{col}_mean"] = float(valid.mean())
            stats[f"{col}_median"] = float(valid.median())
            stats[f"{col}_std"] = float(valid.std())

    if "speech_start_seconds" in df.columns:
        stats["avg_speech_start_seconds"] = float(df["speech_start_seconds"].mean())
    if "longest_silence_seconds" in df.columns:
        stats["avg_longest_silence_seconds"] = float(df["longest_silence_seconds"].mean())
        stats["max_longest_silence_seconds"] = float(df["longest_silence_seconds"].max())

    return stats


def compute_conversation_stats(df: pd.DataFrame) -> dict:
    """Compute comprehensive statistics about the conversations.

    Also emits a ``per_engine`` section mirroring the overall fields for each
    deepfake engine group (e.g. tavus, heygen) so downstream reports can show
    full breakdowns rather than only the aggregate totals.
    """
    ensure_dirs()
    stats = _compute_stats_for_subset(df, include_engine_dist=True)

    per_engine: dict[str, dict] = {}
    if "engine" in df.columns:
        engines = [
            e for e in df["engine"].dropna().unique()
            if isinstance(e, str) and e != ""
        ]
        for eng in sorted(engines):
            subset = df[df["engine"] == eng]
            if len(subset) == 0:
                continue
            per_engine[str(eng)] = _compute_stats_for_subset(
                subset, include_engine_dist=False,
            )
    stats["per_engine"] = per_engine

    return stats


def compute_diarization_stats(df: pd.DataFrame) -> dict:
    """Compute stats from diarization results."""
    ensure_dirs()
    stats = {
        "per_conversation": [],
        "aggregate": {},
    }

    all_human_times = []
    all_agent_times = []
    all_human_turns = []
    all_agent_turns = []
    all_human_avg_utterance = []
    all_agent_avg_utterance = []
    all_human_words = []
    all_agent_words = []
    all_turn_counts = []
    all_overlap_times = []
    all_agent_latencies: list[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing diarization stats"):
        vid = row["final_conversation_id"]
        human_pos = row.get("human_position", "right")
        agent_pos = "left" if human_pos == "right" else "right"

        diar_path = DIARIZATION_DIR / f"{vid}.json"
        if not diar_path.exists():
            continue

        with open(diar_path) as f:
            diar = json.load(f)

        # Load speaker mapping
        map_path = SPEAKER_CLIP_DIR / vid / "speaker_map.json"
        if map_path.exists():
            with open(map_path) as f:
                speaker_map = json.load(f)["speaker_map"]
        else:
            speakers = list(diar.get("speaker_texts", {}).keys())
            if len(speakers) >= 2:
                speaker_map = {human_pos: speakers[0], agent_pos: speakers[1]}
            elif len(speakers) == 1:
                speaker_map = {human_pos: speakers[0], agent_pos: speakers[0]}
            else:
                continue

        human_speaker = speaker_map.get(human_pos)
        agent_speaker = speaker_map.get(agent_pos)

        segments = diar.get("diarization_segments", [])
        aligned = diar.get("aligned_transcript", [])

        # Speaking time per role
        human_time = sum(
            s["end"] - s["start"] for s in segments if s["speaker"] == human_speaker
        )
        agent_time = sum(
            s["end"] - s["start"] for s in segments if s["speaker"] == agent_speaker
        )

        # Turn counts
        human_turns = sum(1 for s in segments if s["speaker"] == human_speaker)
        agent_turns = sum(1 for s in segments if s["speaker"] == agent_speaker)

        # Average utterance duration
        human_durations = [
            s["end"] - s["start"] for s in segments if s["speaker"] == human_speaker
        ]
        agent_durations = [
            s["end"] - s["start"] for s in segments if s["speaker"] == agent_speaker
        ]
        human_avg_utt = np.mean(human_durations) if human_durations else 0
        agent_avg_utt = np.mean(agent_durations) if agent_durations else 0

        # Word counts from transcript
        human_text = diar.get("speaker_texts", {}).get(human_speaker, "")
        agent_text = diar.get("speaker_texts", {}).get(agent_speaker, "")
        human_wc = len(human_text.split()) if human_text else 0
        agent_wc = len(agent_text.split()) if agent_text else 0

        # Total turns (alternations)
        total_turns = 0
        prev_speaker = None
        for s in segments:
            if s["speaker"] != prev_speaker:
                total_turns += 1
                prev_speaker = s["speaker"]

        # Agent response latency: gap between the end of a human segment
        # and the start of the immediately following agent segment. Sorted
        # by start time to be robust to out-of-order segment output.
        # The raw signed list is kept for transparency; stats are reported
        # on the non-negative subset only (see _latency_stats).
        sorted_segs = sorted(segments, key=lambda s: s["start"])
        agent_latencies: list[float] = []
        for prev, nxt in zip(sorted_segs, sorted_segs[1:]):
            if prev["speaker"] == human_speaker and nxt["speaker"] == agent_speaker:
                agent_latencies.append(float(nxt["start"]) - float(prev["end"]))
        pos_latencies = [x for x in agent_latencies if x >= 0]

        conv_stats = {
            "video_id": vid,
            "human_speaking_time": round(human_time, 2),
            "agent_speaking_time": round(agent_time, 2),
            "human_turns": human_turns,
            "agent_turns": agent_turns,
            "total_turns": total_turns,
            "human_avg_utterance_duration": round(human_avg_utt, 2),
            "agent_avg_utterance_duration": round(agent_avg_utt, 2),
            "human_word_count": human_wc,
            "agent_word_count": agent_wc,
            "speaking_ratio_human": round(
                human_time / (human_time + agent_time), 3
            ) if (human_time + agent_time) > 0 else 0,
            "agent_latency_n": len(pos_latencies),
            "agent_latency_n_overlap": len(agent_latencies) - len(pos_latencies),
            "agent_latency_mean": round(float(np.mean(pos_latencies)), 3) if pos_latencies else None,
            "agent_latency_median": round(float(np.median(pos_latencies)), 3) if pos_latencies else None,
            "agent_latency_min": round(float(np.min(pos_latencies)), 3) if pos_latencies else None,
            "agent_latency_max": round(float(np.max(pos_latencies)), 3) if pos_latencies else None,
            "agent_latencies": [round(x, 3) for x in agent_latencies],
            "video_duration": row["video_duration_seconds"],
            "llm": row.get("recording_llm", ""),
            "voice": row.get("recording_voice", ""),
            "scenario": row.get("recording_scenario", ""),
            "gender": row.get("recording_gender", ""),
            "engine": row.get("engine", ""),
        }
        stats["per_conversation"].append(conv_stats)
        all_agent_latencies.extend(agent_latencies)

        all_human_times.append(human_time)
        all_agent_times.append(agent_time)
        all_human_turns.append(human_turns)
        all_agent_turns.append(agent_turns)
        all_human_avg_utterance.append(human_avg_utt)
        all_agent_avg_utterance.append(agent_avg_utt)
        all_human_words.append(human_wc)
        all_agent_words.append(agent_wc)
        all_turn_counts.append(total_turns)

    # Aggregate stats
    if all_human_times:
        stats["aggregate"] = {
            "human_speaking_time_mean": round(float(np.mean(all_human_times)), 2),
            "human_speaking_time_median": round(float(np.median(all_human_times)), 2),
            "human_speaking_time_std": round(float(np.std(all_human_times)), 2),
            "agent_speaking_time_mean": round(float(np.mean(all_agent_times)), 2),
            "agent_speaking_time_median": round(float(np.median(all_agent_times)), 2),
            "agent_speaking_time_std": round(float(np.std(all_agent_times)), 2),
            "human_turns_mean": round(float(np.mean(all_human_turns)), 2),
            "agent_turns_mean": round(float(np.mean(all_agent_turns)), 2),
            "total_turns_mean": round(float(np.mean(all_turn_counts)), 2),
            "total_turns_median": round(float(np.median(all_turn_counts)), 2),
            "human_avg_utterance_mean": round(float(np.mean(all_human_avg_utterance)), 2),
            "agent_avg_utterance_mean": round(float(np.mean(all_agent_avg_utterance)), 2),
            "human_word_count_mean": round(float(np.mean(all_human_words)), 2),
            "agent_word_count_mean": round(float(np.mean(all_agent_words)), 2),
            "human_word_count_total": int(sum(all_human_words)),
            "agent_word_count_total": int(sum(all_agent_words)),
            "speaking_ratio_human_mean": round(
                float(np.mean(all_human_times)) /
                (float(np.mean(all_human_times)) + float(np.mean(all_agent_times))), 3
            ) if (np.mean(all_human_times) + np.mean(all_agent_times)) > 0 else 0,
            **_latency_stats(all_agent_latencies),
        }

    # Per-engine aggregate (same fields, but computed on a per-engine subset).
    stats["per_engine"] = _aggregate_by_engine(stats["per_conversation"])

    return stats


def _latency_stats(latencies: list[float]) -> dict:
    """Summarise a pool of agent response latencies (in seconds).

    Only non-negative gaps are reported. Negative values correspond to
    overlaps between the per-side human and agent transcriptions (which
    this pipeline produces independently per speaker side) and are
    dominated by diarization / VAD artifacts rather than genuine
    barge-ins, so they're counted but excluded from the summary stats.
    """
    total = len(latencies)
    positive = [x for x in latencies if x >= 0]
    n_overlap = total - len(positive)

    if not positive:
        return {
            "agent_latency_n": 0,
            "agent_latency_n_overlap": n_overlap,
            "agent_latency_mean": None,
            "agent_latency_median": None,
            "agent_latency_std": None,
            "agent_latency_min": None,
            "agent_latency_max": None,
            "agent_latency_p25": None,
            "agent_latency_p75": None,
        }
    arr = np.asarray(positive, dtype=float)
    return {
        "agent_latency_n": int(arr.size),
        "agent_latency_n_overlap": n_overlap,
        "agent_latency_mean": round(float(arr.mean()), 3),
        "agent_latency_median": round(float(np.median(arr)), 3),
        "agent_latency_std": round(float(arr.std()), 3),
        "agent_latency_min": round(float(arr.min()), 3),
        "agent_latency_max": round(float(arr.max()), 3),
        "agent_latency_p25": round(float(np.percentile(arr, 25)), 3),
        "agent_latency_p75": round(float(np.percentile(arr, 75)), 3),
    }


def _aggregate_by_engine(per_conv: list[dict]) -> dict[str, dict]:
    """Compute the same aggregate fields per engine group."""
    out: dict[str, dict] = {}
    for conv in per_conv:
        eng = conv.get("engine")
        if eng in (None, "") or (isinstance(eng, float) and np.isnan(eng)):
            continue
        out.setdefault(str(eng), []).append(conv)

    agg: dict[str, dict] = {}
    for eng, rows in out.items():
        human_t = [r["human_speaking_time"] for r in rows]
        agent_t = [r["agent_speaking_time"] for r in rows]
        human_turns = [r["human_turns"] for r in rows]
        agent_turns = [r["agent_turns"] for r in rows]
        turn_counts = [r["total_turns"] for r in rows]
        human_utt = [r["human_avg_utterance_duration"] for r in rows]
        agent_utt = [r["agent_avg_utterance_duration"] for r in rows]
        human_wc = [r["human_word_count"] for r in rows]
        agent_wc = [r["agent_word_count"] for r in rows]
        latencies: list[float] = []
        for r in rows:
            latencies.extend(r.get("agent_latencies") or [])

        mean_h = float(np.mean(human_t)) if human_t else 0.0
        mean_a = float(np.mean(agent_t)) if agent_t else 0.0
        ratio = round(mean_h / (mean_h + mean_a), 3) if (mean_h + mean_a) > 0 else 0.0

        agg[eng] = {
            "n_conversations": len(rows),
            "human_speaking_time_mean": round(mean_h, 2),
            "human_speaking_time_median": round(float(np.median(human_t)), 2) if human_t else 0.0,
            "agent_speaking_time_mean": round(mean_a, 2),
            "agent_speaking_time_median": round(float(np.median(agent_t)), 2) if agent_t else 0.0,
            "human_turns_mean": round(float(np.mean(human_turns)), 2) if human_turns else 0.0,
            "agent_turns_mean": round(float(np.mean(agent_turns)), 2) if agent_turns else 0.0,
            "total_turns_mean": round(float(np.mean(turn_counts)), 2) if turn_counts else 0.0,
            "human_avg_utterance_mean": round(float(np.mean(human_utt)), 2) if human_utt else 0.0,
            "agent_avg_utterance_mean": round(float(np.mean(agent_utt)), 2) if agent_utt else 0.0,
            "human_word_count_mean": round(float(np.mean(human_wc)), 2) if human_wc else 0.0,
            "agent_word_count_mean": round(float(np.mean(agent_wc)), 2) if agent_wc else 0.0,
            "human_word_count_total": int(sum(human_wc)),
            "agent_word_count_total": int(sum(agent_wc)),
            "speaking_ratio_human_mean": ratio,
            **_latency_stats(latencies),
        }
    return agg


def plot_dataset_stats(df: pd.DataFrame, diar_stats: dict):
    """Generate comprehensive plots for the dataset."""
    ensure_dirs()

    fig_size = (12, 8)

    # 1. Duration distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["video_duration_seconds"] / 60, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Count")
    ax.set_title("Video Duration Distribution")
    ax.axvline(df["video_duration_seconds"].mean() / 60, color="r",
               linestyle="--", label=f"Mean: {df['video_duration_seconds'].mean()/60:.1f}m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "duration_distribution.png", dpi=150)
    plt.close(fig)

    # 2. LLM distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    df["recording_llm"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("Set2"))
    ax.set_title("LLM Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "llm_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Voice engine distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    df["recording_voice"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("Set3"))
    ax.set_title("Voice Engine Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "voice_distribution.png", dpi=150)
    plt.close(fig)

    # 3b. Deepfake engine (tavus / heygen) distribution
    if "engine" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 6))
        df["engine"].value_counts(dropna=False).plot(
            kind="bar", ax=ax, color=sns.color_palette("Set2"),
        )
        ax.set_title("Deepfake Engine Distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "engine_distribution.png", dpi=150)
        plt.close(fig)

    # 4. Gender distribution
    fig, ax = plt.subplots(figsize=(6, 6))
    df["recording_gender"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
    ax.set_title("Gender Distribution")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "gender_distribution.png", dpi=150)
    plt.close(fig)

    # 5. Scenario distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    df["recording_scenario"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("husl"))
    ax.set_title("Scenario Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "scenario_distribution.png", dpi=150)
    plt.close(fig)

    # 6. Mobile vs Desktop
    fig, ax = plt.subplots(figsize=(6, 6))
    mobile_counts = df["mobile"].value_counts()
    labels = ["Desktop" if not k else "Mobile" for k in mobile_counts.index]
    mobile_counts.plot(kind="pie", ax=ax, labels=labels, autopct="%1.1f%%")
    ax.set_title("Mobile vs Desktop")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mobile_distribution.png", dpi=150)
    plt.close(fig)

    # 7-12. Diarization-based plots (if we have per-conversation data)
    if diar_stats.get("per_conversation"):
        conv_df = pd.DataFrame(diar_stats["per_conversation"])

        # 7. Speaking time: human vs agent
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(conv_df["human_speaking_time"], bins=20, alpha=0.6, label="Human", edgecolor="black")
        ax.hist(conv_df["agent_speaking_time"], bins=20, alpha=0.6, label="Agent", edgecolor="black")
        ax.set_xlabel("Speaking Time (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Speaking Time Distribution: Human vs Agent")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "speaking_time_distribution.png", dpi=150)
        plt.close(fig)

        # 8. Speaking ratio pie chart
        agg = diar_stats["aggregate"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            [agg.get("human_speaking_time_mean", 1), agg.get("agent_speaking_time_mean", 1)],
            labels=["Human", "Agent"],
            autopct="%1.1f%%",
            colors=["#66b3ff", "#ff9999"],
        )
        ax.set_title("Average Speaking Time Ratio")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "speaking_ratio.png", dpi=150)
        plt.close(fig)

        # 9. Turn count distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(conv_df["total_turns"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Number of Turns")
        ax.set_ylabel("Count")
        ax.set_title("Turn Count Distribution")
        ax.axvline(conv_df["total_turns"].mean(), color="r", linestyle="--",
                    label=f"Mean: {conv_df['total_turns'].mean():.1f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "turn_count_distribution.png", dpi=150)
        plt.close(fig)

        # 10. Average utterance duration: human vs agent
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(conv_df))
        width = 0.35
        ax.bar(x - width/2, conv_df["human_avg_utterance_duration"], width,
               label="Human", alpha=0.7)
        ax.bar(x + width/2, conv_df["agent_avg_utterance_duration"], width,
               label="Agent", alpha=0.7)
        ax.set_xlabel("Conversation")
        ax.set_ylabel("Avg Utterance Duration (s)")
        ax.set_title("Average Utterance Duration per Conversation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "utterance_duration_per_conv.png", dpi=150)
        plt.close(fig)

        # 11. Word count comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(conv_df["human_word_count"], conv_df["agent_word_count"], alpha=0.6)
        ax.set_xlabel("Human Word Count")
        ax.set_ylabel("Agent Word Count")
        ax.set_title("Word Count: Human vs Agent")
        max_val = max(conv_df["human_word_count"].max(), conv_df["agent_word_count"].max())
        ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Equal")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "word_count_scatter.png", dpi=150)
        plt.close(fig)

        # 12. Speaking time by LLM
        if "llm" in conv_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for i, role in enumerate(["human", "agent"]):
                grouped = conv_df.groupby("llm")[f"{role}_speaking_time"].mean()
                grouped.plot(kind="bar", ax=axes[i], color=sns.color_palette("Set2"))
                axes[i].set_title(f"Avg {role.title()} Speaking Time by LLM")
                axes[i].set_ylabel("Seconds")
                axes[i].tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "speaking_time_by_llm.png", dpi=150)
            plt.close(fig)

        # 12b. Speaking time + turns by engine
        if "engine" in conv_df.columns and conv_df["engine"].notna().any():
            engine_df = conv_df[conv_df["engine"].notna() & (conv_df["engine"] != "")]
            if not engine_df.empty:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                axes[0].set_title("Avg Human Speaking Time by Engine")
                engine_df.groupby("engine")["human_speaking_time"].mean().plot(
                    kind="bar", ax=axes[0], color=sns.color_palette("Set2"),
                )
                axes[0].set_ylabel("Seconds")
                axes[1].set_title("Avg Agent Speaking Time by Engine")
                engine_df.groupby("engine")["agent_speaking_time"].mean().plot(
                    kind="bar", ax=axes[1], color=sns.color_palette("Set2"),
                )
                axes[1].set_ylabel("Seconds")
                axes[2].set_title("Avg Total Turns by Engine")
                engine_df.groupby("engine")["total_turns"].mean().plot(
                    kind="bar", ax=axes[2], color=sns.color_palette("Set2"),
                )
                axes[2].set_ylabel("Turns")
                for ax in axes:
                    ax.tick_params(axis="x", rotation=0)
                fig.tight_layout()
                fig.savefig(PLOTS_DIR / "speaking_stats_by_engine.png", dpi=150)
                plt.close(fig)

        # 13. File size vs duration
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["video_duration_seconds"], df["file_size_mb"], alpha=0.6)
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("File Size (MB)")
        ax.set_title("File Size vs Duration")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "filesize_vs_duration.png", dpi=150)
        plt.close(fig)

        # 14. Speech start time distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["speech_start_seconds"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Speech Start (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Speech Start Time Distribution")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "speech_start_distribution.png", dpi=150)
        plt.close(fig)

        # 15. Longest silence distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["longest_silence_seconds"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Longest Silence (seconds)")
        ax.set_ylabel("Count")
        ax.set_title("Longest Silence Duration Distribution")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "silence_distribution.png", dpi=150)
        plt.close(fig)

    print(f"  Saved all plots to {PLOTS_DIR}")


def run_analysis(df: pd.DataFrame):
    """Run full dataset analysis."""
    ensure_dirs()
    ensure_engine_column(df)

    print("Computing dataset statistics...")
    dataset_stats = compute_conversation_stats(df)

    print("Computing diarization statistics...")
    diar_stats = compute_diarization_stats(df)

    # Save stats
    all_stats = {
        "dataset": dataset_stats,
        "diarization": diar_stats,
    }

    stats_path = STATS_DIR / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"  Saved stats: {stats_path}")

    # Generate plots
    print("Generating plots...")
    plot_dataset_stats(df, diar_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total conversations: {dataset_stats['total_conversations']}")
    print(f"Total duration: {dataset_stats['total_duration_hours']} hours")
    print(f"Avg duration: {dataset_stats['avg_duration_seconds']:.1f}s")
    print(f"Gender: {dataset_stats['gender_distribution']}")
    print(f"LLMs: {dataset_stats['llm_distribution']}")
    print(f"Voices: {dataset_stats['voice_distribution']}")
    if dataset_stats.get("engine_distribution"):
        print(f"Engines: {dataset_stats['engine_distribution']}")

    if diar_stats.get("aggregate"):
        agg = diar_stats["aggregate"]
        print(f"\nHuman avg speaking time: {agg.get('human_speaking_time_mean', 0):.1f}s")
        print(f"Agent avg speaking time: {agg.get('agent_speaking_time_mean', 0):.1f}s")
        print(f"Avg total turns: {agg.get('total_turns_mean', 0):.1f}")
        print(f"Human avg utterance: {agg.get('human_avg_utterance_mean', 0):.1f}s")
        print(f"Agent avg utterance: {agg.get('agent_avg_utterance_mean', 0):.1f}s")
        print(f"Human total words: {agg.get('human_word_count_total', 0)}")
        print(f"Agent total words: {agg.get('agent_word_count_total', 0)}")

    return all_stats


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_analysis(df)
