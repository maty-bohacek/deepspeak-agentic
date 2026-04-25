"""Shared detection metrics computation and plotting."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)

from deepspeak.config import PLOTS_DIR, DETECTION_DIR, ensure_dirs


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find the point where FPR == FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer)


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """Compute EER, F1, AUROC, accuracy, and per-class metrics."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    if y_pred is None:
        y_pred = (y_scores >= threshold).astype(int)
    else:
        y_pred = np.array(y_pred)

    metrics = {}

    # Overall metrics
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        metrics["auroc"] = float("nan")

    metrics["eer"] = compute_eer(y_true, y_scores)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for cls_key in ["0", "1"]:
        if cls_key in report:
            label = "real" if cls_key == "0" else "fake"
            metrics[f"{label}_precision"] = report[cls_key]["precision"]
            metrics[f"{label}_recall"] = report[cls_key]["recall"]
            metrics[f"{label}_f1"] = report[cls_key]["f1-score"]
            metrics[f"{label}_support"] = report[cls_key]["support"]

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def plot_roc_curve(
    results: dict[str, dict],
    modality: str,
    output_path: Path | None = None,
    title: str | None = None,
):
    """Plot ROC curves for multiple detectors on the same plot."""
    ensure_dirs()
    if output_path is None:
        output_path = PLOTS_DIR / f"roc_{modality}.png"

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, data in results.items():
        y_true = np.array(data["y_true"])
        y_scores = np.array(data["y_scores"])
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auroc = roc_auc_score(y_true, y_scores)
            ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC Curves - {modality.title()} Deepfake Detection")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ROC plot: {output_path}")


def plot_metrics_comparison(
    results: dict[str, dict],
    modality: str,
    output_path: Path | None = None,
    title: str | None = None,
):
    """Bar chart comparing key metrics across detectors."""
    ensure_dirs()
    if output_path is None:
        output_path = PLOTS_DIR / f"metrics_{modality}.png"

    metric_names = ["accuracy", "f1_macro", "auroc", "eer"]
    display_names = ["Accuracy", "F1 (Macro)", "AUROC", "EER"]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 5))

    detector_names = list(results.keys())
    colors = sns.color_palette("husl", len(detector_names))

    for i, (metric, display) in enumerate(zip(metric_names, display_names)):
        values = []
        for name in detector_names:
            val = results[name].get("metrics", {}).get(metric, 0)
            values.append(val)
        axes[i].bar(detector_names, values, color=colors)
        axes[i].set_title(display)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=45)
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(
        title or f"{modality.title()} Detection - Metrics Comparison",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved metrics plot: {output_path}")


# ── Engine breakdown ──────────────────────────────────────────────────

def compute_metrics_by_engine(preds: list[dict]) -> dict[str, dict]:
    """Partition predictions by ``engine`` and compute metrics for each.

    Returns ``{engine: {"metrics": ..., "y_true": ..., "y_scores": ...,
    "predictions": [...], "n": N}}``. Engines with fewer than two classes
    are still reported (metrics may be NaN) so callers can display counts.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        eng = p.get("engine")
        if eng is None or (isinstance(eng, float) and np.isnan(eng)):
            continue
        groups[str(eng)].append(p)

    out: dict[str, dict] = {}
    for engine, plist in groups.items():
        y_true = [p["is_fake"] for p in plist]
        y_scores = [p["score"] for p in plist]
        out[engine] = {
            "metrics": compute_all_metrics(y_true, y_scores),
            "y_true": y_true,
            "y_scores": y_scores,
            "predictions": plist,
            "n": len(plist),
        }
    return out


def plot_by_engine(results: dict[str, dict], modality: str) -> None:
    """Emit per-engine ROC and metrics-comparison plots.

    ``results`` is the top-level detector→payload dict from each
    ``run_*_detection``. Each payload is expected to carry a ``by_engine``
    sub-dict produced by :func:`compute_metrics_by_engine`.
    """
    ensure_dirs()

    engines: set[str] = set()
    for payload in results.values():
        engines.update((payload or {}).get("by_engine", {}).keys())
    if not engines:
        return

    for engine in sorted(engines):
        sub: dict[str, dict] = {}
        for det_name, payload in results.items():
            eng_data = (payload or {}).get("by_engine", {}).get(engine)
            if eng_data:
                sub[det_name] = eng_data
        if not sub:
            continue
        plot_roc_curve(
            sub, modality,
            output_path=PLOTS_DIR / f"roc_{modality}_{engine}.png",
            title=f"ROC Curves - {modality.title()} Detection (engine={engine})",
        )
        plot_metrics_comparison(
            sub, modality,
            output_path=PLOTS_DIR / f"metrics_{modality}_{engine}.png",
            title=f"{modality.title()} Detection - Metrics (engine={engine})",
        )


def attach_engine(
    predictions: list[dict],
    id_to_engine: dict[str, str],
) -> None:
    """Backfill ``engine`` onto cached predictions that pre-date this column."""
    for p in predictions:
        if not p.get("engine"):
            eng = id_to_engine.get(p.get("video_id"))
            if eng is not None and not (isinstance(eng, float) and np.isnan(eng)):
                p["engine"] = str(eng)


def save_detection_results(results: dict, modality: str):
    """Save detection results to JSON."""
    ensure_dirs()
    output_path = DETECTION_DIR / f"{modality}_results.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for name, data in results.items():
        serializable[name] = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                serializable[name][k] = v.tolist()
            elif isinstance(v, dict):
                serializable[name][k] = {
                    kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for kk, vv in v.items()
                }
            else:
                serializable[name][k] = v

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Saved results: {output_path}")
