"""Text AI detection using Binoculars, Desklib, and DivEye."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoConfig, AutoModel, PreTrainedModel,
)

from deepspeak.config import (
    DIARIZATION_DIR, DETECTION_DIR, SPEAKER_CLIP_DIR, CSV_PATH, ensure_dirs,
)
from deepspeak.detection_metrics import (
    compute_all_metrics, plot_roc_curve, plot_metrics_comparison,
    save_detection_results, compute_metrics_by_engine, plot_by_engine,
    attach_engine,
)


# ── Detector 1: Binoculars ─────────────────────────────────────────────

class BinocularsDetector:
    """Binoculars: Zero-Shot Detection of Machine-Generated Text (ICML 2024).

    Uses perplexity ratio between two LMs to detect AI text.
    Lower scores = more likely AI-generated.
    """

    def __init__(self):
        self.observer_name = "tiiuae/falcon-rw-1b"
        self.performer_name = "tiiuae/falcon-rw-1b"
        self.observer = None
        self.performer = None
        self.tokenizer = None
        # Threshold from the paper
        self.threshold = 0.9015

    def _load_models(self):
        if self.observer is not None:
            return

        print("    Loading Binoculars models...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.observer = AutoModelForCausalLM.from_pretrained(
            self.observer_name, torch_dtype=torch.float32,
        )
        self.performer = AutoModelForCausalLM.from_pretrained(
            self.performer_name, torch_dtype=torch.float32,
        )
        self.observer.eval()
        self.performer.eval()

    def _compute_perplexity(self, model, input_ids):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            avg_log_prob = token_log_probs.mean().item()
            return math.exp(-avg_log_prob)

    def predict(self, text: str) -> dict:
        self._load_models()

        if len(text.strip()) < 20:
            return {"score": 0.5, "label": "unknown"}

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            padding=True,
        )
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 5:
            return {"score": 0.5, "label": "unknown"}

        ppl_observer = self._compute_perplexity(self.observer, input_ids)
        ppl_performer = self._compute_perplexity(self.performer, input_ids)

        # Binoculars score: ratio of perplexities
        if ppl_performer > 0:
            bino_score = ppl_observer / ppl_performer
        else:
            bino_score = 1.0

        # Convert to fake probability (lower bino_score = more likely AI)
        fake_prob = 1.0 / (1.0 + math.exp(5 * (bino_score - self.threshold)))

        return {
            "score": float(fake_prob),
            "label": "fake" if bino_score < self.threshold else "real",
            "bino_score": float(bino_score),
        }


# ── Detector 2: Desklib ────────────────────────────────────────────────

class DesklibAIDetectionModel(PreTrainedModel):
    """Desklib AI text detector (https://huggingface.co/desklib/ai-text-detector-v1.01).

    Mirrors the class in the model card, except the trailing
    ``self.init_weights()`` is replaced by ``self.post_init()``.
    transformers>=5 splits post-init into ``post_init -> init_weights``;
    ``post_init`` is what actually sets the instance attribute
    ``all_tied_weights_keys`` that ``from_pretrained`` looks up after
    loading the state dict. The model card predates the split.
    """

    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output


class DesklibDetector:
    """Desklib AI Text Detector — follows the inference recipe from the
    model card: ``max_length=768``, ``padding='max_length'``, sigmoid over
    the single-logit head, threshold 0.5."""

    def __init__(self):
        self.model_name = "desklib/ai-text-detector-v1.01"
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        if self.model is not None:
            return

        print("    Loading Desklib model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = DesklibAIDetectionModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        self._load_model()

        if len(text.strip()) < 20:
            return {"score": 0.5, "label": "unknown"}

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=768,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            score = float(torch.sigmoid(logits).item())

        return {
            "score": score,
            "label": "fake" if score >= 0.5 else "real",
        }


# ── Detector 3: DivEye ────────────────────────────────────────────────

class DivEyeDetector:
    """DivEye: Diversity-Driven AI Text Detection (TMLR '26).

    Measures unpredictability fluctuations in text using surprisal variance.
    AI text tends to have lower surprisal variance.
    """

    def __init__(self):
        self.model_name = "gpt2"
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model is not None:
            return

        print("    Loading DivEye (GPT-2) model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    def _compute_surprisal_features(self, text: str) -> dict | None:
        """Compute surprisal-based features for text."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        input_ids = inputs["input_ids"]

        if input_ids.shape[1] < 10:
            return None

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

            # Surprisal = -log_prob
            surprisals = -token_log_probs[0].numpy()

        if len(surprisals) < 5:
            return None

        # DivEye features: variance/std of surprisal and its differences
        features = {
            "surprisal_mean": float(np.mean(surprisals)),
            "surprisal_std": float(np.std(surprisals)),
            "surprisal_var": float(np.var(surprisals)),
            "surprisal_range": float(np.max(surprisals) - np.min(surprisals)),
        }

        # First-order differences (fluctuations)
        diffs = np.diff(surprisals)
        features["diff_std"] = float(np.std(diffs))
        features["diff_var"] = float(np.var(diffs))
        features["diff_mean_abs"] = float(np.mean(np.abs(diffs)))

        # Coefficient of variation
        if features["surprisal_mean"] > 0:
            features["cv"] = features["surprisal_std"] / features["surprisal_mean"]
        else:
            features["cv"] = 0

        return features

    def predict(self, text: str) -> dict:
        self._load_model()

        if len(text.strip()) < 20:
            return {"score": 0.5, "label": "unknown"}

        features = self._compute_surprisal_features(text)
        if features is None:
            return {"score": 0.5, "label": "unknown"}

        # DivEye scoring: AI text has lower surprisal variance
        # Use a combination of features to compute fake probability
        # Lower variance + lower diff_std = more likely AI
        var_score = 1.0 / (1.0 + math.exp(0.5 * (features["surprisal_var"] - 15)))
        diff_score = 1.0 / (1.0 + math.exp(0.5 * (features["diff_std"] - 5)))
        cv_score = 1.0 / (1.0 + math.exp(5 * (features["cv"] - 0.8)))

        fake_prob = 0.4 * var_score + 0.4 * diff_score + 0.2 * cv_score

        return {
            "score": float(fake_prob),
            "label": "fake" if fake_prob > 0.5 else "real",
            "features": features,
        }


# ── Main detection pipeline ────────────────────────────────────────────

def get_speaker_text(video_id: str, speaker_label: str) -> str | None:
    """Get transcript text for a specific speaker."""
    diar_path = DIARIZATION_DIR / f"{video_id}.json"
    if not diar_path.exists():
        return None

    with open(diar_path) as f:
        data = json.load(f)

    texts = data.get("speaker_texts", {})
    if speaker_label in texts:
        return texts[speaker_label]

    # Fall back to aligned transcript
    aligned = data.get("aligned_transcript", [])
    speaker_texts = [seg["text"] for seg in aligned if seg.get("speaker") == speaker_label]
    if speaker_texts:
        return " ".join(speaker_texts)

    return None


def run_text_detection(df: pd.DataFrame):
    """Run all text detectors and compute metrics."""
    ensure_dirs()

    cache_path = DETECTION_DIR / "text_raw_predictions.json"
    if cache_path.exists():
        with open(cache_path) as f:
            all_predictions = json.load(f)
    else:
        all_predictions = {}

    id_to_engine = {
        row["final_conversation_id"]: row.get("engine")
        for _, row in df.iterrows()
    }

    detectors = {
        "Binoculars": BinocularsDetector(),
        "Desklib": DesklibDetector(),
        "DivEye": DivEyeDetector(),
    }

    for det_name, detector in detectors.items():
        if det_name in all_predictions:
            print(f"  {det_name}: using cached predictions")
            continue

        print(f"\n  Running {det_name}...")
        predictions = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {det_name}"):
            vid = row["final_conversation_id"]
            human_pos = row.get("human_position", "right")
            agent_pos = "left" if human_pos == "right" else "right"
            engine = row.get("engine")

            diar_path = DIARIZATION_DIR / f"{vid}.json"
            if not diar_path.exists():
                continue

            with open(diar_path) as f:
                diar_data = json.load(f)

            # Load speaker mapping if available
            map_path = SPEAKER_CLIP_DIR / vid / "speaker_map.json"
            if map_path.exists():
                with open(map_path) as f:
                    speaker_map = json.load(f)["speaker_map"]
            else:
                # Use default mapping
                speakers = list(diar_data.get("speaker_texts", {}).keys())
                if len(speakers) >= 2:
                    speaker_map = {"left": speakers[0], "right": speakers[1]}
                elif len(speakers) == 1:
                    speaker_map = {"left": speakers[0], "right": speakers[0]}
                else:
                    continue

            for side in ["left", "right"]:
                speaker_label = speaker_map.get(side)
                if not speaker_label:
                    continue

                text = get_speaker_text(vid, speaker_label)
                if not text or len(text.strip()) < 30:
                    continue

                is_fake = 1 if side == agent_pos else 0

                try:
                    result = detector.predict(text)
                except Exception as e:
                    print(f"    Error on {vid}/{side}: {e}")
                    continue

                predictions.append({
                    "video_id": vid,
                    "side": side,
                    "speaker": speaker_label,
                    "is_fake": is_fake,
                    "score": result["score"],
                    "pred_label": result["label"],
                    "text_length": len(text),
                    "engine": str(engine) if engine is not None and engine == engine else None,
                })

        all_predictions[det_name] = predictions

        with open(cache_path, "w") as f:
            json.dump(all_predictions, f, indent=2)

    # Compute metrics
    results = {}
    for det_name, preds in all_predictions.items():
        if not preds:
            continue
        attach_engine(preds, id_to_engine)
        y_true = [p["is_fake"] for p in preds]
        y_scores = [p["score"] for p in preds]
        metrics = compute_all_metrics(y_true, y_scores)
        by_engine = compute_metrics_by_engine(preds)
        results[det_name] = {
            "metrics": metrics,
            "y_true": y_true,
            "y_scores": y_scores,
            "predictions": preds,
            "by_engine": by_engine,
        }
        print(f"\n  {det_name}: AUROC={metrics['auroc']:.3f}, "
              f"EER={metrics['eer']:.3f}, Acc={metrics['accuracy']:.3f}, "
              f"F1={metrics['f1_macro']:.3f}")
        for eng, ed in by_engine.items():
            em = ed["metrics"]
            print(f"      engine={eng}: AUROC={em['auroc']:.3f} "
                  f"EER={em['eer']:.3f} Acc={em['accuracy']:.3f} "
                  f"F1={em['f1_macro']:.3f} (n={ed['n']})")

    save_detection_results(results, "text")
    if results:
        plot_roc_curve(results, "text")
        plot_metrics_comparison(results, "text")
        plot_by_engine(results, "text")

    return results


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_text_detection(df)
