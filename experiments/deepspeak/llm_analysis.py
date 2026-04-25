"""LLM-based analysis: keyword extraction and summarization via OpenAI."""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from deepspeak.config import (
    DIARIZATION_DIR, LLM_ANALYSIS_DIR, SPEAKER_CLIP_DIR,
    CSV_PATH, OPENAI_API_KEY, ensure_dirs, ensure_engine_column,
)


def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def analyze_conversation(video_id: str, client=None) -> dict | None:
    """Use OpenAI to extract keywords, summary, and metadata from a conversation."""
    ensure_dirs()

    output_path = LLM_ANALYSIS_DIR / f"{video_id}.json"
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)

    diar_path = DIARIZATION_DIR / f"{video_id}.json"
    if not diar_path.exists():
        return None

    with open(diar_path) as f:
        diar = json.load(f)

    full_text = diar.get("full_text", "")
    if not full_text or len(full_text.strip()) < 50:
        return None

    # Build speaker-labeled transcript
    aligned = diar.get("aligned_transcript", [])
    transcript_lines = []
    for seg in aligned:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        transcript_lines.append(f"{speaker}: {text}")
    labeled_transcript = "\n".join(transcript_lines)

    if client is None:
        client = get_openai_client()

    prompt = f"""Analyze this conversation transcript between a human and an AI agent in a video call setting.

TRANSCRIPT:
{labeled_transcript[:4000]}

Provide the following in JSON format:
1. "summary": A 2-3 sentence summary of the conversation
2. "topics": List of main topics discussed (5-10 keywords/phrases)
3. "conversation_type": The type/category of conversation (e.g., "interview", "casual chat", "technical discussion", "customer service", etc.)
4. "sentiment_human": Overall sentiment of the human speaker (positive/neutral/negative)
5. "sentiment_agent": Overall sentiment of the AI agent (positive/neutral/negative)
6. "conversation_quality": Rate the quality/naturalness of the conversation (1-5)
7. "key_moments": List of 2-3 notable moments or exchanges in the conversation
8. "language_complexity": Rate the language complexity (simple/moderate/complex)
9. "turn_taking_quality": How natural is the turn-taking? (smooth/moderate/awkward)
10. "ai_detection_cues": Any observable cues that might indicate one speaker is AI (list)

Return ONLY valid JSON, no markdown formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )

        content = response.choices[0].message.content.strip()
        # Try to parse as JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
    except json.JSONDecodeError:
        analysis = {"raw_response": content, "parse_error": True}
    except Exception as e:
        analysis = {"error": str(e)}

    result = {
        "video_id": video_id,
        "analysis": analysis,
        "transcript_length": len(full_text),
        "num_segments": len(aligned),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _aggregate_llm_results(results: list[dict]) -> dict:
    """Roll up a list of per-conversation LLM analysis results into stats."""
    topics_counter: dict[str, int] = {}
    conv_types: dict[str, int] = {}
    sentiments_human: dict[str, int] = {}
    sentiments_agent: dict[str, int] = {}
    qualities: list[int] = []

    for r in results:
        analysis = r.get("analysis", {})
        if not (isinstance(analysis, dict) and "error" not in analysis):
            continue

        for topic in analysis.get("topics", []):
            topic_lower = topic.lower().strip()
            topics_counter[topic_lower] = topics_counter.get(topic_lower, 0) + 1

        ct = analysis.get("conversation_type", "unknown")
        conv_types[ct] = conv_types.get(ct, 0) + 1

        sh = analysis.get("sentiment_human", "unknown")
        sentiments_human[sh] = sentiments_human.get(sh, 0) + 1
        sa = analysis.get("sentiment_agent", "unknown")
        sentiments_agent[sa] = sentiments_agent.get(sa, 0) + 1

        q = analysis.get("conversation_quality")
        if q is not None:
            qualities.append(int(q) if isinstance(q, (int, float)) else 3)

    return {
        "top_topics": sorted(topics_counter.items(), key=lambda x: -x[1])[:30],
        "conversation_types": conv_types,
        "human_sentiment": sentiments_human,
        "agent_sentiment": sentiments_agent,
        "avg_quality": sum(qualities) / len(qualities) if qualities else 0,
        "total_analyzed": len(results),
    }


def run_llm_analysis(df: pd.DataFrame):
    """Run LLM analysis on all conversations."""
    ensure_dirs()
    ensure_engine_column(df)
    client = get_openai_client()

    id_to_engine: dict[str, str] = {}
    if "engine" in df.columns:
        for _, row in df.iterrows():
            eng = row.get("engine")
            if isinstance(eng, str) and eng:
                id_to_engine[row["final_conversation_id"]] = eng

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM Analysis"):
        vid = row["final_conversation_id"]
        result = analyze_conversation(vid, client)
        if result:
            eng = id_to_engine.get(vid)
            if eng:
                result["engine"] = eng
            results.append(result)

    # Save aggregate
    aggregate_path = LLM_ANALYSIS_DIR / "aggregate_analysis.json"
    with open(aggregate_path, "w") as f:
        json.dump(results, f, indent=2)

    llm_stats = _aggregate_llm_results(results)

    # Per-engine breakdown — same fields, computed on each engine subset.
    per_engine: dict[str, dict] = {}
    engines_seen = {r.get("engine") for r in results if r.get("engine")}
    for eng in sorted(e for e in engines_seen if isinstance(e, str)):
        subset = [r for r in results if r.get("engine") == eng]
        per_engine[eng] = _aggregate_llm_results(subset)
    llm_stats["per_engine"] = per_engine

    stats_path = LLM_ANALYSIS_DIR / "llm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(llm_stats, f, indent=2)

    print(f"\n  LLM Analysis complete: {len(results)} conversations analyzed")
    print(f"  Top topics: {llm_stats['top_topics'][:10]}")
    print(f"  Conversation types: {llm_stats['conversation_types']}")

    return llm_stats


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_llm_analysis(df)
