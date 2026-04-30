#!/usr/bin/env python3
"""
Script 2: Select combinations for a run.
Filters all_combinations.txt based on criteria and writes selected_combinations.txt.
"""
import os
import sys
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline_paths import OUTPUT_DIR, assert_collection_platform, load_app_config


# ============================================================
# CONFIGURATION - Edit these to filter combinations
# ============================================================
PROVIDER = "heygen"            # Options: "heygen", or None for any
LLM = None                     # Options: "heygen-gpt-5.4", "heygen-default", or None for any
VOICE = None                   # Options: "heygen", "elevenlabs", or None for any
SCENARIOS = None               # Options: ["1", "5"], ["1"], or None for any
AVATAR_ID = None               # Specific avatar ID, or None for any
# ============================================================


def get_run_config():
    """Read random seed and combo count from data_collection/config.yaml"""
    cfg = load_app_config()
    random_seed = cfg.get("RANDOM_SEED")
    total_num_combinations = int(cfg.get("N_COMBOS", 6))
    return OUTPUT_DIR, random_seed, total_num_combinations


def parse_combinations(path):
    """Parse all_combinations.txt into list of (combo_id, params_dict)"""
    combos = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            combo_id, rest = line.split(":", 1)
            combo_id = combo_id.strip()
            params = {}
            for part in rest.split(","):
                if "=" in part:
                    key, value = part.strip().split("=", 1)
                    params[key.strip()] = value.strip()
            combos.append((combo_id, params))
    return combos


def filter_combinations(combos):
    """Filter combinations based on configuration."""
    filtered = []
    for combo_id, params in combos:
        if PROVIDER and params.get("provider") != PROVIDER:
            continue
        if LLM and params.get("llm") != LLM:
            continue
        if VOICE and params.get("voice") != VOICE:
            continue
        if SCENARIOS and params.get("scenario") not in SCENARIOS:
            continue
        if AVATAR_ID and params.get("avatar_id") != AVATAR_ID:
            continue
        filtered.append((combo_id, params))
    return filtered


def main():
    assert_collection_platform("heygen")
    run_folder, random_seed, total_num_combinations = get_run_config()

    all_combos_path = os.path.join(run_folder, "1_combinations/all_combinations.txt")
    selected_path = os.path.join(run_folder, "1_combinations/selected_combinations.txt")

    print("="*80)
    print("SELECT COMBINATIONS")
    print("="*80)
    print(f"Output directory: {run_folder}")
    print(f"Random seed: {random_seed}")
    print(f"Target count: {total_num_combinations}")
    print()

    # Parse all combinations
    all_combos = parse_combinations(all_combos_path)
    print(f"Total available: {len(all_combos)}")

    # Filter
    filtered = filter_combinations(all_combos)
    print(f"After filtering: {len(filtered)}")
    print(f"  Provider={PROVIDER}, LLM={LLM}, Voice={VOICE}, Scenarios={SCENARIOS}, Avatar={AVATAR_ID}")

    if not filtered:
        print("\nNo combinations match the filter criteria!")
        return

    # Random selection
    if random_seed is not None:
        random.seed(random_seed)

    count = min(total_num_combinations, len(filtered))
    selected = random.sample(filtered, count)

    # Sort by combo_id for readability
    selected.sort(key=lambda x: int(x[0]))

    # Write output
    with open(selected_path, "w") as f:
        for combo_id, params in selected:
            parts = [f"{k}={v}" for k, v in params.items()]
            f.write(f"{combo_id}: {', '.join(parts)}\n")

    print(f"\nSelected {len(selected)} combinations:")
    print("-"*60)
    for combo_id, params in selected:
        scenario = params.get("scenario", "?")
        avatar = params.get("avatar_id", "?")[:12]
        voice = params.get("voice", "?")
        voice_id = params.get("voice_id", "?")[:12]
        llm = params.get("llm", "?")
        print(f"  Combo {combo_id}: scenario={scenario}, avatar={avatar}..., voice={voice}, voice_id={voice_id}..., llm={llm}")

    print(f"\nSaved to: {selected_path}")
    print(f"\nNext: Run 3_generate_survey_urls.py")


if __name__ == "__main__":
    main()
