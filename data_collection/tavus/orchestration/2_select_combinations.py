#!/usr/bin/env python3
"""
Script to select and filter combinations from all_combinations.txt
Saves selected combinations to selected_combinations.txt

Set TOTAL_NUM_COMBINATIONS to select a total number of combinations
that will be randomly split between male and female genders.

Just edit the settings below and run: python tavus/orchestration/2_select_combinations.py (from data_collection/)
"""

import random
import os
import sys
from typing import List, Tuple, Dict

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline_paths import OUTPUT_DIR, assert_collection_platform, load_app_config

# HTML conversations directory (output/)
COMMON_HTML_DIR = os.path.join(OUTPUT_DIR, "2_html_conversations")


def get_run_config() -> Tuple[str, int]:
    """Output directory + random seed + N_COMBOS from config.yaml"""
    cfg = load_app_config()
    random_seed = cfg.get("RANDOM_SEED")
    total_num_combinations = int(cfg.get("N_COMBOS", 6))
    return OUTPUT_DIR, random_seed, total_num_combinations


# Output directory and random seed from config
RUN_FOLDER, CONFIG_RANDOM_SEED, CONFIG_TOTAL_NUM_COMBINATIONS = get_run_config()

# ============================================================================
# CONFIGURATION - EDIT THESE SETTINGS
# ============================================================================

# How many TOTAL combinations to select (will be randomly split between genders)
TOTAL_NUM_COMBINATIONS = CONFIG_TOTAL_NUM_COMBINATIONS      # Total number of combinations (e.g., 6 might become 3 male + 3 female, or 4 male + 2 female)

# Filters (set to None to disable)
PROVIDER = "tavus"              # Options: "tavus", "anam", or None for any
LLM = None                      # Options: "tavus-gpt-4o", "tavus-llama-4", or None for any
VOICE = None                    # Options: "elevenlabs", "play", or None for any
SCENARIOS = None                # Options: ["1", "5"], ["1"], or None for any
REPLICA_ID = None               # Specific replica ID, or None for any

# Selection method
RANDOM_SELECTION = True         # True = random, False = select first N
RANDOM_SEED = CONFIG_RANDOM_SEED  # From config.yaml RANDOM_SEED (or set manually here if None)

# File paths (relative to run folder)
ALL_COMBOS_PATH = os.path.join(RUN_FOLDER, "1_combinations/all_combinations.txt")
SELECTED_COMBOS_PATH = os.path.join(RUN_FOLDER, "1_combinations/selected_combinations.txt")

# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_used_combo_ids() -> set:
    """
    Get set of combo IDs that have already been used.
    Checks for existence of conversation_combo_{id}.html files in common folder.
    
    Returns:
        Set of combo IDs (as strings) that have been used
    """
    used_combos = set()
    
    if not os.path.exists(COMMON_HTML_DIR):
        return used_combos
    
    try:
        for filename in os.listdir(COMMON_HTML_DIR):
            if filename.startswith('conversation_combo_') and filename.endswith('.html'):
                # Extract combo ID from filename like "conversation_combo_9.html"
                combo_id = filename.replace('conversation_combo_', '').replace('.html', '')
                used_combos.add(combo_id)
    except Exception as e:
        print(f"⚠️  Warning: Could not read used combos from {COMMON_HTML_DIR}: {e}")
    
    return used_combos


def extract_combo_id(line: str) -> str:
    """Extract combo_id from a combination line (e.g., '9: provider=tavus...' -> '9')"""
    if ':' in line:
        return line.split(':', 1)[0].strip()
    return ''


def parse_combination_line(line: str) -> Dict[str, str]:
    """Parse a combination line and extract all parameters."""
    # Remove combo_id prefix if present (e.g., "1: provider=tavus, ...")
    if ':' in line:
        line = line.split(':', 1)[1].strip()
    
    params = {}
    for pair in line.split(', '):
        if '=' in pair:
            key, value = pair.split('=', 1)
            params[key.strip()] = value.strip()
    return params


def load_all_combinations(file_path: str) -> List[Tuple[int, str, Dict[str, str]]]:
    """
    Load all combinations from file.
    Returns list of (line_number, original_line, parsed_params).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    result = []
    for i, line in enumerate(lines, start=1):
        line_content = line.strip()
        if not line_content:
            continue
        params = parse_combination_line(line_content)
        result.append((i, line_content, params))
    
    return result


def filter_combinations(
    combinations: List[Tuple[int, str, Dict[str, str]]],
    provider: str = None,
    gender: str = None,
    llm: str = None,
    voice: str = None,
    scenario: List[str] = None,
    replica_id: str = None
) -> List[Tuple[int, str, Dict[str, str]]]:
    """
    Filter combinations based on criteria.
    
    Args:
        combinations: List of (line_num, line_content, params)
        provider: Filter by provider (e.g., 'tavus', 'anam')
        gender: Filter by gender (e.g., 'female', 'male')
        llm: Filter by LLM (e.g., 'tavus-gpt-4o', 'tavus-llama-4')
        voice: Filter by voice (e.g., 'elevenlabs', 'play')
        scenario: Filter by scenario numbers (list of strings, e.g., ['1', '5'])
        replica_id: Filter by specific replica ID
    
    Returns:
        Filtered list of combinations
    """
    filtered = combinations
    
    if provider:
        filtered = [c for c in filtered if c[2].get('provider') == provider]
    
    if gender:
        filtered = [c for c in filtered if c[2].get('gender') == gender]
    
    if llm:
        filtered = [c for c in filtered if c[2].get('llm') == llm]
    
    if voice:
        filtered = [c for c in filtered if c[2].get('voice') == voice]
    
    if scenario:
        filtered = [c for c in filtered if c[2].get('scenario') in scenario]
    
    if replica_id:
        filtered = [c for c in filtered if c[2].get('default-replica-id') == replica_id]
    
    return filtered


def filter_out_used_combinations(
    combinations: List[Tuple[int, str, Dict[str, str]]],
    used_combo_ids: set
) -> List[Tuple[int, str, Dict[str, str]]]:
    """
    Filter out combinations that have already been used.
    
    Args:
        combinations: List of (line_num, line_content, params)
        used_combo_ids: Set of combo IDs (as strings) that have been used
    
    Returns:
        Filtered list excluding used combinations
    """
    if not used_combo_ids:
        return combinations
    
    filtered = []
    for line_num, line_content, params in combinations:
        combo_id = extract_combo_id(line_content)
        if combo_id not in used_combo_ids:
            filtered.append((line_num, line_content, params))
    
    return filtered


def select_combinations(
    combinations: List[Tuple[int, str, Dict[str, str]]],
    num: int,
    random_selection: bool = True,
    seed: int = None
) -> List[Tuple[int, str, Dict[str, str]]]:
    """
    Select N combinations from the filtered list.
    
    Args:
        combinations: List of combinations to select from
        num: Number of combinations to select
        random_selection: If True, select randomly. If False, select first N
        seed: Random seed for reproducibility
    
    Returns:
        Selected combinations
    """
    if seed is not None:
        random.seed(seed)
    
    if num >= len(combinations):
        print(f"⚠️  Requested {num} combinations but only {len(combinations)} available after filtering.")
        print(f"   Selecting all {len(combinations)} combinations.")
        return combinations
    
    if random_selection:
        return random.sample(combinations, num)
    else:
        return combinations[:num]


def save_selected_combinations(
    selected_combinations: List[Tuple[int, str, Dict[str, str]]],
    output_file: str
):
    """Save selected combinations to file with their line numbers."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line_num, line_content, params in selected_combinations:
            # line_content already includes combo ID, just save it directly
            f.write(f"{line_content}\n")
    print(f"✅ Saved {len(selected_combinations)} selected combinations to:")
    print(f"   {output_file}")


def print_summary(
    total: int,
    filtered: int,
    selected: int,
    filters_applied: Dict[str, any]
):
    """Print summary of selection process."""
    print("\n" + "="*80)
    print("SELECTION SUMMARY")
    print("="*80)
    print(f"Total combinations available:  {total}")
    print(f"After filtering:                {filtered}")
    print(f"Selected:                       {selected}")
    
    if any(filters_applied.values()):
        print("\nFilters applied:")
        for key, value in filters_applied.items():
            if value:
                print(f"  • {key}: {value}")
    else:
        print("\nNo filters applied (random selection from all)")
    print("="*80)


def main():
    assert_collection_platform("tavus")
    print("="*80)
    print("COMBINATION SELECTOR")
    print("="*80)
    print()
    
    # Set random seed early for reproducible gender split
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    print("Configuration:")
    print(f"  Total combinations: {TOTAL_NUM_COMBINATIONS}")
    print(f"  Provider: {PROVIDER or 'Any'}")
    print(f"  LLM: {LLM or 'Any'}")
    print(f"  Voice: {VOICE or 'Any'}")
    print(f"  Scenarios: {SCENARIOS or 'Any'}")
    print(f"  Random: {RANDOM_SELECTION}")
    print(f"  Random seed: {RANDOM_SEED} (from config.yaml)")
    print()
    
    # Load all combinations
    print(f"Loading combinations from: {ALL_COMBOS_PATH}")
    all_combinations = load_all_combinations(ALL_COMBOS_PATH)
    print(f"✅ Loaded {len(all_combinations)} total combinations\n")
    
    # Check for previously used combinations
    print(f"Checking for previously used combinations in: {COMMON_HTML_DIR}")
    used_combo_ids = get_used_combo_ids()
    if used_combo_ids:
        print(f"⚠️  Found {len(used_combo_ids)} previously used combo(s): {sorted(used_combo_ids, key=int)}")
        print(f"   These will be excluded from selection\n")
    else:
        print(f"✅ No previously used combinations found\n")
    
    # Filter female combinations
    print(f"Filtering FEMALE combinations...")
    female_filtered = filter_combinations(
        all_combinations,
        provider=PROVIDER,
        gender="female",
        llm=LLM,
        voice=VOICE,
        scenario=SCENARIOS,
        replica_id=REPLICA_ID
    )
    female_before_used_filter = len(female_filtered)
    female_filtered_with_exclusion = filter_out_used_combinations(female_filtered, used_combo_ids)
    female_excluded = female_before_used_filter - len(female_filtered_with_exclusion)
    
    print(f"  {female_before_used_filter} female combinations after criteria filtering")
    if female_excluded > 0:
        print(f"  {female_excluded} excluded (already used)")
    print(f"  {len(female_filtered_with_exclusion)} female combinations available for selection")
    
    # Filter male combinations
    print(f"\nFiltering MALE combinations...")
    male_filtered = filter_combinations(
        all_combinations,
        provider=PROVIDER,
        gender="male",
        llm=LLM,
        voice=VOICE,
        scenario=SCENARIOS,
        replica_id=REPLICA_ID
    )
    male_before_used_filter = len(male_filtered)
    male_filtered_with_exclusion = filter_out_used_combinations(male_filtered, used_combo_ids)
    male_excluded = male_before_used_filter - len(male_filtered_with_exclusion)
    
    print(f"  {male_before_used_filter} male combinations after criteria filtering")
    if male_excluded > 0:
        print(f"  {male_excluded} excluded (already used)")
    print(f"  {len(male_filtered_with_exclusion)} male combinations available for selection\n")
    
    # Check if we have enough unique combos
    total_unique_available = len(female_filtered_with_exclusion) + len(male_filtered_with_exclusion)
    total_with_reuse = female_before_used_filter + male_before_used_filter
    
    if total_unique_available < TOTAL_NUM_COMBINATIONS and len(used_combo_ids) > 0:
        # Not enough unique combos to meet the requested amount
        print("\n" + "="*80)
        print("⚠️  WARNING: NOT ENOUGH UNIQUE COMBOS")
        print("="*80)
        print(f"Requested: {TOTAL_NUM_COMBINATIONS} combinations")
        print(f"Unique available: {total_unique_available} combinations")
        print(f"Total available (with reuse): {total_with_reuse} combinations")
        print(f"Previously used combos: {sorted(used_combo_ids, key=int)}\n")
        
        if total_unique_available == 0:
            print(f"All {total_with_reuse} combinations matching your criteria have already been used.")
        else:
            print(f"Only {total_unique_available} unique combinations available (need {TOTAL_NUM_COMBINATIONS - total_unique_available} more).")
        
        user_input = input("\nOK to reuse combinations? (y/n): ").strip().lower()
        
        if user_input == 'y':
            print("✅ Proceeding with reused combinations...\n")
            # Use the filtered lists WITHOUT exclusion (allow reuse)
            female_filtered = female_filtered
            male_filtered = male_filtered
        else:
            print(f"❌ Aborted by user. Only {total_unique_available} unique combinations available.")
            return
    elif len(female_filtered_with_exclusion) == 0 and len(male_filtered_with_exclusion) == 0:
        # No combinations found at all
        print("  ❌ No combinations found after filtering!")
        return
    else:
        # Use the filtered lists WITH exclusion (enough unique combos available)
        female_filtered = female_filtered_with_exclusion
        male_filtered = male_filtered_with_exclusion
    
    # Randomly split total between genders
    # Ensure at least 1 of each gender if both are available
    max_female = min(len(female_filtered), TOTAL_NUM_COMBINATIONS - 1 if len(male_filtered) > 0 else TOTAL_NUM_COMBINATIONS)
    max_male = min(len(male_filtered), TOTAL_NUM_COMBINATIONS - 1 if len(female_filtered) > 0 else TOTAL_NUM_COMBINATIONS)
    
    if len(female_filtered) == 0:
        num_female = 0
        num_male = min(TOTAL_NUM_COMBINATIONS, len(male_filtered))
    elif len(male_filtered) == 0:
        num_female = min(TOTAL_NUM_COMBINATIONS, len(female_filtered))
        num_male = 0
    else:
        # Both genders available - aim for roughly equal split (50/50)
        num_female = TOTAL_NUM_COMBINATIONS // 2
        num_male = TOTAL_NUM_COMBINATIONS - num_female  # Gets the extra 1 if odd total
        
        # Adjust if we don't have enough of one gender
        if num_female > len(female_filtered):
            num_female = len(female_filtered)
            num_male = min(TOTAL_NUM_COMBINATIONS - num_female, len(male_filtered))
        elif num_male > len(male_filtered):
            num_male = len(male_filtered)
            num_female = min(TOTAL_NUM_COMBINATIONS - num_male, len(female_filtered))
    
    print(f"Equal split: {num_female} female + {num_male} male = {num_female + num_male} total\n")
    
    # Select female combinations
    female_selected = []
    if num_female > 0:
        print(f"Selecting {num_female} FEMALE combinations...")
        female_selected = select_combinations(
            female_filtered,
            num_female,
            random_selection=RANDOM_SELECTION,
            seed=RANDOM_SEED if RANDOM_SELECTION else None
        )
        print(f"  ✅ Selected {len(female_selected)} female combinations\n")
    
    # Select male combinations
    male_selected = []
    if num_male > 0:
        print(f"Selecting {num_male} MALE combinations...")
        male_selected = select_combinations(
            male_filtered,
            num_male,
            random_selection=RANDOM_SELECTION,
            seed=RANDOM_SEED + 1 if RANDOM_SELECTION and RANDOM_SEED is not None else None
        )
        print(f"  ✅ Selected {len(male_selected)} male combinations\n")
    
    # Combine selections
    all_selected = female_selected + male_selected
    
    # Print selected combinations
    print("="*80)
    print("SELECTED COMBINATIONS")
    print("="*80)
    print(f"\nFEMALE ({len(female_selected)}):")
    print("-" * 80)
    for i, (line_num, line_content, params) in enumerate(female_selected, 1):
        print(f"{i}. Line {line_num}: {line_content}")
    
    print(f"\nMALE ({len(male_selected)}):")
    print("-" * 80)
    for i, (line_num, line_content, params) in enumerate(male_selected, 1):
        print(f"{i}. Line {line_num}: {line_content}")
    print("-" * 80)
    
    # Save to file
    print()
    save_selected_combinations(all_selected, SELECTED_COMBOS_PATH)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total combinations available: {len(all_combinations)}")
    if used_combo_ids:
        print(f"Previously used (excluded): {len(used_combo_ids)}")
    print(f"Female available after filtering: {len(female_filtered)}")
    print(f"Male available after filtering: {len(male_filtered)}")
    print(f"\nEqual split:")
    print(f"  Female selected: {len(female_selected)}")
    print(f"  Male selected: {len(male_selected)}")
    print(f"  TOTAL SELECTED: {len(all_selected)}")
    print("="*80)
    
    print("\n💡 Next step: Run 3_custom_wrapper.py to generate conversations with these combinations")
    print()


if __name__ == "__main__":
    main()

