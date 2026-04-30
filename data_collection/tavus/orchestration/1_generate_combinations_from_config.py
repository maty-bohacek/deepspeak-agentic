#!/usr/bin/env python3
"""
Generate all possible combinations from agent_config.json
Creates combinations with real Tavus replica IDs, LLMs, voices, and scenarios
"""

import json
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline_paths import (
    OUTPUT_DIR,
    agent_config_working_path,
    assert_collection_platform,
    scenarios_working_path,
    seed_working_from_assets,
)

CONFIG_PATH = agent_config_working_path()
SCENARIOS_PATH = scenarios_working_path()
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "1_combinations", "all_combinations.txt")


def load_config(config_path: str) -> dict:
    """Load agent configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_scenarios(scenarios_path: str) -> list:
    """
    Load scenarios from scenarios.json and return list of scenario IDs.
    
    Returns:
        List of scenario IDs (as strings) from the JSON keys
    """
    with open(scenarios_path, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    
    # Return the keys (scenario IDs) as a list
    scenario_ids = list(scenarios_data.keys())
    return scenario_ids


def generate_combinations(config: dict, scenarios: list) -> list:
    """
    Generate all possible combinations from configuration.
    
    Args:
        config: Agent configuration dictionary
        scenarios: List of scenario IDs (from scenarios.json)
    
    Returns:
        List of combination dictionaries.
    """
    combinations = []
    combo_id = 1
    
    # Iterate through each provider
    for provider_name, provider_config in config.items():
        if provider_name == "scenarios":
            continue
            
        provider = provider_config.get("provider")
        llms = provider_config.get("llms", [])
        replicas_by_gender = provider_config.get("replicas", {})
        
        # Iterate through genders
        for gender, gender_config in replicas_by_gender.items():
            # Get replica_ids and voice_providers
            replica_ids = gender_config.get("replica_ids", [])
            voice_providers_config = gender_config.get("voice_providers", {})
            
            # Iterate through each replica ID
            for replica_id in replica_ids:
                # Iterate through LLMs
                for llm in llms:
                    # Iterate through voice providers
                    for voice_provider, voice_config in voice_providers_config.items():
                        vocal_identities = voice_config.get("vocal_identities", [])
                        
                        # Iterate through vocal identities
                        for vocal_identity in vocal_identities:
                            # Iterate through scenarios
                            for scenario in scenarios:
                                combination = {
                                    "combo_id": combo_id,
                                    "provider": provider,
                                    "gender": gender,
                                    "default-replica-id": replica_id,
                                    "llm": llm,
                                    "voice": voice_provider,
                                    "vocal_identity": vocal_identity,
                                    "scenario": scenario
                                }
                                combinations.append(combination)
                                combo_id += 1
    
    return combinations


def format_combination(combo: dict) -> str:
    """Format combination as a string for file output."""
    # Handle vocal_identity which can be a dict with 'id' and 'language'
    vocal_identity = combo['vocal_identity']
    if isinstance(vocal_identity, dict):
        vocal_id = vocal_identity.get('id', vocal_identity)
        language = vocal_identity.get('language', '')
        vocal_str = f"{vocal_id}:{language}" if language else vocal_id
    else:
        vocal_str = vocal_identity
    
    # Order matters for readability
    parts = [
        f"provider={combo['provider']}",
        f"gender={combo['gender']}",
        f"default-replica-id={combo['default-replica-id']}",
        f"llm={combo['llm']}",
        f"voice={combo['voice']}",
        f"vocal_identity={vocal_str}",
        f"scenario={combo['scenario']}"
    ]
    return ", ".join(parts)


def save_combinations(combinations: list, output_path: str):
    """Save combinations to file."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for combo in combinations:
            line = f"{combo['combo_id']}: {format_combination(combo)}\n"
            f.write(line)
    
    print(f"✅ Saved {len(combinations)} combinations to:")
    print(f"   {output_path}")


def print_summary(config: dict, scenarios: list, combinations: list):
    """Print summary of generated combinations."""
    print("\n" + "="*80)
    print("COMBINATION GENERATION SUMMARY")
    print("="*80)
    
    # Count by provider
    for provider_name, provider_config in config.items():
        if provider_name == "scenarios":
            continue
            
        print(f"\nProvider: {provider_name}")
        print(f"  LLMs: {len(provider_config.get('llms', []))}")
        
        replicas_by_gender = provider_config.get("replicas", {})
        for gender, gender_config in replicas_by_gender.items():
            replica_ids = gender_config.get("replica_ids", [])
            voice_providers_config = gender_config.get("voice_providers", {})
            total_vocal_ids = sum(len(vp.get("vocal_identities", [])) for vp in voice_providers_config.values())
            print(f"  {gender.capitalize()} Replicas: {len(replica_ids)}")
            print(f"    Voice Providers: {len(voice_providers_config)}")
            for vp_name, vp_config in voice_providers_config.items():
                vocal_ids = len(vp_config.get("vocal_identities", []))
                print(f"      - {vp_name}: {vocal_ids} vocal identities")
    
    print(f"\nScenarios: {len(scenarios)} {scenarios}")
    print(f"\n{'='*80}")
    print(f"TOTAL COMBINATIONS: {len(combinations)}")
    print("="*80)
    
    # Show calculation
    print("\nCalculation:")
    for provider_name, provider_config in config.items():
        if provider_name == "scenarios":
            continue
        
        llms = len(provider_config.get('llms', []))
        num_scenarios = len(scenarios)
        
        replicas_by_gender = provider_config.get("replicas", {})
        for gender, gender_config in replicas_by_gender.items():
            replica_ids = gender_config.get("replica_ids", [])
            voice_providers_config = gender_config.get("voice_providers", {})
            num_replicas = len(replica_ids)
            
            # Calculate total combinations for this gender
            gender_combos = 0
            for vp_name, vp_config in voice_providers_config.items():
                vocal_ids = len(vp_config.get("vocal_identities", []))
                vp_combos = num_replicas * llms * vocal_ids * num_scenarios
                gender_combos += vp_combos
                print(f"  {provider_name} ({gender}/{vp_name}): {num_replicas} replicas × {llms} LLMs × {vocal_ids} vocal_ids × {num_scenarios} scenarios = {vp_combos}")
            
            print(f"  {provider_name} ({gender} TOTAL): {gender_combos}")


def print_sample_combinations(combinations: list, num_samples: int = 5):
    """Print sample combinations."""
    print(f"\n{'='*80}")
    print(f"SAMPLE COMBINATIONS (first {num_samples})")
    print("="*80)
    for i, combo in enumerate(combinations[:num_samples], 1):
        print(f"{i}. {combo['combo_id']}: {format_combination(combo)}")
    print()


def main():
    print("="*80)
    print("GENERATE COMBINATIONS FROM CONFIG")
    print("="*80)
    print()
    
    assert_collection_platform("tavus")
    seed_working_from_assets()

    # Step 2: Load configuration
    print(f"Loading config from: {CONFIG_PATH}")
    config = load_config(CONFIG_PATH)
    print("✅ Config loaded\n")
    
    # Step 3: Load scenarios
    print(f"Loading scenarios from: {SCENARIOS_PATH}")
    scenarios = load_scenarios(SCENARIOS_PATH)
    print(f"✅ Loaded {len(scenarios)} scenario(s): {scenarios}\n")
    
    # Step 4: Generate combinations
    print("Generating combinations...")
    combinations = generate_combinations(config, scenarios)
    print(f"✅ Generated {len(combinations)} combinations\n")
    
    # Save to file
    save_combinations(combinations, OUTPUT_PATH)
    
    # Print summary
    print_summary(config, scenarios, combinations)
    
    # Print samples
    print_sample_combinations(combinations)
    
    print("💡 Next steps:")
    print("   1. Edit output/1_combinations/agent_config.json (source: assets/agent_config_tavus.json; COLLECTION_PLATFORM: tavus)")
    print("   2. Run this script again to regenerate combinations")
    print("   3. Manually select combinations for selected_combinations.txt")
    print("   4. Or use select_combinations.py with filters")
    print()


if __name__ == "__main__":
    main()

