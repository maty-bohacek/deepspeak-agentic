#!/usr/bin/env python3
"""
Script 0: Import ElevenLabs voices into HeyGen.

HeyGen requires ElevenLabs voices to be registered before use:
  1. Register the ElevenLabs API key as a HeyGen secret
  2. Bind each ElevenLabs voice_id → get a HeyGen UUID
  3. Update agent_config.json with the HeyGen UUIDs

This is a one-time setup step. Safe to re-run (idempotent).
"""
import os
import sys
import json
import yaml
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

from pipeline_paths import (
    agent_config_working_path,
    assert_collection_platform,
    seed_working_from_assets,
)

assert_collection_platform("heygen")
seed_working_from_assets()
AGENT_CONFIG_PATH = agent_config_working_path()
BASE_URL = "https://api.liveavatar.com/v1"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

HEYGEN_API_KEY = cfg["HEYGEN_API_KEY"]
ELEVENLABS_API_KEY = cfg["ELEVENLABS_API_KEY"]

HEADERS = {
    "X-API-KEY": HEYGEN_API_KEY,
    "Content-Type": "application/json",
    "accept": "application/json",
}


def get_existing_secrets():
    r = requests.get(f"{BASE_URL}/secrets", headers=HEADERS)
    r.raise_for_status()
    return r.json().get("data", [])


def register_elevenlabs_secret():
    """Register ElevenLabs API key as a HeyGen secret. Returns secret_id."""
    secrets = get_existing_secrets()
    for s in secrets:
        if s["secret_type"] == "ELEVENLABS_API_KEY":
            print(f"  ElevenLabs secret already exists: {s['id']} ({s['secret_name']})")
            return s["id"]

    print("  Registering ElevenLabs API key with HeyGen...")
    r = requests.post(
        f"{BASE_URL}/secrets",
        headers=HEADERS,
        json={
            "secret_type": "ELEVENLABS_API_KEY",
            "secret_value": ELEVENLABS_API_KEY,
            "secret_name": "ElevenLabs API Key",
        },
    )
    r.raise_for_status()
    data = r.json().get("data", {})
    secret_id = data.get("id") or data.get("secret_id")
    print(f"  Registered: {secret_id}")
    return secret_id


def get_existing_voices():
    """Get all voices in HeyGen account, keyed by id."""
    voices = {}
    page = 1
    while True:
        r = requests.get(
            f"{BASE_URL}/voices", headers=HEADERS, params={"page": page, "page_size": 100}
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        results = data.get("results", [])
        for v in results:
            voices[v["id"]] = v
        if len(results) < 100:
            break
        page += 1
    return voices


def bind_elevenlabs_voice(secret_id, elevenlabs_voice_id):
    """Import an ElevenLabs voice into HeyGen. Returns the HeyGen voice UUID."""
    r = requests.post(
        f"{BASE_URL}/voices/third_party",
        headers=HEADERS,
        json={
            "secret_id": secret_id,
            "provider_voice_id": elevenlabs_voice_id,
        },
    )
    if r.status_code in (200, 201):
        data = r.json().get("data", {})
        heygen_id = data.get("id") or data.get("voice_id")
        name = data.get("name", "")
        print(f"  Imported {elevenlabs_voice_id} -> {heygen_id} ({name})")
        return heygen_id
    else:
        print(f"  FAILED to import {elevenlabs_voice_id}: {r.status_code} {r.text}")
        return None


def main():
    print("=" * 60)
    print("Import ElevenLabs Voices into HeyGen")
    print("=" * 60)

    # Step 1: Register ElevenLabs secret
    print("\nStep 1: Register ElevenLabs API key secret")
    secret_id = register_elevenlabs_secret()
    if not secret_id:
        print("ERROR: Could not register/find ElevenLabs secret")
        sys.exit(1)

    # Step 2: Collect all ElevenLabs voice IDs from agent_config
    print("\nStep 2: Scanning agent_config.json for ElevenLabs voices")
    with open(AGENT_CONFIG_PATH, "r") as f:
        agent_config = json.load(f)

    el_voice_ids = set()
    for provider_name, provider_config in agent_config.items():
        if provider_name == "scenarios":
            continue
        replicas = provider_config.get("replicas", {})
        for gender, gender_config in replicas.items():
            vps = gender_config.get("voice_providers", {})
            if "elevenlabs" in vps:
                for vi in vps["elevenlabs"].get("vocal_identities", []):
                    el_id = vi.get("elevenlabs_id", "")
                    if el_id:
                        el_voice_ids.add(el_id)

    print(f"  Found {len(el_voice_ids)} ElevenLabs voice(s): {el_voice_ids}")

    # Step 3: Check which are already imported
    print("\nStep 3: Checking existing HeyGen voices")
    existing = get_existing_voices()
    existing_names = {}
    for v in existing.values():
        name = v.get("name", "")
        if name.startswith("Imported Voice: "):
            original_el_id = name.replace("Imported Voice: ", "")
            existing_names[original_el_id] = v["id"]

    # Step 4: Import missing voices
    print("\nStep 4: Importing voices")
    el_to_heygen = {}
    for el_id in sorted(el_voice_ids):
        if el_id in existing_names:
            hg_id = existing_names[el_id]
            print(f"  {el_id} already imported as {hg_id}")
            el_to_heygen[el_id] = hg_id
            continue
        heygen_id = bind_elevenlabs_voice(secret_id, el_id)
        if heygen_id:
            el_to_heygen[el_id] = heygen_id
        else:
            el_to_heygen[el_id] = None

    # Step 5: Update agent_config.json — set id to HeyGen UUID, keep elevenlabs_id unchanged
    print("\nStep 5: Updating agent_config.json")
    updated = False
    for provider_name, provider_config in agent_config.items():
        if provider_name == "scenarios":
            continue
        replicas = provider_config.get("replicas", {})
        for gender, gender_config in replicas.items():
            vps = gender_config.get("voice_providers", {})
            if "elevenlabs" not in vps:
                continue
            for vi in vps["elevenlabs"]["vocal_identities"]:
                el_id = vi.get("elevenlabs_id", "")
                if not el_id:
                    continue
                heygen_id = el_to_heygen.get(el_id)
                if heygen_id and vi.get("id") != heygen_id:
                    print(f"  {gender}/elevenlabs: {el_id} -> id={heygen_id}")
                    vi["id"] = heygen_id
                    updated = True
                elif heygen_id and vi.get("id") == heygen_id:
                    print(f"  {gender}/elevenlabs: {el_id} (id already correct: {heygen_id})")
                else:
                    print(f"  {gender}/elevenlabs: {el_id} (FAILED to import, id unchanged)")

    if updated:
        with open(AGENT_CONFIG_PATH, "w") as f:
            json.dump(agent_config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\n  Saved updated config to {AGENT_CONFIG_PATH}")
    else:
        print("\n  No changes needed.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  ElevenLabs secret: {secret_id}")
    print(f"  Voice mappings:")
    for el_id, hg_id in sorted(el_to_heygen.items()):
        status = "OK" if hg_id else "FAILED"
        print(f"    {el_id} -> {hg_id or '???'} [{status}]")
    print(f"\n  After this, re-run script 1 to regenerate combinations.")


if __name__ == "__main__":
    main()
