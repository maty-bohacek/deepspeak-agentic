"""
Layout for data_collection:

- assets/     Template agent_config_{heygen|tavus}.json + scenarios.json (placeholders).
- heygen/, tavus/  Provider code: orchestration/, web_server/, API wrappers.
- output/     All generated and runtime artifacts (gitignored by default).
"""
from __future__ import annotations

import json
import os
import shutil
import sys

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))

ASSETS_DIR = os.path.join(PACKAGE_ROOT, "assets")
OUTPUT_DIR = os.path.join(PACKAGE_ROOT, "output")

VALID_COLLECTION_PLATFORMS = frozenset({"heygen", "tavus"})

SCENARIOS_ASSET = os.path.join(ASSETS_DIR, "scenarios.json")


def collection_platform(app_cfg: dict | None = None) -> str:
    """Read config.yaml COLLECTION_PLATFORM: heygen | tavus (default heygen)."""
    cfg = load_app_config() if app_cfg is None else app_cfg
    raw = cfg.get("COLLECTION_PLATFORM", "heygen")
    p = str(raw).strip().lower()
    if p not in VALID_COLLECTION_PLATFORMS:
        raise ValueError(
            f"COLLECTION_PLATFORM must be one of {sorted(VALID_COLLECTION_PLATFORMS)}, got {raw!r}"
        )
    return p


def agent_config_asset_path(app_cfg: dict | None = None) -> str:
    plat = collection_platform(app_cfg)
    return os.path.join(ASSETS_DIR, f"agent_config_{plat}.json")


def agent_config_platform_marker_path() -> str:
    return os.path.join(OUTPUT_DIR, "1_combinations", ".agent_config_platform")


def infer_agent_workspace_platform(agent_json_path: str) -> str | None:
    """Detect heygen vs tavus from working agent_config.json top-level key (if present)."""
    try:
        with open(agent_json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        if "tavus" in data:
            return "tavus"
        if "heygen" in data:
            return "heygen"
    except (OSError, json.JSONDecodeError):
        pass
    return None


def agent_config_working_path() -> str:
    return os.path.join(OUTPUT_DIR, "1_combinations", "agent_config.json")


def scenarios_working_path() -> str:
    return os.path.join(OUTPUT_DIR, "0_scenarios", "scenarios.json")


def assert_collection_platform(expected: str) -> None:
    """Exit with a clear message if config.yaml platform does not match this orchestration tree."""
    if expected not in VALID_COLLECTION_PLATFORMS:
        raise ValueError(f"expected must be one of {sorted(VALID_COLLECTION_PLATFORMS)}")
    got = collection_platform()
    if got != expected:
        print(
            f"ERROR: This script is for {expected!r}. "
            f"Set COLLECTION_PLATFORM: {expected!r} in data_collection/config.yaml "
            f"(currently {got!r}).",
            file=sys.stderr,
        )
        sys.exit(1)


def ensure_output_dirs() -> None:
    os.makedirs(os.path.join(OUTPUT_DIR, "0_scenarios"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1_combinations"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1.5_survey_templates", "inputs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1.5_survey_templates", "outputs"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "1.5_survey_templates", "outputs", "output_metadata"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "2_html_conversations"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "3_output_videos", "combo_url_convo_links"), exist_ok=True)


def seed_working_from_assets() -> None:
    """
    Copy assets → output working copies. Scenarios: copy only if missing.
    Agent config: copy from assets/agent_config_{COLLECTION_PLATFORM}.json if the
    working file is missing, or if COLLECTION_PLATFORM changed since last seed
    (tracked via output/1_combinations/.agent_config_platform).
    """
    ensure_output_dirs()
    cfg = load_app_config()
    plat = collection_platform(cfg)
    agent_asset = agent_config_asset_path(cfg)
    if not os.path.isfile(agent_asset):
        raise FileNotFoundError(
            f"Missing template for COLLECTION_PLATFORM={plat!r}: {agent_asset}"
        )
    if not os.path.isfile(SCENARIOS_ASSET):
        raise FileNotFoundError(f"Missing template: {SCENARIOS_ASSET}")
    aw = agent_config_working_path()
    sw = scenarios_working_path()
    marker = agent_config_platform_marker_path()
    prev = None
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8") as f:
            prev = f.read().strip() or None
    elif os.path.isfile(aw):
        prev = infer_agent_workspace_platform(aw)

    need_agent_copy = not os.path.isfile(aw)
    if prev is not None and prev != plat:
        need_agent_copy = True

    if need_agent_copy:
        shutil.copy2(agent_asset, aw)
    if not os.path.isfile(sw):
        shutil.copy2(SCENARIOS_ASSET, sw)

    with open(marker, "w", encoding="utf-8") as f:
        f.write(plat)


def load_app_config() -> dict:
    import yaml

    path = os.path.join(PACKAGE_ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
