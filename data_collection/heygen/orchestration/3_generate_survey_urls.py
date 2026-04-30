#!/usr/bin/env python3
"""
Script 3: Generate survey URLs and update QSF template for HeyGen.
Creates /start/ URLs for on-demand conversation creation.
Does NOT create conversations - that happens on-demand via the Node server.
"""
import json
import copy
import os
import sys
import random
import string
import csv
import yaml
from typing import Any, Dict, List, Tuple
from datetime import datetime

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f) or {}
HEYGEN_API_KEY = _cfg["HEYGEN_API_KEY"]
_public_cfg = _cfg
_QUALTRICS_BLOCK_PLACEHOLDER = "PLACEHOLDER_QUALTRICS_TEMPLATE_BLOCK_ID"
QUALTRICS_TEMPLATE_BLOCK_ID = str(_cfg.get("QUALTRICS_TEMPLATE_BLOCK_ID") or "").strip()

from pipeline_paths import OUTPUT_DIR, assert_collection_platform

RUN_FOLDER = OUTPUT_DIR

# Inputs under output/ — find the most recent .qsf file in inputs
_qsf_inputs_dir = os.path.join(RUN_FOLDER, "1.5_survey_templates/inputs")
_qsf_files = sorted(
    [f for f in os.listdir(_qsf_inputs_dir) if f.endswith(".qsf")],
    key=lambda f: os.path.getmtime(os.path.join(_qsf_inputs_dir, f)),
    reverse=True,
) if os.path.isdir(_qsf_inputs_dir) else []
if not _qsf_files:
    print(f"ERROR: No .qsf files found in {_qsf_inputs_dir}")
    print("See assets/EXAMPLE_qualtrics_template/README.md")
    sys.exit(1)
INPUT_QSF = os.path.join(_qsf_inputs_dir, _qsf_files[0])
print(f"Using QSF template: {INPUT_QSF}")
COMBINATIONS_PATH = os.path.join(RUN_FOLDER, "1_combinations/selected_combinations.txt")
SCENARIOS_PATH = os.path.join(RUN_FOLDER, "0_scenarios/scenarios.json")
OUTPUTS_DIR = os.path.join(RUN_FOLDER, "1.5_survey_templates/outputs")
METADATA_DIR = os.path.join(OUTPUTS_DIR, "output_metadata")

# Web deployment configuration
WEB_DOMAIN = (_public_cfg.get("PUBLIC_DOMAIN", "INSERT_YOUR_PUBLIC_DOMAIN_HERE") or "").strip() or "INSERT_YOUR_PUBLIC_DOMAIN_HERE"
WEB_PROTOCOL = "https"

# S3 bucket for recordings (from config.yaml)
_s3_bucket = _cfg.get("S3_BUCKET", "heygen-v3-data-storage")
_s3_prefix = _cfg.get("S3_PREFIX", "heygen/").rstrip("/")
S3_BUCKET_URL = f"https://{_s3_bucket}.s3.us-east-1.amazonaws.com/{_s3_prefix}"


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def deep_replace_string(obj: Any, old_str: str, new_str: str) -> Any:
    if isinstance(obj, dict):
        return {k: deep_replace_string(v, old_str, new_str) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_replace_string(item, old_str, new_str) for item in obj]
    elif isinstance(obj, str):
        return obj.replace(old_str, new_str)
    else:
        return obj


def load_scenarios(path: str) -> Dict[str, Dict[str, str]]:
    """Load scenarios from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_combinations(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Parse selected_combinations.txt.
    Expected HeyGen format:
        1: provider=heygen, gender=female, avatar_id=..., llm=heygen-gpt-5.4,
           voice=heygen, voice_id=..., context_id=..., scenario=1
    """
    combos: List[Tuple[str, Dict[str, str]]] = []
    with open(path, "r") as f:
        for raw in f.readlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            combo_id, rest = line.split(":", 1)
            combo_id = combo_id.strip()
            params: Dict[str, str] = {}
            for part in rest.split(","):
                kv = part.strip()
                if not kv or "=" not in kv:
                    continue
                key, value = kv.split("=", 1)
                params[key.strip()] = value.strip()
            combos.append((combo_id, params))
    return combos


def normalize_combo_params(params: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize combo params to HeyGen field names.
    Handles backward-compatible Tavus-era field names:
      default-replica-id -> avatar_id
      vocal_identity (id:lang) -> voice_id
    """
    out = dict(params)
    if "avatar_id" not in out and "default-replica-id" in out:
        out["avatar_id"] = out["default-replica-id"]
    if "voice_id" not in out and "vocal_identity" in out:
        vi = out["vocal_identity"]
        out["voice_id"] = vi.split(":")[0] if ":" in vi else vi
    return out


def generate_start_urls(combos: List[Tuple[str, Dict[str, str]]]) -> List[Tuple[str, str]]:
    """
    Generate /start/ URLs for each combination.
    Returns list of (combo_id, start_url)
    """
    results = []
    for combo_id, params in combos:
        start_url = f"{WEB_PROTOCOL}://{WEB_DOMAIN}/start/{combo_id}"
        results.append((combo_id, start_url))
    return results


def find_template_block(survey_data: Dict, template_block_id: str) -> Tuple[str, Dict, int]:
    """Locate the template block in the survey."""
    for elem_idx, elem in enumerate(survey_data['SurveyElements']):
        if elem.get('Element') == 'BL':
            payload = elem.get('Payload', {})
            for key, block_data in payload.items():
                if isinstance(block_data, dict) and block_data.get('ID') == template_block_id:
                    return key, block_data, elem_idx
    return None, None, None


def get_next_available_qid(survey_data: Dict) -> int:
    """Find next available QID."""
    max_qid = 0
    for elem in survey_data['SurveyElements']:
        if elem.get('Element') == 'SQ':
            qid = elem.get('Payload', {}).get('QuestionID', '')
            if qid.startswith('QID'):
                try:
                    num = int(qid[3:])
                    max_qid = max(max_qid, num)
                except ValueError:
                    pass
    return max_qid + 1


def generate_block_id() -> str:
    chars = string.ascii_letters + string.digits
    random_part = ''.join(random.choices(chars, k=15))
    return f"BL_{random_part}"


def generate_flow_id() -> str:
    return f"FL_{random.randint(100, 999)}"


def delete_template_block(survey_data: Dict, template_block_id: str, template_key: str, bl_element_idx: int):
    """Delete template block and update counts."""
    bl_element = survey_data['SurveyElements'][bl_element_idx]
    template_block = bl_element['Payload'][template_key]

    template_question_ids: List[str] = []
    for elem in template_block.get('BlockElements', []):
        if elem.get('Type') == 'Question' and 'QuestionID' in elem:
            template_question_ids.append(elem['QuestionID'])

    survey_elements = survey_data['SurveyElements']
    for i in range(len(survey_elements) - 1, -1, -1):
        elem = survey_elements[i]
        if elem.get('Element') == 'SQ':
            qid = elem.get('Payload', {}).get('QuestionID', '')
            if qid in template_question_ids:
                del survey_elements[i]

    if template_key in bl_element['Payload']:
        del bl_element['Payload'][template_key]

    total_sq = sum(1 for elem in survey_data['SurveyElements'] if elem.get('Element') == 'SQ')
    for elem in survey_data['SurveyElements']:
        if elem.get('Element') == 'QC':
            elem['SecondaryAttribute'] = str(total_sq)
            break


def duplicate_question_with_url(survey_data: Dict, old_qid: str, new_qid: str, url_value: str, human_instructions: str = "") -> Dict:
    for elem in survey_data['SurveyElements']:
        if elem.get('Element') == 'SQ':
            if elem.get('Payload', {}).get('QuestionID') == old_qid:
                new_question = copy.deepcopy(elem)
                
                replacement_html = ""
                if human_instructions:
                    replacement_html += f"{human_instructions}<br><br>\n"
                replacement_html += f"<strong>Conversation URL:</strong><br>\n<a href=\"{url_value}\" target=\"_blank\">{url_value}</a><br>\n"
                
                new_question = deep_replace_string(new_question, 'URL: url_1', replacement_html)
                new_question = deep_replace_string(new_question, 'url_1', replacement_html)
                
                if 'DataExportTag' in new_question['Payload']:
                    old_tag = new_question['Payload']['DataExportTag']
                    new_question['Payload']['DataExportTag'] = f"{old_tag}_url"
                new_question['Payload']['QuestionID'] = new_qid
                if new_question.get('PrimaryAttribute') == old_qid:
                    new_question['PrimaryAttribute'] = new_qid
                return new_question
    return None


def build_survey_with_urls(
    input_qsf: str,
    combos: List[Tuple[str, Dict[str, str]]],
    url_results: List[Tuple[str, str]],
    scenarios: Dict[str, Dict[str, str]],
    template_block_id: str,
) -> Dict[str, Any]:
    """Build survey with /start/ URLs."""
    survey_data = read_json(input_qsf)

    combo_id_to_url: Dict[str, str] = {combo_id: url for combo_id, url in url_results}

    template_key, template_block, bl_element_idx = find_template_block(survey_data, template_block_id)
    if not template_block:
        raise RuntimeError(f"Template block {template_block_id} not found in survey")

    template_question_ids: List[str] = []
    for elem in template_block.get('BlockElements', []):
        if elem.get('Type') == 'Question' and 'QuestionID' in elem:
            template_question_ids.append(elem['QuestionID'])

    bl_element = survey_data['SurveyElements'][bl_element_idx]
    payload_keys = [int(k) for k in bl_element['Payload'].keys() if k.isdigit()]
    next_block_key = max(payload_keys) + 1 if payload_keys else 1
    next_qid_num = get_next_available_qid(survey_data)

    new_blocks: List[Tuple[str, Dict, str]] = []
    new_questions_by_block: List[List[Dict]] = []

    for i, (combo_id, params) in enumerate(combos, start=1):
        start_url = combo_id_to_url.get(combo_id, f"MISSING_URL_FOR_{combo_id}")
        
        scenario_id = params.get("scenario", "")
        human_instructions = ""
        if scenario_id in scenarios:
            human_instructions = scenarios[scenario_id].get("human", "")

        new_block_id = generate_block_id()
        new_block = copy.deepcopy(template_block)
        new_block['ID'] = new_block_id
        new_block['Description'] = f'3_url_{i}_block'

        old_to_new_qid: Dict[str, str] = {}
        block_questions: List[Dict] = []

        for old_qid in template_question_ids:
            new_qid = f"QID{next_qid_num}"
            next_qid_num += 1
            old_to_new_qid[old_qid] = new_qid

            new_question = duplicate_question_with_url(survey_data, old_qid, new_qid, start_url, human_instructions)
            if new_question:
                block_questions.append(new_question)

        for elem in new_block.get('BlockElements', []):
            if elem.get('Type') == 'Question' and 'QuestionID' in elem:
                old_qid = elem['QuestionID']
                if old_qid in old_to_new_qid:
                    elem['QuestionID'] = old_to_new_qid[old_qid]

        block_key = str(next_block_key)
        next_block_key += 1
        new_blocks.append((block_key, new_block, new_block_id))
        new_questions_by_block.append(block_questions)

    for block_key, block_data, block_id in new_blocks:
        bl_element['Payload'][block_key] = block_data

    last_sq_index = None
    for idx in range(len(survey_data['SurveyElements']) - 1, -1, -1):
        if survey_data['SurveyElements'][idx].get('Element') == 'SQ':
            last_sq_index = idx
            break
    if last_sq_index is not None:
        insert_idx = last_sq_index + 1
        for block_questions in new_questions_by_block:
            for question in block_questions:
                survey_data['SurveyElements'].insert(insert_idx, question)
                insert_idx += 1

    fl_element = None
    for idx, elem in enumerate(survey_data['SurveyElements']):
        if elem.get('Element') == 'FL':
            fl_element = elem
            break
    if fl_element:
        flow_list = fl_element['Payload']['Flow']
        for flow_idx, flow_item in enumerate(flow_list):
            if flow_item.get('Type') == 'BlockRandomizer':
                flow_item['Flow'] = []
                for _, block_data, block_id in new_blocks:
                    flow_entry = {
                        "Type": "Standard",
                        "ID": block_id,
                        "FlowID": generate_flow_id(),
                        "Autofill": []
                    }
                    flow_item['Flow'].append(flow_entry)
                flow_item['SubSet'] = 1
                break

    delete_template_block(survey_data, template_block_id, template_key, bl_element_idx)

    return survey_data


def main():
    print("="*80)
    print("SURVEY URL GENERATOR - HeyGen (On-Demand Conversations)")
    print("="*80)
    print()

    if (
        not QUALTRICS_TEMPLATE_BLOCK_ID
        or QUALTRICS_TEMPLATE_BLOCK_ID == _QUALTRICS_BLOCK_PLACEHOLDER
    ):
        print("ERROR: Set QUALTRICS_TEMPLATE_BLOCK_ID in data_collection/config.yaml.")
        print(
            "  It must match the \"ID\" of your URL template block in the QSF "
            "(see assets/EXAMPLE_qualtrics_template/README.md)."
        )
        sys.exit(1)

    assert_collection_platform("heygen")

    # Ensure all necessary directories exist
    print("Creating output directories...")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(RUN_FOLDER, "2_html_conversations"), exist_ok=True)
    os.makedirs(os.path.join(RUN_FOLDER, "3_output_videos/combo_url_convo_links"), exist_ok=True)
    
    common_html_dir = os.path.join(RUN_FOLDER, "2_html_conversations")
    os.makedirs(common_html_dir, exist_ok=True)
    
    print("Directories created\n")
    
    scenarios = load_scenarios(SCENARIOS_PATH)
    combos = parse_combinations(COMBINATIONS_PATH)
    if not combos:
        print("No combinations found; aborting.")
        return

    print(f"Found {len(combos)} combinations\n")
    
    # Generate /start/ URLs
    url_results = generate_start_urls(combos)
    
    print("Generated URLs:")
    print("-"*60)
    for combo_id, start_url in url_results:
        params = dict(combos)[combo_id] if combo_id in dict(combos) else {}
        scenario_id = params.get("scenario", "?")
        print(f"  Combo {combo_id} (scenario {scenario_id}): {start_url}")
    print()

    # Build survey with URLs
    survey_with_urls = build_survey_with_urls(
        INPUT_QSF, combos, url_results, scenarios, QUALTRICS_TEMPLATE_BLOCK_ID
    )

    # Save populated QSF
    in_base_name = os.path.basename(INPUT_QSF)
    base_name_no_ext, ext = os.path.splitext(in_base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_qsf = os.path.join(OUTPUTS_DIR, f"{base_name_no_ext}_heygen_ondemand_{timestamp}{ext}")
    write_json(output_qsf, survey_with_urls)
    size = os.path.getsize(output_qsf)
    print(f"Saved populated survey to: {output_qsf} ({size:,} bytes)")

    # Write URL mapping CSV with HeyGen-specific fields
    csv_path = os.path.join(METADATA_DIR, f"url_mapping_{timestamp}.csv")
    combo_dict = dict(combos)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "combo_id", "start_url", "provider", "scenario", "gender",
            "avatar_id", "voice_id", "voice", "llm", "context_id",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for combo_id, start_url in url_results:
            p = normalize_combo_params(combo_dict.get(combo_id, {}))
            writer.writerow({
                "combo_id": combo_id,
                "start_url": start_url,
                "provider": p.get("provider", "heygen"),
                "scenario": p.get("scenario", ""),
                "gender": p.get("gender", ""),
                "avatar_id": p.get("avatar_id", ""),
                "voice_id": p.get("voice_id", ""),
                "voice": p.get("voice", ""),
                "llm": p.get("llm", ""),
                "context_id": p.get("context_id", ""),
            })
    print(f"Saved URL mapping to: {csv_path}")
    
    # Write combo lookup JSON used by the Node server to create sessions on-demand
    combo_lookup_path = os.path.join(RUN_FOLDER, "combo_lookup.json")
    combo_lookup = {}
    for combo_id, params in combos:
        p = normalize_combo_params(params)
        scenario_id = p.get("scenario", "")
        llm_prompt = ""
        if scenario_id in scenarios:
            llm_prompt = scenarios[scenario_id].get("llm", "")
        combo_lookup[combo_id] = {
            **p,
            "llm_prompt": llm_prompt,
            "heygen_api_key": HEYGEN_API_KEY,
            "s3_bucket_url": S3_BUCKET_URL,
        }
    write_json(combo_lookup_path, combo_lookup)
    print(f"Saved combo lookup to: {combo_lookup_path}")

    print("\nReloading combo lookup on running server...")
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://127.0.0.1:3000/api/reload-combos", method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"  Server reloaded: {resp.read().decode()}")
    except Exception as e:
        print(f"  Could not reload server (may not be running): {e}")
        print("  Run: sudo systemctl restart heygen-server")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"Generated {len(url_results)} /start/ URLs")
    print(f"Survey template saved to: {output_qsf}")
    print(f"URL mapping saved to: {csv_path}")
    print(f"Combo lookup saved to: {combo_lookup_path}")
    print(f"Recording S3 target: {S3_BUCKET_URL}")
    print()
    print("Next steps:")
    print("   1. Import the QSF file into Qualtrics")
    print(f"   2. Test URLs are working: curl {WEB_PROTOCOL}://{WEB_DOMAIN}/start/<combo_id>")
    print("   3. Conversations will be created ON-DEMAND when participants visit the URL")
    print(f"   4. Recordings will be uploaded to: {S3_BUCKET_URL}")
    print("   5. Transcripts available via HeyGen API: GET /v1/sessions/<session_id>/transcript")


if __name__ == "__main__":
    main()
