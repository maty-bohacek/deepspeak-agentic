#!/usr/bin/env python3
"""
Flask web server for on-demand HeyGen LiveAvatar conversation creation.
Creates HeyGen sessions only when users click "I'm Ready".
Handles recording uploads and S3 forwarding.
"""
import os
import re
import sys
import json
import traceback
import yaml
import threading
from datetime import datetime
import requests as http_requests
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, send_from_directory

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from heygen.heygen import HeyGenLiveAvatarWrapper
from pipeline_paths import OUTPUT_DIR as WORK_ROOT

app = Flask(__name__, template_folder="templates", static_folder="static")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)
HEYGEN_API_KEY = _cfg["HEYGEN_API_KEY"]
PUBLIC_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(PUBLIC_CONFIG_PATH, "r") as _f:
    _public_cfg = yaml.safe_load(_f) or {}
PUBLIC_DOMAIN = (_public_cfg.get("PUBLIC_DOMAIN", "INSERT_YOUR_PUBLIC_DOMAIN_HERE") or "").strip() or "INSERT_YOUR_PUBLIC_DOMAIN_HERE"

S3_BUCKET_NAME = _cfg.get("S3_BUCKET", "heygen-v3-data-storage")
S3_PREFIX = _cfg.get("S3_PREFIX", "heygen/")

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
BACKUP_DIR = os.path.join(RECORDINGS_DIR, "backup")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

_scenario_context_cache = None


def get_scenario_context_map():
    """Build a map of scenario_id -> HeyGen context_id, creating contexts as needed."""
    global _scenario_context_cache
    if _scenario_context_cache is not None:
        return _scenario_context_cache

    headers = {
        "X-API-KEY": HEYGEN_API_KEY,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    base = "https://api.liveavatar.com/v1"

    try:
        r = http_requests.get(f"{base}/contexts", headers=headers, timeout=10)
    except Exception as e:
        print(f"WARN: Failed to fetch existing contexts: {e}")
        _scenario_context_cache = {}
        return _scenario_context_cache

    existing = {}
    if r.status_code == 200:
        for ctx in r.json().get("data", {}).get("results", []):
            existing[ctx["name"]] = ctx["id"]

    scenarios = load_scenarios()
    scenario_names = {
        "1": "Casual Conversation",
        "2": "Scenario 2",
        "3": "Teaching Interview",
        "4": "Party Planning",
        "5": "Collaborative Storytelling",
    }

    mapping = {}
    for sid, sdata in scenarios.items():
        name = scenario_names.get(sid, f"Scenario {sid}")
        if name in existing:
            mapping[sid] = existing[name]
            continue
        llm_prompt = sdata.get("llm", "")
        opening = llm_prompt[:80].split("'")[1] if "'" in llm_prompt[:120] else "Hello!"
        try:
            resp = http_requests.post(
                f"{base}/contexts",
                headers=headers,
                json={"name": name, "prompt": llm_prompt, "opening_text": opening},
                timeout=10,
            )
            if resp.status_code in (200, 201):
                ctx_id = resp.json().get("data", {}).get("id")
                mapping[sid] = ctx_id
                print(f"Created context for scenario {sid}: {ctx_id}")
            else:
                print(f"WARN: Failed to create context for scenario {sid}: {resp.text[:200]}")
        except Exception as e:
            print(f"WARN: Timeout/error creating context for scenario {sid}: {e}")

    _scenario_context_cache = mapping
    print(f"Scenario->context mapping: {mapping}")
    return mapping


_cached_run_folder = None
_combo_lookup = None
_combo_lookup_path = None
_scenarios = None
_scenarios_path = None
_session_log_dir = None
_session_log_run_folder = None


def _resolve_paths():
    """Refresh derived paths if the pipeline output root changed (e.g. symlink)."""
    global _cached_run_folder, _combo_lookup, _combo_lookup_path
    global _scenarios, _scenarios_path, _session_log_dir, _session_log_run_folder
    global _scenario_context_cache

    run_folder = WORK_ROOT
    if run_folder != _cached_run_folder:
        _cached_run_folder = run_folder
        _combo_lookup_path = os.path.join(run_folder, "combo_lookup.json")
        _scenarios_path = os.path.join(run_folder, "0_scenarios/scenarios.json")
        _combo_lookup = None
        _scenarios = None
        _scenario_context_cache = None
        _session_log_dir = None
        _session_log_run_folder = None
        print(f"Pipeline output root: {run_folder} (caches cleared including contexts)")

    return run_folder


def load_combo_lookup(force=False):
    global _combo_lookup
    _resolve_paths()
    if _combo_lookup is None or force:
        with open(_combo_lookup_path, "r") as f:
            _combo_lookup = json.load(f)
        print(f"Loaded {len(_combo_lookup)} combos from {_combo_lookup_path}")
    return _combo_lookup


def load_scenarios():
    global _scenarios
    _resolve_paths()
    if _scenarios is None:
        with open(_scenarios_path, "r") as f:
            _scenarios = json.load(f)
    return _scenarios


def get_session_log_dir():
    global _session_log_dir, _session_log_run_folder
    run_folder = _resolve_paths()
    if _session_log_dir is None or _session_log_run_folder != run_folder:
        convo_logs_dir = os.path.join(run_folder, "3_output_videos/combo_url_convo_links")
        now = datetime.now()
        _session_log_dir = os.path.join(
            convo_logs_dir, now.strftime("%Y-%m-%d"), now.strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(_session_log_dir, exist_ok=True)
        _session_log_run_folder = run_folder
    return _session_log_dir


def log_conversation(combo_id, session_id, livekit_url, params):
    log_dir = get_session_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    base = f"combo_{combo_id}"
    suffix = 1
    log_file = os.path.join(log_dir, f"{base}_{suffix}.txt")
    while os.path.exists(log_file):
        suffix += 1
        log_file = os.path.join(log_dir, f"{base}_{suffix}.txt")

    ts = datetime.now().isoformat()
    start_url = f"https://{PUBLIC_DOMAIN}/start/{combo_id}"
    content = (
        f"Conversation Creation Log\n"
        f"========================\n"
        f"Timestamp: {ts}\n"
        f"Combo ID: {combo_id}\n"
        f"Start URL: {start_url}\n"
        f"Session ID: {session_id}\n"
        f"LiveKit URL: {livekit_url}\n\n"
        f"Combo Parameters:\n{json.dumps(params, indent=2)}\n"
    )
    with open(log_file, "w") as f:
        f.write(content)

    master = os.path.join(log_dir, "_master_log.txt")
    with open(master, "a") as f:
        f.write(f"{ts}\t{combo_id}\t{start_url}\t{session_id}\t{livekit_url}\n")

    print(f"Logged conversation: combo={combo_id}, session={session_id}")


def upload_to_s3(local_path, s3_key):
    """Upload a file to S3. Returns True on success."""
    try:
        import boto3
        s3 = boto3.client("s3")
        content_type = "video/webm" if s3_key.endswith(".webm") else "application/octet-stream"
        s3.upload_file(
            local_path, S3_BUCKET_NAME, s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        print(f"S3 upload OK: s3://{S3_BUCKET_NAME}/{s3_key}")
        return True
    except Exception as e:
        print(f"S3 upload FAILED for {s3_key}: {e}")
        return False


# --------------- routes ---------------

@app.route("/")
def index():
    return "HeyGen Conversation Server is running."


@app.route("/health")
def health():
    return jsonify(status="healthy", timestamp=datetime.now().isoformat())


@app.route("/start/<combo_id>")
def landing_page(combo_id):
    combos = load_combo_lookup()
    if combo_id not in combos:
        return f"Error: Combo {combo_id} not found", 404

    params = combos[combo_id]
    scenario_id = params.get("scenario", "")
    scenarios = load_scenarios()
    human_instructions = scenarios.get(scenario_id, {}).get("human", "")

    return render_template(
        "landing.html",
        combo_id=combo_id,
        human_instructions=human_instructions,
    )


@app.route("/api/create-conversation/<combo_id>", methods=["POST"])
def create_conversation(combo_id):
    try:
        combos = load_combo_lookup()
        if combo_id not in combos:
            return jsonify(error=f"Combo {combo_id} not found"), 404

        params = combos[combo_id]
        api_key = params.get("heygen_api_key", HEYGEN_API_KEY)

        heygen = HeyGenLiveAvatarWrapper(api_key=api_key, debug=True)

        avatar_id = params.get("avatar_id")
        voice_id = params.get("voice_id")
        context_id = params.get("context_id")

        if not context_id:
            scenario_id = params.get("scenario", "")
            context_map = get_scenario_context_map()
            context_id = context_map.get(scenario_id)
            if context_id:
                print(f"Resolved scenario {scenario_id} -> context_id {context_id}")
            else:
                print(f"WARN: No context found for scenario {scenario_id}")

        if voice_id and not UUID_RE.match(voice_id):
            msg = f"voice_id '{voice_id}' for combo {combo_id} is not a valid HeyGen UUID. Check agent_config.json voice IDs."
            print(f"ERROR: {msg}")
            return jsonify(error=msg), 400

        result = heygen.create_conversation(
            avatar_id=avatar_id,
            voice_id=voice_id,
            context_id=context_id if context_id else None,
            session_duration_limit=1200,
        )

        if not result.get("success"):
            err = result.get("error", "Unknown HeyGen error")
            print(f"HeyGen API error for combo {combo_id}: {err}")
            return jsonify(error=str(err)), 502

        data = result["data"]
        session_id = data["session_id"]
        livekit_url = data["livekit_url"]
        livekit_token = data["livekit_token"]

        log_conversation(combo_id, session_id, livekit_url, params)

        from urllib.parse import urlencode
        qs = urlencode({
            "session_id": session_id,
            "livekit_url": livekit_url,
            "livekit_token": livekit_token,
            "combo_id": combo_id,
        })
        redirect_url = f"/heygen/index.html?{qs}"

        return jsonify(
            success=True,
            redirect_url=redirect_url,
            session_id=session_id,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/heygen/<path:filename>")
def serve_heygen_static(filename):
    """Serve files from the heygen/ directory (index.html etc.)."""
    heygen_dir = os.path.join(PROJECT_ROOT, "heygen")
    return send_from_directory(heygen_dir, filename)


def _background_s3_upload(local_path, meta_path, s3_key, filename):
    """Upload to S3 in a background thread, then clean up local files."""
    try:
        s3_ok = upload_to_s3(local_path, s3_key)
        if s3_ok:
            os.remove(local_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            print(f"[bg] Cleaned up local file after S3 upload: {filename}")
        else:
            print(f"[bg] S3 upload failed, local file kept: {local_path}")
    except Exception as e:
        print(f"[bg] Error in background S3 upload for {filename}: {e}")


@app.route("/webhook/recording", methods=["POST"])
def webhook_recording():
    try:
        if "recording" not in request.files:
            return jsonify(error="no recording received"), 400

        file = request.files["recording"]
        session_id = request.form.get("session_id", "unknown")
        timestamp = request.form.get("timestamp", datetime.now().isoformat())
        combo_id = request.form.get("combo_id", "")

        filename = secure_filename(file.filename or f"recording_{session_id}.webm")
        local_path = os.path.join(RECORDINGS_DIR, filename)
        file.save(local_path)
        file_size = os.path.getsize(local_path)

        metadata = {
            "session_id": session_id,
            "combo_id": combo_id,
            "timestamp": timestamp,
            "filename": filename,
            "size": file_size,
            "uploaded_at": datetime.now().isoformat(),
        }
        meta_path = os.path.join(RECORDINGS_DIR, f"{filename}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Recording saved: {filename} ({file_size / 1024 / 1024:.2f} MB)")

        s3_key = f"{S3_PREFIX}{session_id}/{filename}"
        threading.Thread(
            target=_background_s3_upload,
            args=(local_path, meta_path, s3_key, filename),
            daemon=True,
        ).start()

        return jsonify(
            status="success",
            filename=filename,
            size=file_size,
            s3_key=s3_key,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(error="failed to process recording"), 500


BACKUP_MAX_MB = 2000  # keep backups under 2 GB total


def _cleanup_old_backups():
    """Delete oldest backup files if total exceeds BACKUP_MAX_MB."""
    try:
        files = []
        for f in os.listdir(BACKUP_DIR):
            fp = os.path.join(BACKUP_DIR, f)
            if os.path.isfile(fp):
                files.append((fp, os.path.getmtime(fp), os.path.getsize(fp)))
        files.sort(key=lambda x: x[1])  # oldest first
        total = sum(s for _, _, s in files)
        limit = BACKUP_MAX_MB * 1024 * 1024
        removed = 0
        while total > limit and files:
            fp, _, sz = files.pop(0)
            os.remove(fp)
            total -= sz
            removed += 1
        if removed:
            print(f"[backup cleanup] Removed {removed} old backup(s), {total / 1024 / 1024:.0f} MB remaining")
    except Exception as e:
        print(f"[backup cleanup] Error: {e}")


@app.route("/webhook/backup", methods=["POST"])
def webhook_backup():
    try:
        if "recording" not in request.files:
            return jsonify(error="no recording received"), 400

        file = request.files["recording"]
        session_id = request.form.get("session_id", "unknown")
        filename = secure_filename(file.filename or "backup.webm")
        target = os.path.join(BACKUP_DIR, filename)
        file.save(target)
        file_size = os.path.getsize(target)
        print(f"Backup: {filename} ({file_size / 1024 / 1024:.2f} MB)")

        _cleanup_old_backups()

        return jsonify(status="success", filename=filename, size=file_size)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error="failed to save backup"), 500


@app.route("/api/recordings")
def list_recordings():
    try:
        if not os.path.isdir(RECORDINGS_DIR):
            return jsonify(recordings=[])
        files = []
        for f in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
            if not f.endswith(".webm"):
                continue
            fp = os.path.join(RECORDINGS_DIR, f)
            stat = os.stat(fp)
            meta = {}
            mp = fp + ".json"
            if os.path.exists(mp):
                with open(mp) as mf:
                    meta = json.load(mf)
            files.append({**meta, "filename": f, "size": stat.st_size, "path": f"/recordings/{f}"})
        return jsonify(recordings=files)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/recordings/<filename>")
def download_recording(filename):
    return send_from_directory(RECORDINGS_DIR, secure_filename(filename))


@app.route("/api/reload-combos", methods=["POST"])
def reload_combos():
    """Force-reload combo_lookup.json without restarting."""
    _resolve_paths()
    load_combo_lookup(force=True)
    return jsonify(status="reloaded", count=len(_combo_lookup), run_folder=_cached_run_folder)


if __name__ == "__main__":
    _resolve_paths()
    print("=" * 80)
    print("Starting HeyGen Conversation Server...")
    print("=" * 80)
    print(f"Output root:    {_cached_run_folder}")
    print(f"Combo lookup:   {_combo_lookup_path}")
    print(f"Scenarios:      {_scenarios_path}")
    print(f"Recordings:     {RECORDINGS_DIR}")
    print(f"S3 bucket:      s3://{S3_BUCKET_NAME}/{S3_PREFIX}")
    print()
    print("Routes:")
    print("  GET  /start/<combo_id>")
    print("  POST /api/create-conversation/<combo_id>")
    print("  POST /webhook/recording")
    print("  POST /webhook/backup")
    print("  POST /api/reload-combos")
    print("=" * 80)

    load_combo_lookup()
    app.run(host="0.0.0.0", port=3000, debug=True)
