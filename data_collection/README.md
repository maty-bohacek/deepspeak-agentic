# Data Collection

Orchestration scripts, **HeyGen** and **Tavus** API wrappers, and **Flask** web servers that issue session URLs (with optional **S3** recording uploads). Participant-facing surveys use **Qualtrics**; this repo does **not** design the survey UI for you—it fills exported survey templates with conversation URLs.

## Layout

| Path | Role |
|------|------|
| `assets/` | Templates: `agent_config_heygen.json`, `agent_config_tavus.json`, and `scenarios.json` (placeholders; safe to commit). Which agent file is copied into `output/1_combinations/agent_config.json` is controlled by **`COLLECTION_PLATFORM`** in `config.yaml`. |
| `output/` | Generated artifacts for the released pipeline: combinations, survey templates, HTML, conversation logs (gitignored). |
| `config.yaml` | Non-secret settings: `PUBLIC_DOMAIN`, `COLLECTION_PLATFORM`, `HEYGEN_S3_*`, `TAVUS_S3_*`, `QUALTRICS_TEMPLATE_BLOCK_ID`, `RANDOM_SEED`, `N_COMBOS`. **API keys are not stored here** — use environment variables (see **Credentials**). |
| `heygen/orchestration/` | Numbered HeyGen pipeline scripts (`0_import_elevenlabs_voices.py`, `1_…`, `2_…`, `3_…`). |
| `heygen/web_server/` | HeyGen Flask app (`app.py`) for on-demand LiveAvatar sessions. |
| `heygen/heygen.py`, `heygen/index.html` | LiveAvatar API wrapper and **browser client** (served at `/heygen/index.html` by the HeyGen server). |
| `tavus/orchestration/` | Numbered Tavus pipeline scripts (`1_…`, `2_…`, `3_…`). |
| `tavus/web_server/` | Tavus Flask app (`app.py`, `run.sh`). |
| `tavus/tavus.py` | Tavus CVI API wrapper. |
| `assets/EXAMPLE_qualtrics_template/` | Documents required **placeholder text** and structure inside your Qualtrics survey export (`.qsf`). **No survey file is shipped** in this repo. |
| `EXAMPLE_output_layout/` | Documents the directory tree under `output/`. |

Run orchestration from `data_collection/` (so imports resolve), for example:

```bash
cd data_collection
python heygen/orchestration/1_generate_combinations_from_config.py
python tavus/orchestration/1_generate_combinations_from_config.py
```

## Install

All Python dependencies for this folder live in **`requirements.txt` here** (separate from `experiments/requirements.txt`).

```bash
cd data_collection
pip install -r requirements.txt
```

Set non-secret values in `data_collection/config.yaml` (at minimum **`COLLECTION_PLATFORM`**, **`PUBLIC_DOMAIN`**, **`QUALTRICS_TEMPLATE_BLOCK_ID`**). Export **API keys** in your shell (see **Credentials**).

Example snippet:

```yaml
COLLECTION_PLATFORM: heygen   # or tavus — must match the orchestration tree you use
PUBLIC_DOMAIN: "your-domain.example"
QUALTRICS_TEMPLATE_BLOCK_ID: "BL_your_template_block_from_qsf"
HEYGEN_S3_BUCKET: "heygen-v3-data-storage"
HEYGEN_S3_PREFIX: "heygen/"
HEYGEN_S3_REGION: "us-east-1"
# Tavus: TAVUS_S3_BUCKET, TAVUS_S3_REGION, TAVUS_AWS_ASSUME_ROLE_ARN in config.yaml
```

`COLLECTION_PLATFORM` must be **`heygen`** when using `heygen/orchestration/*` and **`tavus`** when using `tavus/orchestration/*`; scripts exit if it does not match. Switching platforms re-seeds `output/1_combinations/agent_config.json` from the matching asset file (see `pipeline_paths.seed_working_from_assets`).

## Qualtrics: design separately, export, populate here, re-upload

This codebase **does not** create or host Qualtrics surveys via API. You maintain the survey in Qualtrics yourself.

1. **Design** a survey in Qualtrics that includes a **template block** with the literal placeholders this tooling replaces (`URL: url_1` and `url_1` in question text). Structure requirements are in **`assets/EXAMPLE_qualtrics_template/README.md`**.
2. **Export** that survey as a **`.qsf`** file from Qualtrics.
3. **Copy** the `.qsf` into **`output/1.5_survey_templates/inputs/`** (create folders if needed). Step **3** (`heygen/orchestration/3_generate_survey_urls.py` or `tavus/orchestration/3_generate_survey_urls.py`) picks the newest `.qsf` there.
4. Set **`QUALTRICS_TEMPLATE_BLOCK_ID`** in **`config.yaml`** to the **`"ID"`** of that template block inside the exported JSON (not the shipped placeholder).
5. Run the pipeline through combinations → selection → step **3**. The script writes a **populated** `.qsf` under **`output/1.5_survey_templates/outputs/`** with real `/start/<combo_id>` URLs for your **`PUBLIC_DOMAIN`**.
6. **Re-import** that output `.qsf` into Qualtrics so the live study matches the generated links. Until you import, participants still see the old survey.

Crowdsourcing platforms (e.g. routing workers into Qualtrics) are **outside** this repo—you configure those tools to send participants to your Qualtrics URL once the imported survey is active.

### Provider concurrency and how many URLs to generate

At the time this was documented, **HeyGen (LiveAvatar)** and **Tavus** each imposed on the order of **roughly 15–30 concurrent** real-time sessions (limits vary by plan and change over time—confirm in your vendor dashboard). That ceiling matters for how you run a study: step 3 generates one `/start/<combo_id>` URL per selected combination (`N_COMBOS` in `config.yaml` and your selection logic). If many participants are in the conversation phase at once, active sessions can approach that concurrency bound. Size **`N_COMBOS`**, recruitment, and survey flow so you do not expect more simultaneous conversations than your provider allows, or stagger cohorts accordingly.

## Credentials

**API keys are read from the environment only** (not from `config.yaml`). Export them before running servers or scripts that call the APIs, for example:

```bash
export HEYGEN_API_KEY=...       # HeyGen LiveAvatar + web server + orchestration step 3
export TAVUS_API_KEY=...        # Tavus API wrapper
export ELEVENLABS_API_KEY=...    # heygen/orchestration/0_import_elevenlabs_voices.py
```

**AWS** credentials for S3 uploads use the usual environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, etc.) or your runtime’s IAM role.

Non-secret paths and flags stay in **`data_collection/config.yaml`**; scripts load that file for `PUBLIC_DOMAIN`, S3 bucket **names** (`HEYGEN_S3_*`, `TAVUS_S3_*`), and pipeline settings.

## Web servers

- **HeyGen** (port 3000 by default): from `data_collection/`, run `python heygen/web_server/app.py` — reads `output/combo_lookup.json`, `output/0_scenarios/scenarios.json`, and related paths (same root as orchestration).
- **Tavus** (port 5000): from `data_collection/`, run `python tavus/web_server/app.py`, or `bash tavus/web_server/run.sh` (optional `venv` under `data_collection/`).

Production: use `gunicorn` (included in `requirements.txt`); see comments in `tavus/web_server/run.sh`. Terminate TLS and map your **`PUBLIC_DOMAIN`** to this service at your infrastructure layer.

### HeyGen: reloading the combo list after step 3

`heygen/orchestration/3_generate_survey_urls.py` finishes by **POSTing to** `http://127.0.0.1:3000/api/reload-combos` so a **local** HeyGen Flask server (default port 3000) picks up the new `output/combo_lookup.json` without a restart. If nothing is listening, the script prints a warning and continues. In production, restart your process manager or reload combos by hand after deploying the updated JSON.

## Scenarios

Four scenarios are supported: `conversation`, `professional`, `collaborative_planning`, `creative`. Prompts are described in the paper appendix.
