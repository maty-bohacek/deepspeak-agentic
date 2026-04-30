# Data Collection

Orchestration scripts, **HeyGen** and **Tavus** API wrappers, and **Flask** web servers that issue session URLs (with optional **S3** recording uploads). Participant-facing surveys use **Qualtrics**; this repo does **not** design the survey UI for you—it fills exported survey templates with conversation URLs.

## Layout

| Path | Role |
|------|------|
| `assets/` | Templates: `agent_config_heygen.json`, `agent_config_tavus.json`, and `scenarios.json` (placeholders; safe to commit). Which agent file is copied into `output/1_combinations/agent_config.json` is controlled by **`COLLECTION_PLATFORM`** in `config.yaml`. |
| `output/` | Generated artifacts for the released pipeline: combinations, survey templates, HTML, conversation logs (gitignored). |
| `config.yaml` | API keys, bucket names, `PUBLIC_DOMAIN`, `COLLECTION_PLATFORM` (`heygen` or `tavus`), `QUALTRICS_TEMPLATE_BLOCK_ID`, pipeline knobs (`RANDOM_SEED`, `N_COMBOS`). See the file for full keys. |
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

Set project-wide settings in `data_collection/config.yaml` (keys vary by provider; at minimum set **`COLLECTION_PLATFORM`**, **`PUBLIC_DOMAIN`**, provider API keys, and for survey generation **`QUALTRICS_TEMPLATE_BLOCK_ID`**).

Example snippet:

```yaml
COLLECTION_PLATFORM: heygen   # or tavus — must match the orchestration tree you use
PUBLIC_DOMAIN: "your-domain.example"
QUALTRICS_TEMPLATE_BLOCK_ID: "BL_your_template_block_from_qsf"
# Tavus recording / S3 (when using Tavus): see config.yaml for TAVUS_* keys
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

Put secrets in **`config.yaml`** or your environment; **do not commit** populated files.

Typical values include provider API keys (**HeyGen**, **Tavus**, **ElevenLabs** where used), **AWS** credentials if you upload recordings to **S3**, and similar keys listed in **`config.yaml`**. This README does not enumerate every export—copy from `config.yaml` comments and wrapper code as needed.

Wrappers and orchestration scripts read **`data_collection/config.yaml`**.

## Web servers

- **HeyGen** (port 3000 by default): from `data_collection/`, run `python heygen/web_server/app.py` — reads `output/combo_lookup.json`, `output/0_scenarios/scenarios.json`, and related paths (same root as orchestration).
- **Tavus** (port 5000): from `data_collection/`, run `python tavus/web_server/app.py`, or `bash tavus/web_server/run.sh` (optional `venv` under `data_collection/`).

Production: use `gunicorn` (included in `requirements.txt`); see comments in `tavus/web_server/run.sh`. Terminate TLS and map your **`PUBLIC_DOMAIN`** to this service at your infrastructure layer.

### HeyGen: reloading the combo list after step 3

`heygen/orchestration/3_generate_survey_urls.py` finishes by **POSTing to** `http://127.0.0.1:3000/api/reload-combos` so a **local** HeyGen Flask server (default port 3000) picks up the new `output/combo_lookup.json` without a restart. If nothing is listening, the script prints a warning and continues. In production, restart your process manager or reload combos by hand after deploying the updated JSON.

## Scenarios

Four scenarios are supported: `conversation`, `professional`, `collaborative_planning`, `creative`. Prompts are described in the paper appendix.
