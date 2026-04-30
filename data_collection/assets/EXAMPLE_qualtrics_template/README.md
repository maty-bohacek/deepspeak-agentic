# Qualtrics survey template (QSF)

**We do not ship a `.qsf` file in this repository.** You must export your own Qualtrics survey and place it where the orchestration scripts expect it.

## Where to put your file

Under **`data_collection/output/`** (the working directory for both HeyGen and Tavus orchestration):

```text
output/1.5_survey_templates/inputs/
```

Put one or more `.qsf` files there:

- **HeyGen** (`heygen/orchestration/3_generate_survey_urls.py`): uses the **most recently modified** `.qsf` in that folder.
- **Tavus** (`tavus/orchestration/3_generate_survey_urls.py`): same rule (newest `.qsf` in that folder).

Exported Qualtrics surveys are JSON; the scripts read them as JSON.

## What the survey must contain for URL injection to work

The generator duplicates a **template block** in the QSF and replaces URL placeholders with each participant’s conversation link (`https://<PUBLIC_DOMAIN>/start/<combo_id>`).

1. **Template block ID (must match `config.yaml`)**  
   The generator looks for **one** block inside a survey **block list** (`Element`: `BL`) whose inner block object has an `"ID"` field (Qualtrics exports look like `"BL_…"`). That block is duplicated once per combo URL.

   - Export your survey as `.qsf` (JSON) and search for the block you use as the URL template (the same block whose questions contain `URL: url_1` / `url_1`).
   - Copy that block’s `"ID"` string exactly (e.g. `"BL_abc123xyz"`).
   - Set **`QUALTRICS_TEMPLATE_BLOCK_ID`** in `data_collection/config.yaml` to that value (not the shipped placeholder `PLACEHOLDER_QUALTRICS_TEMPLATE_BLOCK_ID`).

   Both `heygen/orchestration/3_generate_survey_urls.py` and `tavus/orchestration/3_generate_survey_urls.py` read this key. If it is missing or still the placeholder, step 3 exits with an error.

2. **URL placeholders inside the template question text**  
   Inside the question(s) in that template block, the question text must contain **both** of these literals (as plain text), which the script replaces with HTML containing the real URL:

   - `URL: url_1`
   - `url_1`

   (The script runs two replacements; having both ensures older and newer question wordings still match.)

3. **After generation**  
   The populated survey is written under:

   ```text
   output/1.5_survey_templates/outputs/
   ```

   **You must bring that file back into Qualtrics.** The pipeline only edits a local `.qsf`; your live study in Qualtrics does not change until you **import** (or re-upload / replace via your org’s workflow) the **output** `.qsf` so Qualtrics has the new blocks and embedded `/start/` URLs. Until you do that, participants still see the old survey. After import, open the survey in Qualtrics and sanity-check links and flow before launching.

## Optional: participant identifiers

If you will match Qualtrics rows to sessions or recordings in your own tooling, add **text-entry** questions that collect identifiers you need (for example **session ID** and **Prolific ID**). Wording is up to you unless you reuse automation that searches for specific phrases.

## Related files (not the QSF)

- **`0_scenarios/scenarios.json`** — **Required** for combo generation, survey URL script, and both Flask servers (human/LLM text per scenario). Templates live in `assets/`; working copies are under `output/0_scenarios/`. See `../../EXAMPLE_output_layout/README.md` for the full tree.
- **`1_combinations/agent_config.json`** — **Required** working copy (seeded from `assets/agent_config_heygen.json` or `assets/agent_config_tavus.json` per `COLLECTION_PLATFORM`). Used by `heygen/orchestration/1_generate_combinations_from_config.py` or `tavus/orchestration/1_generate_combinations_from_config.py`; HeyGen voice import (`heygen/orchestration/0_import_elevenlabs_voices.py`) updates this file when `COLLECTION_PLATFORM: heygen`.
