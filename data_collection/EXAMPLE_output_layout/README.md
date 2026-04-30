# Example `output/` layout

All pipeline artifacts live under **`data_collection/output/`** (gitignored). On first use, scripts seed working copies from **`data_collection/assets/`**: `scenarios.json` and either **`agent_config_heygen.json`** or **`agent_config_tavus.json`** depending on **`COLLECTION_PLATFORM`** in `config.yaml` (written to `output/1_combinations/agent_config.json`).

Typical tree for the **released** orchestration steps (through survey URL / QSF generation):

```text
output/
  0_scenarios/
    scenarios.json          # working copy; edit or replace after seeding from assets/
  1_combinations/
    agent_config.json       # working copy; fill placeholders (avatars, voices, etc.)
    all_combinations.txt    # produced by step 1
    selected_combinations.txt # produced by step 2
    combo_lookup.json       # produced as part of survey URL step (HeyGen)
  1.5_survey_templates/
    inputs/                   # your .qsf (see assets/EXAMPLE_qualtrics_template/)
    outputs/                  # populated .qsf + metadata from step 3
  2_html_conversations/
  3_output_videos/
    combo_url_convo_links/
```

You can add your own folders under `output/` for exports or downstream processing; they are not created by this repo’s scripts.
