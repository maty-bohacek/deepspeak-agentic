<!-- Banner image goes here once available -->

# The DeepSpeak-Agentic Dataset

#### [Author One](https://example.com), [Author Two](https://example.com), [Author Three](https://example.com), [Author Four](https://example.com)

This repository accompanies the paper *The DeepSpeak-Agentic Dataset* and contains (1) the code used to capture real-time, audio-visual conversations between human crowd workers and embodied AI agents, and (2) the code used to compute the dataset statistics, run the deepfake detection benchmarks, and reproduce all figures and tables in the paper.

The dataset itself — 200 conversations totalling 37 hours, with isolated per-speaker streams, transcripts, and metadata — is hosted on HuggingFace: [`faridlab/deepspeak_agentic`](https://huggingface.co/datasets/faridlab/deepspeak_agentic).

> [Website](https://example.com) — [Paper](https://example.com) — [Contact us](mailto:email@example.com)
>
> *Pre-print. Under review.*

## Repository Layout

- [`data_collection/`](./data_collection) — the video-streaming web app that pairs a Prolific worker with a fresh AI agent, records the call, and uploads it to S3.
- [`experiments/`](./experiments) — the post-processing pipeline (diarization, speaker isolation, transcription) and the deepfake-detection / dataset-analysis benchmarks reported in the paper.

## Getting Started

### Experiments

```bash
cd experiments
pip install -r requirements.txt
export HF_TOKEN=...           # HuggingFace token with read access to the dataset
export OPENAI_API_KEY=...     # only needed for the LLM analysis step
python run_pipeline.py        # runs all steps in order; --help for individual steps
```

Outputs land in `data/` (intermediate artifacts) and `output/` (statistics, plots, detection results).

To regenerate Table 1 (detection accuracy by engine) after running the `detect` step:

```bash
python make_detection_table.py > detection_table.tex
```

### Data Collection

*Setup and run instructions coming soon.*

```bash
cd data_collection
# TODO: install instructions
# TODO: how to configure the agent providers (Tavus, HeyGen, ElevenLabs, Cartesia)
# TODO: how to launch the video-streaming server
# TODO: how to issue per-participant session URLs
```

## Citation

If you find our work useful, please consider citing our paper.

```bibtex
@inproceedings{tbd2026deepspeakagentic,
  title={The DeepSpeak-Agentic Dataset},
  author={TBD},
  booktitle={TBD},
  year={TBD}
}
```
