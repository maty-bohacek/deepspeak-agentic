<img width="2911" height="822" alt="sample" src="https://github.com/user-attachments/assets/c029891a-f006-418d-bb35-d1d5b4205014" />
<p align="center"><i>A representative example of a conversation between an AI agent (left) and human (right) in our dataset.</i></p>

<br>

# The *DeepSpeak-Agentic* Dataset

#### [Sarah Barrington](https://www.sbarrington.com), [Maty Bohacek](https://www.matybohacek.com), and [Hany Farid](https://farid.berkeley.edu)

This repository accompanies the paper *The DeepSpeak-Agentic Dataset* and contains (1) the code used to capture real-time, audio-visual conversations between human crowd workers and embodied AI agents, and (2) the code used to compute the dataset statistics, run the deepfake detection benchmarks, and reproduce all figures and tables in the paper. The dataset we release contains 200 conversations totalling 37 hours, with isolated per-speaker streams, transcripts, and metadata.

> [Paper]() — [Data](https://huggingface.co/datasets/faridlab/deepspeak_agentic) — [Contact us](mailto:maty@stanford.edu)
>
> *Pre-print. Under review.*

## Repository Layout

- [`data_collection/`](./data_collection) <br> the video-streaming web app that pairs a Prolific worker with a fresh AI agent, records the call, and uploads it to S3
- [`experiments/`](./experiments) <br> the post-processing pipeline (diarization, speaker isolation, transcription) and the deepfake-detection / dataset-analysis benchmarks reported in the paper

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
