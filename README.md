<img width="2911" height="822" alt="sample" src="https://github.com/user-attachments/assets/c029891a-f006-418d-bb35-d1d5b4205014" />
<p align="center"><i>A representative example of a conversation between an AI agent (left) and human (right) in our dataset.</i></p>

# The *DeepSpeak-Agentic* Dataset

#### [Sarah Barrington](https://www.sbarrington.com), [Maty Bohacek](https://www.matybohacek.com), and [Hany Farid](https://farid.berkeley.edu)

This repository accompanies the paper *The DeepSpeak-Agentic Dataset* and contains (1) the code used to capture real-time, audio-visual conversations between human crowd workers and embodied AI agents, and (2) the code used to compute the dataset statistics, run the deepfake detection benchmarks, and reproduce all figures and tables in the paper. The dataset we release contains 200 conversations totalling 37 hours, with isolated per-speaker streams, transcripts, and metadata.

> [Paper](https://arxiv.org/abs/2606.03686) — [Data](https://huggingface.co/datasets/faridlab/deepspeak_agentic) — [Contact us](mailto:maty@stanford.edu)
>
> *Published at the WACV 2026 SIGNAL Workshop.*

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

```bash
cd data_collection
pip install -r requirements.txt
```

Orchestration, Tavus/HeyGen wrappers, and Flask recording servers live under `data_collection/` (per-provider trees: `data_collection/heygen/`, `data_collection/tavus/`); see [`data_collection/README.md`](./data_collection/README.md) for layout, environment variables, and how to run the web servers.

## Citation

If you find our work useful, please consider citing our paper.

```bibtex
@article{barrington2026deepspeak,
  title={The DeepSpeak-Agentic Dataset},
  author={Barrington, Sarah and Bohacek, Maty and Farid, Hany},
  journal={arXiv preprint arXiv:2606.03686},
  year={2026}
}
```
