# Data Collection

Video-streaming web application that pairs a consenting human participant (recruited via Prolific) with a freshly instantiated AI agent, records the audio-visual conversation, and uploads it to S3.

*Setup and run instructions coming soon.*

## Components

- **Agent factory** — instantiates a Tavus or HeyGen visual persona, binds it to an ElevenLabs / Cartesia / HeyGen voice, and seeds the LLM (Llama-4, GPT-4o, GPT-4o-mini, GPT-5.4-mini) with the scenario prompt.
- **Pairing service** — issues a unique room URL per Prolific participant and routes them to a single concurrent session.
- **Recorder** — captures the combined audio-visual stream (Daily.co for Tavus, native WebM for HeyGen) and uploads it to S3.

## Getting Started

```bash
# TODO: install instructions
# TODO: how to configure the agent providers (Tavus, HeyGen, ElevenLabs, Cartesia)
# TODO: how to set up the AWS S3 bucket
# TODO: how to launch the video-streaming server
# TODO: how to issue per-participant session URLs
```

## Scenarios

Four scenarios are supported: `conversation`, `professional`, `collaborative_planning`, `creative`. Prompts are described in the paper appendix.
