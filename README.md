# TRIBE v2 Communication Lab

Local web app for uploading one text, audio, or video item, running TRIBE v2 on it, and returning the model output plus optional natural-language explanation.

This repo is intentionally scoped as a non-commercial research prototype. It does not claim to predict behavior, persuasion, comprehension, outcomes, or intent.

## What This Project Is

This project runs Meta's TRIBE v2 model locally and uses it as a cautious neural-proxy inference engine.

- You give it one text, audio, or video item
- It runs TRIBE v2 on that item
- It aggregates predicted cortical responses into ROI and proxy summaries
- It returns raw prediction paths, cortical summaries, and an optional natural-language explanation

It is not a human-reaction predictor. It is a local inference tool built on average-subject fMRI response predictions.

## What It Does

- Accepts one `.txt`, audio, or video input via pasted text, local path, or browser upload
- Runs TRIBE v2 inference on the average-subject model
- Aggregates fsaverage5 vertex predictions into HCP ROI summaries
- Computes fixed proxy families:
  - `language_processing_proxy`
  - `emotional_social_proxy`
  - `auditory_salience_proxy`
  - `visual_salience_proxy`
  - `cross_region_spread_proxy`
- Exports:
  - raw `.npy` prediction arrays
  - reproducible JSON result artifact
  - HTML report with charts, ROI tables, caveats, and cortical images when plotting extras are available
  - optional LLM-written explanation when `OPENAI_API_KEY` is set

## Constraints

- TRIBE v2 is licensed `CC-BY-NC-4.0`. Keep usage non-commercial.
- The model predicts average-subject fMRI-like responses, not direct human reactions.
- TRIBE v2 includes a 5-second hemodynamic lag and 1 Hz output resolution.
- The underlying paper explicitly states the model does not model behavior directly.

## Setup

1. Install Python 3.11 and create a local environment:

```bash
uv python install 3.11
uv venv --python 3.11
```

2. Sync dependencies:

```bash
uv sync --extra runtime --extra test
```

3. Authenticate with Hugging Face and ensure your token has access to Meta's gated `meta-llama/Llama-3.2-3B` model:

```bash
uv run python -m pip install huggingface_hub
huggingface-cli login
```

4. Check the runtime:

```bash
uv run communication-lab-check
```

5. Optional: enable natural-language explanations with an OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

6. Run the backend:

```bash
uv run communication-lab-api
```

7. In another terminal, start the TypeScript frontend:

```bash
npm install
npm run dev
```

## Project Layout

- `src/communication_lab/api.py`: FastAPI backend for upload/path/text inference
- `web/`: React + TypeScript frontend
- `src/communication_lab/pipeline.py`: orchestration layer
- `src/communication_lab/tribe_service.py`: lazy TRIBE v2 wrapper
- `src/communication_lab/analysis.py`: ROI aggregation, proxy scoring, pairwise deltas
- `src/communication_lab/explanations.py`: optional LLM explanation layer
- `src/communication_lab/reporting.py`: JSON/HTML exports and cortical image rendering
- `samples/`: starter text variants
- `tests/`: fake-service pipeline tests

## Tests

The automated tests use a fake prediction service so they do not require TRIBE weights or gated model access.

```bash
uv run --extra test pytest
```

## Notes

- On Apple Silicon, the app defaults to CPU unless CUDA is available.
- The first runtime inference can be slow because TRIBE caches extracted features and MNE may fetch the HCP atlas.
- Domain packs are interpretation templates only. They are not separate trained models.
- OpenAI is optional and only used to explain TRIBE outputs in natural language after inference.
