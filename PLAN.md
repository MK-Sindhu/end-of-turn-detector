# Plan: End-of-turn detector for Indian speech

Status: plan only. Nothing is built yet.

## Goal

A small model that listens to a user's speech up to a pause and returns the probability that they have finished their turn. Voice agents today mostly wait for a fixed length of silence, which either cuts people off mid-thought (short wait) or makes the agent feel slow (long wait). The model lets an agent reply quickly when the person is done and keep waiting when they are only pausing.

Target speech: Hindi, Indian English, Hinglish code-switching, and other Indic languages (Tamil, Telugu, Bengali) as far as the data allows.

## Ground rules

- Zero cost: open datasets, open models, training on the Mac (M4, 24 GB, no CUDA).
- Every number in the README comes from a script in the repo that was actually run, with its output saved under `results/`.
- Small, regular commits under the owner's git identity. No pushing.
- Datasets and large checkpoints are never committed. The final exported model is committed only if it is under about 25 MB.

## Steps

### 1. Data

No public end-of-turn dataset for Indic speech is known, so it is constructed. (The first task is a short search to confirm this.)

- **Audio source**: complete utterances from open Indic speech sets. FLEURS (Hindi, Tamil, Telugu, Bengali, Indian English) is the starting point. Kathbath, IndicVoices, Svarah and Common Voice are added if they can actually be downloaded.
- **"Finished" examples**: a full utterance.
- **"Not finished" examples**: an utterance cut mid-sentence at a word boundary. Word timings come from forced alignment or Whisper word timestamps.
- **Text source**: Hindi, Hinglish (romanised and Devanagari), English and other Indic sentences, as complete sentences versus truncated prefixes. Open code-switched text corpora are used for Hinglish.
- **Splits**: train, validation and test sets share no speakers. One language is held out entirely to measure how well the model transfers to a language it never saw.
- **Size**: tens of thousands of clips at most, under about 20 GB of downloads.

### 2. Shortcut check

A model could cheat by detecting the cut itself rather than understanding the speech. To prevent that:

- Both kinds of clip end the same way: same trailing silence, same fade, cuts placed at quiet points between words.
- Noise and volume changes are added during training.
- A trivial classifier using only the last 100 ms of each clip is trained as a check. It should score near chance, and its score is reported.

### 3. Models

| Model | Input | Design |
|---|---|---|
| Audio | Last 8 seconds of speech | Small pretrained speech encoder (Whisper tiny or base) with a light classification head |
| Text | Transcript so far | Small multilingual transformer |
| Fusion | Both | Simple combination of the two scores |

All three must run in well under 100 ms on a CPU.

### 4. Evaluation

- **Per clip**: accuracy, F1 and AUROC for each language, with confidence intervals and sample sizes.
- **Headline result, a streaming simulation**: test utterances with mid-utterance pauses of varied length are replayed against different policies:
  - plain silence timeouts of 300, 500, 700, 1000 and 1500 ms;
  - model-gated: end the turn after a short silence only if the model's probability is above a threshold, with a long fallback timeout.

  The result is a plot of false cut-off rate against median response delay for each policy. The README leads with this.
- **Existing open detectors**: Pipecat's smart-turn and LiveKit's turn-detector are run on the same test sets if they can be made to run. If one cannot, the README says so.
- **Speed**: CPU inference time of the exported models.

### 5. Packaging

```python
from indic_eot import EndOfTurnDetector

det = EndOfTurnDetector.load()        # bundled model, no network needed
p = det.predict(audio, text=None)     # audio: float32 mono 16 kHz; returns 0..1
```

- Installable package named `indic_eot`.
- Exported to ONNX so that inference needs only onnxruntime and numpy.
- Command-line use: `python -m indic_eot file.wav`.
- A live microphone demo script.

The stress-test arena uses this exact interface, so it must not change.

### 6. MLOps

| Tool | Used for |
|---|---|
| **DVC pipeline** | `dvc.yaml` defines the stages download, align, build dataset, shortcut check, train (audio, text, fusion), evaluate, streaming simulation, export. `dvc repro` rebuilds only what changed. |
| **DVC data versioning** | Raw corpora, the constructed dataset and checkpoints are tracked by DVC, with only the small `.dvc` and `dvc.lock` files in git. The remote is a local folder during the build; the owner can later point it at Google Drive or DagsHub (both have free tiers). |
| **`params.yaml`** | All settings in one file: languages, cut rules, trailing silence, model sizes, learning rates, thresholds. No values hard-coded in scripts. |
| **DVC metrics and plots** | Evaluation writes JSON metrics and plot data, so `dvc metrics diff` and `dvc plots diff` compare any two commits. |
| **MLflow tracking** | Every training run logs its parameters, per-epoch metrics, per-language test metrics, the trade-off plot, the git commit and the dataset version (DVC hash). Local store (SQLite plus an artifacts folder), viewed with `mlflow ui`. |
| **MLflow model registry** | Each exported ONNX model is registered with its metrics. The model that ships in the package is the one carrying the `champion` alias, and the README records which run produced it. |
| **Data validation** | A pipeline stage checks the constructed dataset before training: class balance, clip lengths, sample rate, no speaker shared between splits, no duplicate clips. The pipeline stops if a check fails. |
| **CI (GitHub Actions)** | On every push: lint, unit tests, and a smoke test that loads the bundled model and scores a sample clip. Free for public repos; it starts running once the owner pushes. |
| **Serving** | A small FastAPI service (`POST /predict` with audio, returns the probability) with a Dockerfile, plus a latency test for it. |
| **Model card** | Intended use, training data, metrics per language, limitations. |

MLflow run folders and the DVC cache stay out of git.

### 7. Repo contents

- One script per stage, wired together by `dvc.yaml`. Reproducible from a clean clone with `dvc repro`.
- Unit tests for the dataset construction and the package interface.
- README: trade-off plot and result table first, then method, how to reproduce, the MLOps setup, and limitations.

## Known limitations (stated in the README)

- Cutting read speech mid-sentence is a stand-in for real hesitation, not the same thing. Real pauses carry tone and rhythm cues that cut clips lack.
- The source corpora are mostly read speech, not conversation.
- Hinglish audio is scarce in open data, so code-switching may be covered better by the text model than the audio model.

## What needs the owner

- **A real-speech test set**: a recording script will be provided so that 50 to 100 prompted utterances with natural pauses can be recorded and scored. Until that is done, all results are on constructed data, and the README will say so. This is the single thing that most strengthens the project.
- **Publishing the model** to Hugging Face is left to the owner.
- **A shared DVC remote** (Google Drive or DagsHub), if others should be able to pull the data. Without one, a fresh clone rebuilds the data by running the pipeline.
- **Docker**: the Dockerfile will be written, but the image can only be built and tested if Docker is installed and running on the Mac. This was not checked.

## Order of work

1. Thin slice: Hindi only, small sample, tiny training run, evaluation, package interface working end to end. DVC pipeline, `params.yaml` and MLflow logging are in place from this first slice, not added afterwards.
2. Scale up data and languages; add data validation; run the shortcut check.
3. Full training of the audio, text and fusion models, all tracked in MLflow.
4. Streaming simulation and comparison with existing detectors.
5. ONNX export, model registry, speed measurement, demo, tests.
6. FastAPI service and Dockerfile, CI workflow, model card, README.

## Decisions to confirm

- Languages: Hindi, Indian English and Hinglish as the core, with Tamil, Telugu and Bengali as extras. Change this list if different languages matter more.
- Held-out language for the transfer test: Bengali is proposed.
- Package name: `indic_eot`.
