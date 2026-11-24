# Walkthrough: how this project was built, and why each step exists

This file explains the project from the first step to the current state. For each step it says what was done, why it was needed, and what was learned. All numbers come from result files in `results/`.

## The problem in one paragraph

A voice agent has to decide when the user has finished speaking. Most agents wait for a fixed length of silence. A short wait (500 ms) makes the agent interrupt people who pause to think. A long wait (1,000 ms or more) makes it feel slow. This project trains a model that listens to the speech so far and estimates whether the speaker is finished, so the agent can answer quickly when they are and keep waiting when they are only pausing. The focus is Indian speech, starting with Hindi.

## Terms used below

| Term | Meaning |
|---|---|
| Decision point | A moment when the speaker goes quiet and the agent must choose: respond or wait |
| Complete / incomplete | Whether the speaker had finished their turn at that moment |
| AUROC | A score for how well a model separates two classes. 0.5 is guessing, 1.0 is perfect |
| False cut-off | The agent started responding before the speaker had finished |
| Stage | One step of the DVC pipeline (a command with declared inputs and outputs) |
| Run | One training or evaluation recorded in MLflow |

## The tools, and why they are here

| Tool | What it does in this project | Why it matters |
|---|---|---|
| `uv` | Creates the Python environment and pins every package version in `uv.lock` | Anyone who clones the repo gets the same environment |
| DVC | Runs the pipeline in `dvc.yaml`, versions data and models, records fingerprints in `dvc.lock` | Large files stay out of git; only the stages affected by a change are rerun |
| `params.yaml` | Holds every setting in one file | No numbers are hidden in scripts; DVC knows which stage depends on which setting |
| MLflow | Records each training run: settings, curves, final metrics, git commit, dataset fingerprint | Any result can be traced back to the exact code and data that produced it |
| pytest | Tests the dataset logic and text clean-up | The dataset rules are easy to get subtly wrong |

## Step 1: Project setup

**What**: `pyproject.toml`, a pinned Python 3.12 environment, `params.yaml`, a config loader, one test, and `dvc init`.

**Why**: the system Python (3.14) is too new for some machine-learning libraries, so the project pins its own. Putting settings in `params.yaml` from the first day means later experiments are changes to one file, which DVC and MLflow can both track.

## Step 2: Download the data

**What**: [scripts/download_fleurs.py](scripts/download_fleurs.py) downloads Hindi recordings and transcripts from FLEURS, an open multilingual speech corpus, and writes a manifest listing every clip. It became the first DVC stage.

**Why**: a model needs speech with known transcripts. FLEURS is free, public, and covers several Indian languages, so the same code will extend to Tamil, Telugu and Bengali later.

**Learned**: FLEURS has no speaker IDs, only sentence IDs, and each sentence is read by several people. The plan was to keep speakers separate between training and test data; that cannot be checked here. Instead the project uses FLEURS's own train, dev and test splits and verifies that no sentence appears in more than one.

## Step 3: Word timings

**What**: [scripts/align_words.py](scripts/align_words.py) works out when each word is spoken in each clip, using Meta's MMS forced-alignment model.

**Why**: the transcript says what was said, not when. To find the pauses inside a sentence, the project needs the start and end time of every word.

**How it works**: the model estimates which Latin letter is being spoken in every 20 ms slice. The Hindi transcript is converted to Latin letters (`uroman`), and a search finds the best way to lay the letters along the audio. One model covers many languages this way.

**Learned**: some clips align badly (their recording does not match their transcript). Clips with low alignment confidence are dropped in the next step.

## Step 4: Build the dataset (first version)

**What**: [src/indic_eot/dataset.py](src/indic_eot/dataset.py) and [scripts/build_dataset.py](scripts/build_dataset.py) turned each recording into short clips labelled complete or incomplete.

**Why**: no public end-of-turn dataset exists for Indian speech, so one had to be constructed from ordinary recordings.

**First design**: an incomplete clip was a sentence cut at a word boundary partway through; a complete clip was the end of the sentence. Both were faded out and given the same background hiss, so that the edit itself could not reveal the label.

**Learned**: the first build made complete clips nearly twice as long as incomplete ones (18.7 words against 9.2). A model could have scored well by measuring length. The fix was to build clips in matched pairs with the same word count.

*This design was later replaced. See step 11.*

## Step 5: Validation and shortcut check

**What**: two stages. [scripts/validate_dataset.py](scripts/validate_dataset.py) checks the data (balanced classes, no leaked sentences, valid audio files, no duplicates) and stops the pipeline if anything fails. [scripts/shortcut_check.py](scripts/shortcut_check.py) trains very simple classifiers on features that contain no words, such as clip length and loudness.

**Why**: a model will use any easy pattern that separates the classes, including ones that are accidents of how the data was built. The shortcut check measures how far a model could get without understanding speech at all. That number is the bar a real model must clearly beat.

**Learned**: length scored 0.54 (near guessing, as intended). Loudness of the last moment of speech scored about 0.78. Further checks showed this is mostly genuine: people get quieter at the end of a sentence.

## Step 6: Train the audio model

**What**: [src/indic_eot/audio_model.py](src/indic_eot/audio_model.py) and [scripts/train_audio.py](scripts/train_audio.py). The model is the encoder from Whisper-tiny (8 million parameters, already trained on about 100 languages) with a small classifier on top. It looks at the last 8 seconds of audio.

**Why this design**: Whisper already knows a great deal about how speech sounds, so only a little training is needed to teach it one new question. Short clips are padded with silence at the start, so the end of the speech, which is what matters, always sits in the same place.

**Why the training details**:
- The best epoch is chosen on the validation split, and the test split is scored once at the end. Choosing on the test split would make the reported number too optimistic.
- Training clips get random volume changes and background noise, so the model cannot rely on absolute loudness.
- Confidence intervals are computed by resampling whole recordings, because several clips come from the same recording and are not independent.

**Learned**: 95.9% test accuracy, far above the loudness-only bar. But a breakdown showed a weakness: where the speaker had really paused, the model caught only 77% of incomplete clips.

## Step 7: Rebalance toward real pauses

**What**: changed how cut points were chosen, so that 60% came from places where the speaker really paused. Retrained.

**Why**: only about 11% of incomplete training clips were real pauses, yet real pauses are what an agent faces. The model had seen too few.

**Learned**: recall at real pauses rose from 77% to 87%. Overall accuracy fell slightly to 94.0%, because the test set now held more hard cases. This was the first sign that the easy cases had been flattering the model.

## Step 8: Train the text model

**What**: [src/indic_eot/text_model.py](src/indic_eot/text_model.py) and [scripts/train_text.py](scripts/train_text.py). A small multilingual transformer (MiniLM) that reads only the transcript.

**Why**: the audio model hears how something was said; the text model reads what was said. "मुझे दो टिकट चाहिए" is a finished thought and "मुझे दो टिकट" is not, whatever the voice did.

**Why punctuation is stripped**: a live speech recogniser gives no reliable punctuation while someone is still talking. And in this data every complete sentence ends with "।", which would give the answer away.

**Learned**: 92.7% test accuracy, a little below audio. The best epoch was the first, a sign that the text data has little variety (about 1,500 formal sentences).

## Step 9: Fusion

**What**: [scripts/fuse.py](scripts/fuse.py) combines the two models with three learned numbers: a weight for audio, a weight for text, and an offset. Fitted on validation data, scored on test data.

**Why**: the two models make different mistakes, so together they should beat either alone.

**Learned**: fusion reached 95.2% accuracy. A paired test showed the gain over audio alone was unlikely to be chance (1.2 points, 95% interval 0.2 to 2.3).

## Step 10: Streaming simulation

**What**: [scripts/stream_points.py](scripts/stream_points.py) takes each test recording exactly as recorded, with no editing, and scores the models at every natural pause and at the true end. [scripts/simulate.py](scripts/simulate.py) then replays two policies: a plain silence timeout, and "ask the model after 200 ms of silence".

**Why**: clip accuracy does not tell you how an agent behaves. What matters is how often the agent cuts someone off and how long it takes to respond. This step measures both.

**Learned**: this was the turning point. Despite 95% clip accuracy, the models cut off about half of all recordings.

| Reason | Detail |
|---|---|
| Trained on the wrong moments | Most incomplete clips were cut where the next word followed immediately. An agent never decides there, because there is no silence |
| Pauses at commas were excluded | They had been left out of training as ambiguous, but they are 22% of real pauses, and the models called most of them complete |
| Errors compound | A recording has about three pauses, so a 13% error per pause cuts off roughly a third of recordings |

Without this step the project would have reported 95% and been wrong about its usefulness.

## Step 11: Rebuild the dataset around real decision points

**What**: the dataset was redesigned. An incomplete example is now the unedited audio up to a natural pause of 200 ms or more, with the real silence that followed. A complete example is the unedited audio up to the true end. Pauses at commas are included. Nothing is cut, faded or padded at the end of a clip. All models were retrained.

**Why**: training data should match the moments the model will face in use. The first design did not.

**Learned** (fusion model, unedited test recordings):

| Measure | Before | After |
|---|---|---|
| True ends called complete (406) | 97.0% | 96.1% |
| Mid-clause pauses wrongly called complete (1,035) | 13.4% | 7.8% |
| Pauses at a comma wrongly called complete (285) | 55.4% | 41.8% |
| Clip accuracy | 95.2% | 89.7% |

Clip accuracy fell because the task is now the real, harder one.

Agent behaviour, pause lengths as recorded (406 recordings):

| Policy | Recordings cut off | Mean delay |
|---|---|---|
| Silence 500 ms | 55.7% | 500 ms |
| Fusion, threshold 0.5 | 38.2% | 241 ms |
| Silence 700 ms | 24.1% | 700 ms |
| Fusion, threshold 0.86 | 24.1% | 441 ms |
| Silence 1,000 ms | 7.9% | 1,000 ms |

The model now beats a 500 ms timeout on both measures and matches a 700 ms timeout with about 260 ms less delay. It cannot match a 1,000 ms timeout's low cut-off rate.

## Step 12: Compare with an existing detector

**What**: [scripts/baseline_smart_turn.py](scripts/baseline_smart_turn.py) runs Smart Turn v3.2, an open audio turn detector from Pipecat, on exactly the same decision points. The code that builds those points moved to [src/indic_eot/streaming.py](src/indic_eot/streaming.py) so that every model is given identical audio.

**Why**: on its own, "42% of comma pauses called complete" cannot be judged. A reader needs to know how an established model does on the same audio. Smart Turn has the same shape as our audio model (Whisper-tiny encoder, last 8 seconds) but was trained on far more data in 23 languages, Hindi included.

**Learned** (threshold 0.5, unedited test recordings):

| Model | True ends called complete (406) | Pauses wrongly called complete (1,320) | Recordings cut off | Mean delay |
|---|---|---|---|---|
| Ours: audio | 98.3% | 23.8% | 46.8% | 218 ms |
| Ours: fusion | 96.1% | 15.2% | 38.2% | 241 ms |
| Smart Turn v3.2 | 75.9% | 20.8% | 46.1% | 556 ms |

- On pauses, Smart Turn and our audio model are close (21% against 24%). So pauses at commas are hard for an established model too, not a flaw peculiar to ours.
- Smart Turn misses about a quarter of the true ends, so it falls back to the long wait more often and responds more slowly.

**This comparison is not fair to Smart Turn, and must not be read as "ours is better".** Our models were trained on the training split of this same corpus: same recording style, same kind of sentences. Smart Turn has never seen it. The fair reverse test is to run our models on Smart Turn's own published test set, where we would be the outsider.

**Not included**: LiveKit's turn detector. Its licence allows use only inside the LiveKit Agents framework, not standalone, so it was not run.

## Where the project stands

**What works**
- A reproducible pipeline: one command (`uv run dvc repro`) rebuilds everything from download to plot.
- An evaluation on unedited recordings that measures agent behaviour, not just clip accuracy.
- A fusion model that beats short silence timeouts.
- On this corpus, pause errors comparable to an established open detector, with faster responses (an in-domain comparison; see step 12).

**What limits it**
- Pauses at commas: the model still calls 42% of them complete.
- Little data: 4,359 training pairs from one corpus of read speech. Both models start overfitting after one or two epochs.
- Only Hindi so far. No Hinglish, no conversational speech.
- The text model is given perfect transcripts. A live agent would have a speech recogniser's errors.
- The "with hesitations" simulation uses assumed pause lengths.
- Nothing has been tested on real recordings of people hesitating.

**What comes next**
1. The reverse comparison: run our models on Smart Turn's published test set, where our models are the outsider.
2. A smarter policy that waits a middling time when the model is unsure.
3. A larger encoder and more Hindi data; then more languages and Hinglish.
4. Export to ONNX, package as `indic_eot`, measure speed.
5. Record and test on real hesitant speech.

## The main lesson

The first model scored 95% and would have failed in use. The metric was measuring a task that was easier than the real one. Building the evaluation that mirrors real use (step 10) was what exposed this, and it changed the dataset design. The order to remember: decide how the model will be used, build the test for that, and only then trust a number.
