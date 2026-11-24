---
title: Indic End-of-Turn Detector
emoji: 🎙️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Detects when a speaker has finished their turn
---

# Indic end-of-turn detector

Live demo and HTTP API for a model that listens to speech up to a pause and
estimates whether the speaker has finished or is only pausing.

- `GET /` the microphone demo
- `GET /health` service and model information
- `POST /predict` audio in, probability out
- `GET /docs` interactive API reference

Research preview: trained on read Hindi speech; less reliable elsewhere.
