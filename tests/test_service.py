"""Tests for the HTTP service. They need the exported model."""

import io

import numpy as np
import pytest
import soundfile as sf

from indic_eot.detector import ASSETS

pytestmark = pytest.mark.skipif(
    not (ASSETS / "export.json").exists(), reason="run the export stage first"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from indic_eot.service.app import app

    with TestClient(app) as test_client:
        yield test_client


def noise(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.random.default_rng(0).normal(0, 0.05, int(seconds * rate)).astype(np.float32)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["max_seconds"] == 8.0


def test_demo_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_raw_samples(client):
    response = client.post(
        "/predict",
        content=noise(2).tobytes(),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["seconds_received"] == 2.0


def test_predict_file_upload_matches_raw(client):
    audio = noise(2)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format="WAV", subtype="FLOAT")
    uploaded = client.post("/predict", files={"file": ("clip.wav", buffer.getvalue())}).json()
    raw = client.post(
        "/predict",
        content=audio.tobytes(),
        headers={"content-type": "application/octet-stream"},
    ).json()
    assert abs(uploaded["probability"] - raw["probability"]) < 1e-3


def test_predict_resamples_other_rates(client):
    buffer = io.BytesIO()
    sf.write(buffer, noise(2, 44100), 44100, format="WAV")
    response = client.post("/predict", files={"file": ("clip.wav", buffer.getvalue())})
    assert response.status_code == 200
    assert response.json()["seconds_received"] == 2.0


def test_predict_rejects_bad_input(client):
    octets = {"content-type": "application/octet-stream"}
    assert client.post("/predict", content=b"abc", headers=octets).status_code == 400
    assert client.post("/predict", content=noise(0.01).tobytes(), headers=octets).status_code == 400
    assert client.post("/predict", json={"a": 1}).status_code == 415
    assert client.post("/predict", files={"file": ("x.wav", b"not audio")}).status_code == 400
