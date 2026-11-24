"""HTTP API for the end-of-turn detector.

    GET  /          the demo page (speak into the microphone)
    GET  /health    is the service up, and which model is loaded
    POST /predict   audio in, probability out

/predict accepts audio in two ways:
  - a WAV/FLAC/OGG file as a normal form upload (field name "file"), or
  - raw samples: 32-bit floats, mono, 16 kHz, with the header
    "Content-Type: application/octet-stream". This is what the demo page sends.

Where the model comes from, in order:
  1. the folder in the EOT_MODEL_DIR environment variable
  2. the files bundled in the package (present after `dvc repro export`)
  3. the Hugging Face model repo named in EOT_MODEL_REPO (downloaded at startup)
"""

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from indic_eot import __version__
from indic_eot.detector import ASSETS, EndOfTurnDetector
from indic_eot.features import SAMPLE_RATE

STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 5 * 1024 * 1024  # 8 s of raw audio is 0.5 MB; refuse anything huge
MIN_SAMPLES = SAMPLE_RATE // 10  # under 0.1 s there is nothing to judge

state: dict = {"detector": None, "source": None}


def find_model_dir() -> tuple[Path, str]:
    if os.environ.get("EOT_MODEL_DIR"):
        return Path(os.environ["EOT_MODEL_DIR"]), "EOT_MODEL_DIR"
    if (ASSETS / "export.json").exists():
        return ASSETS, "bundled"
    repo = os.environ.get("EOT_MODEL_REPO")
    if repo:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo)), f"hub:{repo}"
    raise RuntimeError(
        "No model found. Run `dvc repro export`, or set EOT_MODEL_DIR or EOT_MODEL_REPO."
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_dir, source = find_model_dir()
    state["detector"] = EndOfTurnDetector.load(model_dir)
    state["source"] = source
    yield


app = FastAPI(title="Indic end-of-turn detector", version=__version__, lifespan=lifespan)

# Which websites may call the API from a browser. "*" allows any; set
# EOT_ALLOWED_ORIGINS to a comma-separated list to restrict it to your own site.
origins = [o.strip() for o in os.environ.get("EOT_ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"])


def decode_file(data: bytes) -> np.ndarray:
    """Read an audio file, mix to mono, and resample to 16 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as error:
        raise HTTPException(400, f"Could not read the audio file: {error}") from error
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, rate).astype(np.float32)
    return audio


@app.get("/health")
def health() -> dict:
    detector = state["detector"]
    return {
        "status": "ok",
        "version": __version__,
        "model_source": state["source"],
        "model": detector.info if detector else None,
    }


@app.post("/predict")
async def predict(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/octet-stream"):
        data = await request.body()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Audio too large.")
        if len(data) % 4:
            raise HTTPException(400, "Raw audio must be 32-bit floats.")
        audio = np.frombuffer(data, dtype="<f4")
    elif content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(400, 'Send the audio as a form field named "file".')
        data = await upload.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Audio too large.")
        audio = decode_file(data)
    else:
        raise HTTPException(415, "Send a file upload or application/octet-stream.")

    if len(audio) < MIN_SAMPLES:
        raise HTTPException(400, "Audio is shorter than 0.1 seconds.")
    if not np.isfinite(audio).all():
        raise HTTPException(400, "Audio contains invalid numbers.")

    start = time.perf_counter()
    probability = state["detector"].predict(audio)
    return {
        "probability": round(probability, 4),
        "complete": probability >= 0.5,
        "seconds_received": round(len(audio) / SAMPLE_RATE, 2),
        "inference_ms": round((time.perf_counter() - start) * 1000, 1),
    }


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    return FileResponse(STATIC / "index.html")
