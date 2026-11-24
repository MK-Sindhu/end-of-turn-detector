# The deployable service: API + demo page. About 400 MB, no PyTorch.
#
#   docker build -t indic-eot .
#   docker run -p 7860:7860 -e EOT_MODEL_REPO=<user>/indic-eot indic-eot
#
# The model is not baked into the image. It is downloaded at startup from the
# Hugging Face repo named in EOT_MODEL_REPO, or read from a folder mounted at
# EOT_MODEL_DIR. This keeps code releases and model releases separate.
FROM python:3.12-slim

# libsndfile reads uploaded audio files; libgomp is needed by onnxruntime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as an ordinary user, not root (Hugging Face Spaces expects uid 1000).
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app
ENV PATH="/home/app/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir --user ".[serve]"

EXPOSE 7860
CMD ["uvicorn", "indic_eot.service.app:app", "--host", "0.0.0.0", "--port", "7860"]
