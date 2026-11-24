"""Stage: download Smart Turn's own published test set (selected languages).

Why: in the streaming simulation our models had home advantage, because they
were trained on the same corpus they were tested on. This is the away game:
a test set built by another team, from different speakers and recordings,
that our models have never seen. Smart Turn has the home advantage here.

The dataset (CC-BY-4.0) is ~4.8 GB across 10 files with all 23 languages mixed
together, so each file is downloaded, the wanted languages are extracted as
16 kHz WAV, and the file is deleted again.

It has labels (complete / incomplete) but NO transcripts, so only audio models
can be scored on it.

Output: data/external/smart_turn_test/<lang>/<id>.wav and manifest.csv
"""

import csv
import io
import shutil

import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as F
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from indic_eot.config import REPO_ROOT, load_params

OUT_DIR = REPO_ROOT / "data" / "external" / "smart_turn_test"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads" / "smart_turn_test"
COLUMNS = [
    "audio",
    "id",
    "language",
    "endpoint_bool",
    "midfiller",
    "endfiller",
    "synthetic",
    "dataset",
]


def main() -> None:
    params = load_params()
    cfg = params["baselines"]["smart_turn_test"]
    sr = params["audio"]["sample_rate"]
    wanted = set(cfg["languages"])

    shards = sorted(
        f
        for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
        if f.endswith(".parquet")
    )
    if cfg.get("max_shards"):
        shards = shards[: cfg["max_shards"]]

    rows = []
    for shard in tqdm(shards, desc="shards"):
        path = hf_hub_download(cfg["repo"], shard, repo_type="dataset", local_dir=DOWNLOAD_DIR)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=COLUMNS):
            for item in batch.to_pylist():
                if item["language"] not in wanted:
                    continue
                audio, file_sr = sf.read(
                    io.BytesIO(item["audio"]["bytes"]), dtype="float32", always_2d=True
                )
                audio = audio.mean(axis=1)
                if file_sr != sr:
                    audio = F.resample(torch.from_numpy(audio), file_sr, sr).numpy()
                wav = OUT_DIR / item["language"] / f"{item['id']}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, audio, sr, subtype="PCM_16")
                rows.append(
                    {
                        "id": item["id"],
                        "lang": item["language"],
                        "label": int(item["endpoint_bool"]),
                        "path": str(wav.relative_to(REPO_ROOT)),
                        "seconds": round(len(audio) / sr, 3),
                        "synthetic": int(bool(item["synthetic"])),
                        "source": item["dataset"],
                        "midfiller": int(bool(item["midfiller"])),
                        "endfiller": int(bool(item["endfiller"])),
                    }
                )
        (DOWNLOAD_DIR / shard).unlink()  # free the disk space

    shutil.rmtree(DOWNLOAD_DIR, ignore_errors=True)
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by_lang = {lang: sum(1 for r in rows if r["lang"] == lang) for lang in sorted(wanted)}
    print(f"Wrote {len(rows)} clips: {by_lang}")


if __name__ == "__main__":
    main()
