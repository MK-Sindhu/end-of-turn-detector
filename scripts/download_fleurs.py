"""Stage 1: download FLEURS speech for the languages listed in params.yaml.

For each language and split this fetches two files from the Hugging Face Hub:
  - <split>.tsv          transcripts, one row per clip
  - audio/<split>.tar.gz the recordings (16 kHz WAV)

It unpacks the audio and writes one manifest.csv listing every clip, which is
what the later stages read.

Output layout:
  data/raw/fleurs/<lang>/<split>/<clip>.wav
  data/raw/fleurs/manifest.csv
"""

import csv
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from indic_eot.config import REPO_ROOT, load_params

REPO_ID = "google/fleurs"
OUT_DIR = REPO_ROOT / "data" / "raw" / "fleurs"
DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads" / "fleurs"  # archives, deleted after unpacking

# Columns of the FLEURS .tsv files (they have no header row).
TSV_COLUMNS = ["sentence_id", "file_name", "raw_text", "text", "phonemes", "num_samples", "gender"]


def fetch(filename: str) -> Path:
    """Download one file from the dataset repo and return its local path."""
    return Path(
        hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=DOWNLOAD_DIR)
    )


def unpack_audio(archive: Path, dest: Path) -> None:
    """Extract every .wav in the archive straight into dest (no sub-folders)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".wav"):
                source = tar.extractfile(member)
                with open(dest / Path(member.name).name, "wb") as out:
                    shutil.copyfileobj(source, out)


def main() -> None:
    params = load_params()["data"]
    rows = []

    for lang in params["languages"]:
        for split in params["splits"]:
            print(f"[{lang}/{split}] downloading ...")
            tsv = fetch(f"data/{lang}/{split}.tsv")
            archive = fetch(f"data/{lang}/audio/{split}.tar.gz")

            audio_dir = OUT_DIR / lang / split
            unpack_audio(archive, audio_dir)
            archive.unlink()  # free the disk space

            missing = 0
            with open(tsv, encoding="utf-8") as f:
                for line in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    record = dict(zip(TSV_COLUMNS, line))
                    wav = audio_dir / record["file_name"]
                    if not wav.exists():
                        missing += 1
                        continue
                    rows.append(
                        {
                            "lang": lang,
                            "split": split,
                            "sentence_id": record["sentence_id"],
                            "path": str(wav.relative_to(REPO_ROOT)),
                            "text": record["raw_text"],
                            "num_samples": record["num_samples"],
                            "gender": record["gender"],
                        }
                    )
            kept = sum(1 for r in rows if r["lang"] == lang and r["split"] == split)
            print(f"[{lang}/{split}] {kept} clips ({missing} listed in the tsv but missing audio)")

    shutil.rmtree(DOWNLOAD_DIR.parent, ignore_errors=True)

    manifest = OUT_DIR / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips to {manifest.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
