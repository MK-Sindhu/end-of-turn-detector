"""Upload the exported model to a Hugging Face model repo.

The deployed service downloads the model from this repo when it starts, so
publishing here is how a new model goes live. Run it after `dvc repro` has
produced a model you are happy with:

    HF_TOKEN=... python scripts/publish_model.py <user>/indic-eot

Uploads the files in src/indic_eot/assets/ plus the metrics that describe them.
"""

import argparse
import json
import subprocess

from huggingface_hub import HfApi

from indic_eot.config import REPO_ROOT
from indic_eot.detector import ASSETS

METRICS = ("export_metrics.json", "audio_metrics.json", "smart_turn_test_metrics.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="model repo, for example yourname/indic-eot")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not (ASSETS / "export.json").exists():
        raise SystemExit("No exported model. Run `uv run dvc repro export` first.")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    ).stdout.strip()
    export = json.loads((REPO_ROOT / "results" / "export_metrics.json").read_text())
    message = (
        f"Model from commit {commit}: "
        f"test accuracy {export['int8']['accuracy']}, {export['int8']['size_mb']} MB"
    )

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=ASSETS, repo_id=args.repo, repo_type="model", commit_message=message
    )
    for name in METRICS:
        path = REPO_ROOT / "results" / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=path, path_in_repo=f"metrics/{name}",
                repo_id=args.repo, repo_type="model", commit_message=f"Metrics: {name}",
            )
    print(f"Published to https://huggingface.co/{args.repo}\n{message}")


if __name__ == "__main__":
    main()
