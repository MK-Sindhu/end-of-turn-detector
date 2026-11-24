"""Deploy the service to a Hugging Face Space (free hosting for Docker apps).

    HF_TOKEN=... python scripts/deploy_space.py <user>/indic-eot-demo --model-repo <user>/indic-eot

Uploads only what the service needs: the Dockerfile, the package source and the
Space's own README. The Space then builds the image and runs it. The model is
not uploaded here; the Space downloads it from --model-repo when it starts.

The CD workflow (.github/workflows/deploy.yml) runs this same script.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP = shutil.ignore_patterns("__pycache__", "assets", "*.pyc", ".gitignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space", help="Space name, for example yourname/indic-eot-demo")
    parser.add_argument("--model-repo", required=True, help="model repo the Space loads")
    parser.add_argument("--allowed-origins", default="*", help="sites allowed to call the API")
    parser.add_argument("--message", default="Deploy service")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(args.space, "EOT_MODEL_REPO", args.model_repo)
    api.add_space_variable(args.space, "EOT_ALLOWED_ORIGINS", args.allowed_origins)

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        shutil.copy(REPO_ROOT / "Dockerfile", bundle)
        shutil.copy(REPO_ROOT / "pyproject.toml", bundle)
        shutil.copy(REPO_ROOT / "deploy" / "space" / "README.md", bundle)
        shutil.copytree(REPO_ROOT / "src", bundle / "src", ignore=SKIP)
        api.upload_folder(
            folder_path=bundle, repo_id=args.space, repo_type="space",
            commit_message=args.message, delete_patterns="*",
        )
    print(f"Deployed. Build log and app: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
