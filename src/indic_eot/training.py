"""Shared pieces for training: dataset loading, augmentation, metrics, MLflow setup."""

import csv
import subprocess

import mlflow
import numpy as np
import soundfile as sf
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from indic_eot.config import REPO_ROOT

METADATA = REPO_ROOT / "data" / "processed" / "eot" / "metadata.csv"


def load_rows(split: str, limit: int | None = None) -> list[dict]:
    with open(METADATA, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if limit:  # a random sample, so both classes stay represented
        picked = np.random.default_rng(0).permutation(len(rows))[:limit]
        rows = [rows[i] for i in sorted(picked)]
    return rows


class ClipDataset(torch.utils.data.Dataset):
    """Loads a clip from disk and, for training, randomly changes its volume and
    adds background noise. This stops the model relying on absolute loudness and
    makes it more robust to real microphones."""

    def __init__(self, rows: list[dict], augment: dict | None = None, seed: int = 0):
        self.rows = rows
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        row = self.rows[i]
        audio, _ = sf.read(REPO_ROOT / row["path"], dtype="float32")
        if self.augment:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            gain_db = rng.uniform(-self.augment["gain_db"], self.augment["gain_db"])
            audio = audio * 10 ** (gain_db / 20)
            if rng.random() < self.augment["noise_prob"]:
                snr_db = rng.uniform(*self.augment["snr_db_range"])
                signal_power = np.mean(audio**2) + 1e-10
                noise_power = signal_power / 10 ** (snr_db / 10)
                noise = rng.normal(0, np.sqrt(noise_power), len(audio))
                audio = audio + noise.astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return audio, int(row["label"])


def classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)),
    }


def bootstrap_ci(
    labels: np.ndarray, probs: np.ndarray, groups: list[str], n: int = 1000, seed: int = 0
) -> dict[str, list[float]]:
    """95% confidence intervals, resampling whole source recordings.

    Several test clips come from the same recording, so they are not
    independent. Resampling recordings (not clips) gives honest intervals.
    """
    rng = np.random.default_rng(seed)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    group_indices = list(by_group.values())

    samples: dict[str, list[float]] = {"accuracy": [], "f1": [], "auroc": []}
    for _ in range(n):
        picked = rng.integers(0, len(group_indices), len(group_indices))
        idx = np.concatenate([group_indices[j] for j in picked])
        if len(set(labels[idx])) < 2:
            continue
        for name, value in classification_metrics(labels[idx], probs[idx]).items():
            samples[name].append(value)
    return {
        name: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        for name, v in samples.items()
    }


def start_mlflow_run(experiment: str, run_name: str) -> mlflow.ActiveRun:
    """Open an MLflow run stored locally in mlflow.db, tagged with exactly which
    code and data produced it."""
    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    run = mlflow.start_run(run_name=run_name)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    mlflow.set_tags({"git_commit": commit, "git_dirty": dirty, "dataset_md5": dataset_md5()})
    return run


def dataset_md5() -> str:
    """The DVC fingerprint of the dataset this run trained on (from dvc.lock)."""
    lock = REPO_ROOT / "dvc.lock"
    if not lock.exists():
        return "unknown"
    stages = yaml.safe_load(lock.read_text())["stages"]
    for out in stages.get("build_dataset", {}).get("outs", []):
        if out["path"] == "data/processed/eot":
            return out["md5"]
    return "unknown"


def flatten(d: dict, prefix: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}, the form MLflow wants for parameters."""
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat
