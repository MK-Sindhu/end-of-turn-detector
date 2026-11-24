"""Stage 6: train the audio end-of-turn model.

Each epoch: train on the train split, score the validation split, and keep the
weights from the epoch with the best validation AUROC. At the end, load those
best weights and score the test split ONCE. (Picking the best epoch on the
test split would be cheating: the reported number would be optimistically
biased.)

Everything is logged to MLflow: settings, per-epoch curves, final metrics.

Outputs:
  models/audio/model.pt             best weights
  results/audio_metrics.json        validation and test metrics with 95% CIs
  results/audio_predictions.csv     P(complete) for every val and test clip
"""

import argparse
import csv
import json
import time

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from indic_eot.audio_model import AudioEOT, FeaturePrep
from indic_eot.config import REPO_ROOT, load_params
from indic_eot.training import (
    ClipDataset,
    bootstrap_ci,
    classification_metrics,
    flatten,
    load_rows,
    start_mlflow_run,
)

MODEL_FILE = REPO_ROOT / "models" / "audio" / "model.pt"
METRICS_FILE = REPO_ROOT / "results" / "audio_metrics.json"
PREDICTIONS_FILE = REPO_ROOT / "results" / "audio_predictions.csv"


class Collate:
    """Turns a list of (waveform, label) into model-ready tensors. Runs inside
    the data-loading worker processes, so the spectrograms are computed in
    parallel while the GPU trains."""

    def __init__(self, prep: FeaturePrep):
        self.prep = prep

    def __call__(self, batch):
        clips, labels = zip(*batch)
        mel, mask = self.prep(list(clips))
        return mel, mask, torch.tensor(labels, dtype=torch.float32)


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model over a loader. Returns (labels, probabilities, mean loss)."""
    model.eval()
    labels, probs, losses = [], [], []
    with torch.inference_mode():
        for mel, mask, y in loader:
            logits = model(mel.to(device), mask.to(device)).cpu()
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
            )
            probs.append(torch.sigmoid(logits))
            labels.append(y)
    labels = torch.cat(labels).numpy().astype(int)
    probs = torch.cat(probs).numpy()
    return labels, probs, float(sum(losses) / len(labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="use only N clips per split (quick trial)")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--experiment", help="override the MLflow experiment name")
    args = parser.parse_args()

    params = load_params()
    cfg = params["train_audio"]
    epochs = args.epochs or cfg["epochs"]
    max_seconds = params["audio"]["max_seconds"]
    torch.manual_seed(params["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    collate = Collate(FeaturePrep(cfg["encoder"], max_seconds, params["audio"]["sample_rate"]))
    workers = cfg["num_workers"]

    rows = {split: load_rows(split, args.limit) for split in ("train", "val", "test")}
    train_set = ClipDataset(rows["train"], augment=cfg["augment"], seed=params["seed"])
    loaders = {
        "train": DataLoader(
            train_set,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
            num_workers=workers,
        ),
        "val": DataLoader(
            ClipDataset(rows["val"]), batch_size=64, collate_fn=collate, num_workers=workers
        ),
        "test": DataLoader(
            ClipDataset(rows["test"]), batch_size=64, collate_fn=collate, num_workers=workers
        ),
    }

    model = AudioEOT(cfg["encoder"], max_seconds, cfg["dropout"]).to(device)
    # The pretrained encoder gets a small learning rate (it already knows speech,
    # we only nudge it); the brand-new head gets a larger one.
    head_params = list(model.attention.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg["lr_encoder"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    total_steps = epochs * len(loaders["train"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg["warmup_fraction"] * total_steps), total_steps
    )

    experiment = args.experiment or params["mlflow"]["experiment"]
    with start_mlflow_run(experiment, run_name="audio"):
        mlflow.log_params(flatten({"audio": params["audio"], "train_audio": cfg, "seed": params["seed"]}))
        mlflow.log_params(
            {"device": device, "train_clips": len(rows["train"]), "epochs_run": epochs}
        )

        best_auroc, step = -1.0, 0
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, epochs + 1):
            model.train()
            train_set.epoch = epoch  # new random augmentation each epoch
            started = time.time()
            running = []
            for mel, mask, y in tqdm(loaders["train"], desc=f"epoch {epoch}/{epochs}"):
                logits = model(mel.to(device), mask.to(device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running.append(loss.item())
                step += 1
                if step % 20 == 0:
                    mlflow.log_metric("train_loss", float(np.mean(running[-20:])), step=step)

            labels, probs, val_loss = predict(model, loaders["val"], device)
            val = classification_metrics(labels, probs)
            mlflow.log_metrics(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val["accuracy"],
                    "val_auroc": val["auroc"],
                    "epoch_seconds": time.time() - started,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}: train_loss {np.mean(running):.4f}  val_loss {val_loss:.4f}  "
                f"val_acc {val['accuracy']:.4f}  val_auroc {val['auroc']:.4f}"
            )
            if val["auroc"] > best_auroc:
                best_auroc, best_epoch = val["auroc"], epoch
                torch.save(model.state_dict(), MODEL_FILE)

        # Final scoring with the best epoch's weights.
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        report = {"best_epoch": best_epoch, "encoder": cfg["encoder"]}
        prediction_rows = []
        for split in ("val", "test"):
            labels, probs, loss = predict(model, loaders[split], device)
            groups = [r["source_path"] for r in rows[split]]
            report[split] = {
                "clips": len(labels),
                "loss": round(loss, 4),
                **{k: round(v, 4) for k, v in classification_metrics(labels, probs).items()},
                "ci95": bootstrap_ci(labels, probs, groups),
            }
            # Does the model still catch incomplete clips when our cut removed
            # nothing audible (the speaker really paused there)? If it only
            # worked on cuts inside flowing speech, it would be detecting the
            # edit rather than the speech.
            gap = np.array([int(r["gap_ms"]) for r in rows[split]])
            check = params["shortcut_check"]
            for name, mask in {
                "real_pause": gap >= check["real_pause_ms"],
                "connected_speech": gap <= check["connected_speech_ms"],
            }.items():
                subset = (labels == 0) & mask
                report[split][f"incomplete_recall_{name}"] = {
                    "clips": int(subset.sum()),
                    "recall": round(float((probs[subset] < 0.5).mean()), 4),
                }
            prediction_rows += [
                {"id": r["id"], "split": split, "label": r["label"], "prob": f"{p:.5f}"}
                for r, p in zip(rows[split], probs)
            ]

        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        METRICS_FILE.write_text(json.dumps(report, indent=2) + "\n")
        with open(PREDICTIONS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "prob"])
            writer.writeheader()
            writer.writerows(prediction_rows)

        mlflow.log_metrics(
            {f"test_{k}": report["test"][k] for k in ("accuracy", "f1", "auroc", "loss")}
        )
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_artifact(str(METRICS_FILE))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
