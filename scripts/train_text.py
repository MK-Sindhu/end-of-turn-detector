"""Stage 7: train the text end-of-turn model.

Same clips, same splits and same labels as the audio model, but this model
only sees the transcript. Because both models are scored on identical test
clips, their predictions can later be combined (fusion) and compared fairly.

Outputs:
  models/text/                     best model and its tokenizer
  results/text_metrics.json        validation and test metrics with 95% CIs
  results/text_predictions.csv     P(complete) for every val and test clip
"""

import argparse
import csv
import json
import shutil

import mlflow
import numpy as np
import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from indic_eot.config import REPO_ROOT, load_params
from indic_eot.evaluation import split_report
from indic_eot.text_model import encode, load_model, load_tokenizer
from indic_eot.training import classification_metrics, flatten, load_rows, start_mlflow_run

MODEL_DIR = REPO_ROOT / "models" / "text"
METRICS_FILE = REPO_ROOT / "results" / "text_metrics.json"
PREDICTIONS_FILE = REPO_ROOT / "results" / "text_predictions.csv"


def batches(rows: list[dict], size: int, order: np.ndarray | None = None):
    index = order if order is not None else np.arange(len(rows))
    for i in range(0, len(index), size):
        yield [rows[j] for j in index[i : i + size]]


def predict(model, tokenizer, rows, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    with torch.inference_mode():
        for batch in batches(rows, 128):
            inputs = encode(tokenizer, [r["text"] for r in batch]).to(device)
            probs.append(torch.sigmoid(model(**inputs).logits.squeeze(-1)).cpu())
    labels = np.array([int(r["label"]) for r in rows])
    return labels, torch.cat(probs).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="use only N clips per split (quick trial)")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--experiment", help="override the MLflow experiment name")
    args = parser.parse_args()

    params = load_params()
    cfg = params["train_text"]
    epochs = args.epochs or cfg["epochs"]
    torch.manual_seed(params["seed"])
    rng = np.random.default_rng(params["seed"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    rows = {split: load_rows(split, args.limit) for split in ("train", "val", "test")}
    tokenizer = load_tokenizer(cfg["encoder"])
    model = load_model(cfg["encoder"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    steps_per_epoch = len(rows["train"]) // cfg["batch_size"]
    total_steps = epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg["warmup_fraction"] * total_steps), total_steps
    )

    experiment = args.experiment or params["mlflow"]["experiment"]
    with start_mlflow_run(experiment, run_name="text"):
        mlflow.log_params(flatten({"train_text": cfg, "seed": params["seed"]}))
        mlflow.log_params({"device": device, "train_clips": len(rows["train"])})

        best_auroc, best_epoch, step = -1.0, 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            order = rng.permutation(len(rows["train"]))[: steps_per_epoch * cfg["batch_size"]]
            running = []
            for batch in tqdm(
                batches(rows["train"], cfg["batch_size"], order),
                total=steps_per_epoch,
                desc=f"epoch {epoch}/{epochs}",
            ):
                inputs = encode(tokenizer, [r["text"] for r in batch]).to(device)
                target = torch.tensor([float(r["label"]) for r in batch], device=device)
                logits = model(**inputs).logits.squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running.append(loss.item())
                step += 1
                if step % 20 == 0:
                    mlflow.log_metric("train_loss", float(np.mean(running[-20:])), step=step)

            labels, probs = predict(model, tokenizer, rows["val"], device)
            val = classification_metrics(labels, probs)
            mlflow.log_metrics(
                {"val_accuracy": val["accuracy"], "val_auroc": val["auroc"]}, step=epoch
            )
            print(
                f"epoch {epoch}: train_loss {np.mean(running):.4f}  "
                f"val_acc {val['accuracy']:.4f}  val_auroc {val['auroc']:.4f}"
            )
            if val["auroc"] > best_auroc:
                best_auroc, best_epoch = val["auroc"], epoch
                shutil.rmtree(MODEL_DIR, ignore_errors=True)
                model.save_pretrained(MODEL_DIR)
                tokenizer.save_pretrained(MODEL_DIR)

        # Final scoring with the best epoch's weights.
        model = load_model(MODEL_DIR).to(device)
        report = {"best_epoch": best_epoch, "encoder": cfg["encoder"]}
        prediction_rows = []
        for split in ("val", "test"):
            labels, probs = predict(model, tokenizer, rows[split], device)
            report[split] = split_report(labels, probs, rows[split])
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

        mlflow.log_metrics({f"test_{k}": report["test"][k] for k in ("accuracy", "f1", "auroc")})
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_artifact(str(METRICS_FILE))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
