#!/usr/bin/env python3
"""Create pseudo-label files from zero-shot train predictions.

Reads predictions from results/zeroshot/{model}/{task}/{modality}/train/predictions.tsv
and writes to data/pseudo_labelled/{model}/{task}/{modality}/train_pred.tsv
in the format expected by the co-training pipeline.

Usage:
    python scripts/create_pseudo_labels.py
    python scripts/create_pseudo_labels.py --model llama-3.2-11b
    python scripts/create_pseudo_labels.py --model llama-3.2-11b --task humanitarian --modality text_only
"""

import argparse
import csv
import os
from pathlib import Path

TASKS = ["informative", "humanitarian"]
MODALITIES = ["text_only", "image_only", "text_image"]


def create_pseudo_labels(model, task, modality, results_root, data_root):
    """Convert zero-shot train predictions to pseudo-label format.

    Input:  results/zeroshot/{model}/{task}/{modality}/train/predictions.tsv
    Output: data/pseudo_labelled/{model}/{task}/{modality}/train_pred.tsv

    The co-training pipeline expects columns:
        tweet_id, tweet_text, predicted_label, confidence

    Confidence scores are carried over from the zero-shot predictions.
    Falls back to 1.0 if not present in the source predictions.
    """
    pred_path = (
        Path(results_root) / "zeroshot" / model / task / modality
        / "train" / "predictions.tsv"
    )
    if not pred_path.exists():
        print(f"  SKIP: {task}/{modality} — no train predictions found")
        return False

    out_dir = Path(data_root) / "pseudo_labelled" / model / task / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_pred.tsv"

    # Read predictions
    rows = []
    with open(pred_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        in_columns = reader.fieldnames
        for row in reader:
            out_row = {}
            # Copy ID column (tweet_id for text_only/text_image, image_id for image_only)
            if "tweet_id" in row:
                out_row["tweet_id"] = row["tweet_id"]
            if "image_id" in row:
                out_row["image_id"] = row["image_id"]
            # Copy text if available
            if "tweet_text" in row:
                out_row["tweet_text"] = row["tweet_text"]
            # Copy image path if available
            if "image_path" in row:
                out_row["image_path"] = row["image_path"]
            # Add pseudo-label columns
            out_row["predicted_label"] = row["predicted_label"]
            out_row["confidence"] = row.get("confidence", 1.0)
            rows.append(out_row)

    # Write output
    if not rows:
        print(f"  SKIP: {task}/{modality} — no rows in predictions")
        return False

    out_fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  OK: {task}/{modality} — {len(rows)} pseudo-labels -> {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create pseudo-label files from zero-shot train predictions"
    )
    parser.add_argument("--model", default="llama-3.2-11b",
                        help="Model slug (default: llama-3.2-11b)")
    parser.add_argument("--task", choices=TASKS, default=None,
                        help="Specific task (default: all)")
    parser.add_argument("--modality", choices=MODALITIES, default=None,
                        help="Specific modality (default: all)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_root = args.data_root or str(repo_root / "data")
    results_root = args.results_root or str(repo_root / "results")

    tasks = [args.task] if args.task else TASKS
    modalities = [args.modality] if args.modality else MODALITIES

    print(f"Creating pseudo-labels from: {results_root}/zeroshot/{args.model}/")
    print(f"Writing to: {data_root}/pseudo_labelled/{args.model}/\n")

    count = 0
    for task in tasks:
        for modality in modalities:
            if create_pseudo_labels(args.model, task, modality, results_root, data_root):
                count += 1

    print(f"\nDone: {count} pseudo-label files created")


if __name__ == "__main__":
    main()
