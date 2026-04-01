#!/usr/bin/env python3
"""Backfill avg_confidence and avg_entropy into existing metrics.json files.

Reads confidence/entropy from predictions.tsv and adds avg_confidence/avg_entropy
to the corresponding metrics.json. Safe to run multiple times (idempotent).

Usage:
    python scripts/backfill_metrics.py
"""

import csv
import json
from pathlib import Path


def backfill(results_root: str):
    root = Path(results_root) / "zeroshot"
    if not root.exists():
        print(f"No zeroshot directory at {root}")
        return

    count = 0
    for metrics_path in sorted(root.rglob("metrics.json")):
        pred_path = metrics_path.parent / "predictions.tsv"
        if not pred_path.exists():
            continue

        # Read predictions to compute averages
        conf_vals, ent_vals = [], []
        with open(pred_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                c = row.get("confidence")
                e = row.get("entropy")
                if c:
                    conf_vals.append(float(c))
                if e:
                    ent_vals.append(float(e))

        avg_conf = round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else 0.0
        avg_ent = round(sum(ent_vals) / len(ent_vals), 4) if ent_vals else 0.0

        # Update metrics.json
        with open(metrics_path) as f:
            metrics = json.load(f)

        metrics["avg_confidence"] = avg_conf
        metrics["avg_entropy"] = avg_ent

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"  OK: {metrics_path.relative_to(root)} — conf={avg_conf}, ent={avg_ent}")
        count += 1

    print(f"\nDone: {count} metrics.json files updated")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    backfill(str(repo_root / "results"))
