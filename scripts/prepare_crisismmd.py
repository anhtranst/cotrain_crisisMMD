"""Preprocess CrisisMMD into per-task, per-modality datasets.

Reads the raw CrisisMMD TSV files (agreed-label subset) and produces
cleaned datasets at: data/CrisisMMD/tasks/{task}/{modality}/{split}.tsv

Two tasks:
  - informative  (2 classes, has label_text and label_image)
  - humanitarian (5 classes, has label_text and label_image)

Three modalities:
  - text_only   — deduplicated by tweet_id
  - image_only  — one row per image_id
  - text_image  — one row per image_id, paired with tweet text

Also generates labeled/unlabeled budget splits for each modality.
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Source directory
SOURCE_DIR = Path(__file__).parent.parent / "data" / "CrisisMMD"
ORIGINAL_DIR = SOURCE_DIR / "original"
OUTPUT_DIR = SOURCE_DIR / "tasks"

TASKS = {
    "informative": {
        "prefix": "task_informative_text_img_agreed_lab",
        "has_label_text": True,
    },
    "humanitarian": {
        "prefix": "task_humanitarian_text_img_agreed_lab",
        "has_label_text": True,
    },
}

SPLITS = ["train", "dev", "test"]
BUDGETS = [5, 10, 25, 50]
SEEDS = [1, 2, 3]


def load_source(task_name: str, split: str) -> pd.DataFrame:
    """Load a raw CrisisMMD TSV file."""
    prefix = TASKS[task_name]["prefix"]
    path = ORIGINAL_DIR / f"{prefix}_{split}.tsv"
    df = pd.read_csv(path, sep="\t", dtype={"tweet_id": str, "image_id": str})
    logger.info(f"Loaded {path.name}: {len(df)} rows")
    return df


def resolve_image_path(relative_path: str) -> str:
    """Convert relative image path to absolute path."""
    return str(SOURCE_DIR / relative_path)


def extract_class_label(df: pd.DataFrame, task_name: str, modality: str) -> pd.Series:
    """Pick the right label column based on task and modality."""
    has_label_text = TASKS[task_name]["has_label_text"]

    if not has_label_text:
        # Damage task: only 'label' column exists (image-based)
        return df["label"]

    if modality == "text_only":
        return df["label_text"]
    elif modality == "image_only":
        return df["label_image"]
    else:  # text_image
        # For multimodal, use label_text as the ground truth
        return df["label_text"]


def prepare_text_only(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """Prepare text-only dataset: deduplicate by tweet_id."""
    out = df.drop_duplicates(subset="tweet_id", keep="first").copy()
    out["class_label"] = extract_class_label(out, task_name, "text_only")
    out = out[["tweet_id", "tweet_text", "class_label"]].reset_index(drop=True)
    return out


def prepare_image_only(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """Prepare image-only dataset: one row per image_id."""
    out = df.copy()
    out["class_label"] = extract_class_label(out, task_name, "image_only")
    out["image_path"] = out["image"].apply(resolve_image_path)
    out = out[["image_id", "image_path", "class_label"]].reset_index(drop=True)
    return out


def prepare_text_image(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """Prepare text+image dataset: one row per image_id with tweet text."""
    out = df.copy()
    out["class_label"] = extract_class_label(out, task_name, "text_image")
    out["image_path"] = out["image"].apply(resolve_image_path)
    out = out[["tweet_id", "image_id", "tweet_text", "image_path", "class_label"]]
    return out.reset_index(drop=True)


def generate_budget_splits(
    df: pd.DataFrame, budgets: list, seeds: list, output_dir: Path, id_col: str
):
    """Generate stratified labeled/unlabeled splits for each (budget, seed).

    Args:
        df: Full training set DataFrame.
        budgets: List of budget levels (samples per class).
        seeds: List of random seeds.
        output_dir: Directory to write split files.
        id_col: Primary ID column name (tweet_id or image_id).
    """
    for budget in budgets:
        for seed in seeds:
            rng = random.Random(seed)
            labeled_indices = []
            warnings = []

            for label in sorted(df["class_label"].unique()):
                class_idx = df.index[df["class_label"] == label].tolist()
                rng.shuffle(class_idx)

                if len(class_idx) < budget:
                    warnings.append(
                        f"  class '{label}': only {len(class_idx)} samples "
                        f"(budget={budget}), using all"
                    )
                    labeled_indices.extend(class_idx)
                else:
                    labeled_indices.extend(class_idx[:budget])

            unlabeled_indices = sorted(set(df.index) - set(labeled_indices))

            df_labeled = df.loc[labeled_indices].reset_index(drop=True)
            df_unlabeled = df.loc[unlabeled_indices].reset_index(drop=True)

            labeled_path = output_dir / f"labeled_{budget}_set{seed}.tsv"
            unlabeled_path = output_dir / f"unlabeled_{budget}_set{seed}.tsv"

            df_labeled.to_csv(labeled_path, sep="\t", index=False)
            df_unlabeled.to_csv(unlabeled_path, sep="\t", index=False)

            if warnings:
                logger.warning(
                    f"Budget={budget}, seed={seed} — best-effort sampling:\n"
                    + "\n".join(warnings)
                )
            logger.info(
                f"  budget={budget} seed={seed}: "
                f"labeled={len(df_labeled)}, unlabeled={len(df_unlabeled)}"
            )


def process_task(task_name: str, budgets: list, seeds: list):
    """Process one task across all modalities."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing task: {task_name}")
    logger.info(f"{'='*60}")

    # Load source data for all splits
    dfs = {}
    for split in SPLITS:
        dfs[split] = load_source(task_name, split)

    # Define modality processors and their ID columns
    modalities = {
        "text_only": (prepare_text_only, "tweet_id"),
        "image_only": (prepare_image_only, "image_id"),
        "text_image": (prepare_text_image, "image_id"),
    }

    for modality, (prepare_fn, id_col) in modalities.items():
        logger.info(f"\n--- {task_name} / {modality} ---")

        mod_dir = OUTPUT_DIR / task_name / modality
        mod_dir.mkdir(parents=True, exist_ok=True)

        all_empty = True
        for split in SPLITS:
            df_out = prepare_fn(dfs[split], task_name)
            if df_out.empty:
                break  # This modality is not applicable for this task

            out_path = mod_dir / f"{split}.tsv"
            df_out.to_csv(out_path, sep="\t", index=False)
            logger.info(f"Wrote {out_path.name}: {len(df_out)} rows")

            # Print class distribution
            dist = df_out["class_label"].value_counts().sort_index()
            for label, count in dist.items():
                logger.info(f"  {label}: {count}")

            all_empty = False

        if all_empty:
            logger.info(f"Skipped {modality} for {task_name} (not applicable)")
            continue

        # Generate budget splits from train set
        train_path = mod_dir / "train.tsv"
        if train_path.exists():
            df_train = pd.read_csv(
                train_path, sep="\t", dtype={id_col: str}
            )
            logger.info(f"\nGenerating budget splits for {task_name}/{modality}:")
            generate_budget_splits(df_train, budgets, seeds, mod_dir, id_col)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CrisisMMD v2.0 into per-task, per-modality datasets."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASKS.keys()),
        default=list(TASKS.keys()),
        help="Tasks to process (default: all)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=BUDGETS,
        help=f"Budget levels for labeled splits (default: {BUDGETS})",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Random seeds for splits (default: {SEEDS})",
    )
    args = parser.parse_args()

    logger.info(f"Source: {ORIGINAL_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Budgets: {args.budgets}")
    logger.info(f"Seeds: {args.seeds}")

    for task_name in args.tasks:
        process_task(task_name, args.budgets, args.seeds)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
