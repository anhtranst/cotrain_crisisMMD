"""Data loading, label encoding, dataset splitting, and PyTorch Dataset."""

import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger("lg_cotrain")

# Per-task label sets for CrisisMMD (agreed-label subset).
TASK_LABELS = {
    "humanitarian": sorted([
        "affected_individuals",
        "infrastructure_and_utility_damage",
        "not_humanitarian",
        "other_relevant_information",
        "rescue_volunteering_or_donation_effort",
    ]),
    "informative": sorted([
        "informative",
        "not_informative",
    ]),
}

# Default label set (humanitarian task) for backward compatibility.
CLASS_LABELS = TASK_LABELS["humanitarian"]


def detect_classes(*dataframes) -> List[str]:
    """Detect the sorted list of unique class labels across DataFrames or list-of-dicts.

    Accepts pandas DataFrames (with a 'class_label' column) or lists of dicts
    (with a 'class_label' key). Returns the alphabetically sorted union.
    """
    classes = set()
    for df in dataframes:
        if isinstance(df, list):
            classes.update(rec["class_label"] for rec in df)
        else:
            classes.update(df["class_label"].unique())
    return sorted(classes)


# Backward-compatible alias.
detect_event_classes = detect_classes


def build_label_encoder(labels=None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label-to-id and id-to-label mappings.

    Args:
        labels: Optional list of class names. If None, uses the full CLASS_LABELS.
    """
    labels = labels if labels is not None else CLASS_LABELS
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


# Expected columns per modality.
MODALITY_COLUMNS = {
    "text_only": {"tweet_id", "tweet_text", "class_label"},
    "image_only": {"image_id", "image_path", "class_label"},
    "text_image": {"tweet_id", "image_id", "tweet_text", "image_path", "class_label"},
}

# --- Functions requiring pandas ---

def load_tsv(path: str, modality: str = "text_only"):
    """Load a tab-separated file, validating columns for the given modality."""
    import pandas as pd
    dtype = {}
    if modality in ("text_only", "text_image"):
        dtype["tweet_id"] = str
    if modality in ("image_only", "text_image"):
        dtype["image_id"] = str
    df = pd.read_csv(path, sep="\t", dtype=dtype)
    expected = MODALITY_COLUMNS.get(modality, MODALITY_COLUMNS["text_only"])
    if not expected.issubset(set(df.columns)):
        raise ValueError(
            f"TSV at {path} missing columns for {modality}. "
            f"Expected: {sorted(expected)}, Found: {list(df.columns)}"
        )
    return df


def load_pseudo_labels(path: str):
    """Load pseudo-label TSV/CSV with predicted_label, confidence, etc."""
    import pandas as pd
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep, dtype={"tweet_id": str, "image_id": str})
    required = {"predicted_label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Pseudo-label file at {path} missing columns. Found: {list(df.columns)}"
        )
    return df


def split_labeled_set(df, seed: int):
    """Split labeled set into D_l1 and D_l2 with stratified per-class split.

    For each class, shuffle indices and assign first half to D_l1, second to D_l2.
    Works with pandas DataFrame or list-of-dicts.
    """
    try:
        import numpy as np
        return _split_labeled_set_pandas(df, seed)
    except ImportError:
        return _split_labeled_set_pure(df, seed)


def _split_labeled_set_pandas(df, seed: int):
    """Pandas-based split implementation."""
    import numpy as np
    rng = np.random.RandomState(seed)
    idx1, idx2 = [], []

    for label in sorted(df["class_label"].unique()):
        class_indices = df.index[df["class_label"] == label].tolist()
        rng.shuffle(class_indices)
        mid = len(class_indices) // 2
        split = mid if len(class_indices) % 2 == 0 else mid + 1
        idx1.extend(class_indices[:split])
        idx2.extend(class_indices[split:])

    return df.loc[idx1].reset_index(drop=True), df.loc[idx2].reset_index(drop=True)


def _split_labeled_set_pure(records: list, seed: int):
    """Pure-Python split on list-of-dicts. Returns (list, list)."""
    rng = random.Random(seed)
    by_class = {}
    for i, rec in enumerate(records):
        label = rec["class_label"]
        by_class.setdefault(label, []).append(i)

    idx1, idx2 = [], []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        mid = len(indices) // 2
        split = mid if len(indices) % 2 == 0 else mid + 1
        idx1.extend(indices[:split])
        idx2.extend(indices[split:])

    return [records[i] for i in idx1], [records[i] for i in idx2]


def build_d_lg(df_unlabeled, df_pseudo, modality: str = "text_only"):
    """Join unlabeled data with pseudo-labels, modality-aware.

    Join key and verification column depend on modality:
    - text_only: join on tweet_id, verify tweet_text
    - image_only: join on image_id, verify image_path
    - text_image: join on image_id (unique per sample), verify image_path

    Result has predicted_label (for training) and class_label (for evaluation).
    """
    if modality == "image_only":
        join_key = "image_id"
        verify_col = "image_path"
    elif modality == "text_image":
        join_key = "image_id"
        verify_col = "image_path"
    else:
        join_key = "tweet_id"
        verify_col = "tweet_text"

    pseudo_cols = df_pseudo[[join_key, verify_col, "predicted_label", "confidence"]].copy()
    pseudo_cols = pseudo_cols.rename(columns={verify_col: f"{verify_col}_pseudo"})
    merged = df_unlabeled.merge(pseudo_cols, on=join_key, how="inner")

    n_unmatched = len(df_unlabeled) - len(merged)
    if n_unmatched > 0:
        logger.warning(
            f"build_d_lg: {n_unmatched} unlabeled samples had no matching pseudo-label"
        )

    mismatch = merged[verify_col].str.strip() != merged[f"{verify_col}_pseudo"].str.strip()
    if mismatch.any():
        mismatched_ids = merged.loc[mismatch, join_key].tolist()
        logger.warning(
            f"build_d_lg: {mismatch.sum()} entries have mismatched {verify_col} "
            f"between unlabeled TSV and pseudo-label file ({join_key}s: {mismatched_ids})"
        )

    return merged.drop(columns=[f"{verify_col}_pseudo"]).reset_index(drop=True)


# --- TweetDataset requires torch + transformers ---

def _get_tweet_dataset_class():
    """Lazy import to avoid requiring torch/transformers at module load."""
    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer

    class TweetDataset(Dataset):
        """PyTorch Dataset for tokenized tweets with labels."""

        def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 128,
        ):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "sample_idx": torch.tensor(idx, dtype=torch.long),
            }

    return TweetDataset


# Expose TweetDataset as a module-level name that's lazy
class TweetDataset:
    """Proxy that lazily imports the real TweetDataset when instantiated."""
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _get_tweet_dataset_class()
        return cls._real_class(*args, **kwargs)


# --- ImageDataset requires torch + PIL + transformers ---

def _get_image_dataset_class():
    """Lazy import to avoid requiring torch/PIL at module load."""
    import torch
    from pathlib import Path
    from torch.utils.data import Dataset

    class ImageDataset(Dataset):
        """PyTorch Dataset for images with labels, using CLIP image processor."""

        def __init__(
            self,
            image_paths: List[str],
            labels: List[int],
            project_root: str,
            image_processor,
        ):
            self.image_paths = image_paths
            self.labels = labels
            self.project_root = Path(project_root)
            self.image_processor = image_processor

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            from PIL import Image

            img_path = self.project_root / self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                logger.warning(f"Failed to load image {img_path}, using black image")
                image = Image.new("RGB", (224, 224), (0, 0, 0))

            processed = self.image_processor(images=image, return_tensors="pt")
            return {
                "pixel_values": processed["pixel_values"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "sample_idx": torch.tensor(idx, dtype=torch.long),
            }

    return ImageDataset


class ImageDataset:
    """Proxy that lazily imports the real ImageDataset when instantiated."""
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _get_image_dataset_class()
        return cls._real_class(*args, **kwargs)


# --- MultimodalDataset requires torch + PIL + transformers ---

def _get_multimodal_dataset_class():
    """Lazy import to avoid requiring torch/PIL at module load."""
    import torch
    from pathlib import Path
    from torch.utils.data import Dataset

    class MultimodalDataset(Dataset):
        """PyTorch Dataset for text+image pairs with labels."""

        def __init__(
            self,
            texts: List[str],
            image_paths: List[str],
            labels: List[int],
            tokenizer,
            image_processor,
            project_root: str,
            max_length: int = 128,
        ):
            self.texts = texts
            self.image_paths = image_paths
            self.labels = labels
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.project_root = Path(project_root)
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            from PIL import Image

            # Text branch
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Image branch
            img_path = self.project_root / self.image_paths[idx]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                logger.warning(f"Failed to load image {img_path}, using black image")
                image = Image.new("RGB", (224, 224), (0, 0, 0))

            processed = self.image_processor(images=image, return_tensors="pt")

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "sample_idx": torch.tensor(idx, dtype=torch.long),
            }

    return MultimodalDataset


class MultimodalDataset:
    """Proxy that lazily imports the real MultimodalDataset when instantiated."""
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _get_multimodal_dataset_class()
        return cls._real_class(*args, **kwargs)
