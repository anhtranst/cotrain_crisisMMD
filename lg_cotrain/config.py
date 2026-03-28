"""Configuration dataclass for LG-CoTrain experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LGCoTrainConfig:
    # Experiment identifiers
    task: str = "humanitarian"       # informative | humanitarian
    modality: str = "text_only"      # text_only | image_only | text_image
    method: str = "lg-cotrain"       # lg-cotrain | co-teaching | co-teaching-plus | vanilla-cotrain | dividemix
    budget: int = 5
    seed_set: int = 1

    # Pseudo-label source model (folder name under data/pseudo_labelled/)
    pseudo_label_source: str = "llama-3.2-11b"

    # Model
    model_name: str = "vinai/bertweet-base"
    num_labels: int = 5
    max_seq_length: int = 128

    # Phase 1: Weight generation
    weight_gen_epochs: int = 7

    # Phase 2: Co-training
    cotrain_epochs: int = 10

    # Phase 3: Fine-tuning
    finetune_max_epochs: int = 100
    finetune_patience: int = 5
    # Phase 3 early stopping strategy: "baseline" | "no_early_stopping" | "per_class_patience"
    #                                  | "weighted_macro_f1" | "balanced_dev" | "scaled_threshold"
    stopping_strategy: str = "baseline"
    # Phase 1 → Phase 2 seeding: "last" (default, per Algorithm 1) | "best" (best ensemble dev F1)
    phase1_seed_strategy: str = "last"

    # Optimization
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Device override: "cuda:0", "cuda:1", "cpu", or None (auto-detect)
    device: Optional[str] = None

    # Paths (base) — default to sibling directories of this package, which
    # resolves correctly on any OS regardless of where the repo is cloned.
    data_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "data")
    )
    results_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "results")
    )

    # Auto-computed paths (set in __post_init__)
    labeled_path: str = field(init=False, default="")
    unlabeled_path: str = field(init=False, default="")
    pseudo_label_path: str = field(init=False, default="")
    dev_path: str = field(init=False, default="")
    test_path: str = field(init=False, default="")
    output_dir: str = field(init=False, default="")

    def __post_init__(self):
        task_dir = (
            Path(self.data_root) / "CrisisMMD" / "tasks"
            / self.task / self.modality
        )
        pseudo_dir = (
            Path(self.data_root) / "pseudo_labelled" / self.pseudo_label_source
            / self.task / self.modality
        )

        self.labeled_path = str(
            task_dir / f"labeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.unlabeled_path = str(
            task_dir / f"unlabeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.pseudo_label_path = str(pseudo_dir / "train_pred.tsv")
        self.dev_path = str(task_dir / "dev.tsv")
        self.test_path = str(task_dir / "test.tsv")
        self.output_dir = str(
            Path(self.results_root) / "cotrain" / self.method
            / self.pseudo_label_source / self.task / self.modality
            / f"{self.budget}_set{self.seed_set}"
        )
