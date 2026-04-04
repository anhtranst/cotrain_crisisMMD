"""Configuration dataclass for Vanilla Co-Training experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VanillaCoTrainConfig:
    # Experiment identifiers
    task: str = "humanitarian"       # informative | humanitarian
    modality: str = "text_only"      # text_only | image_only | text_image
    method: str = "vanilla-cotrain"
    budget: int = 5
    seed_set: int = 1

    # Run identifier for grouping experiment results (e.g. "run-1", "run-2").
    # None (default) omits it from the output path.
    run_id: Optional[str] = None

    # Model
    model_name: str = "vinai/bertweet-base"
    image_model_name: str = "openai/clip-vit-base-patch32"
    num_labels: int = 5
    max_seq_length: int = 128
    image_size: int = 224

    # Vanilla co-training specific
    num_iterations: int = 10         # Number of co-training rounds
    samples_per_class: int = 5       # Top-k per class per model per iteration
    train_epochs: int = 5            # Epochs per model per iteration

    # Fine-tuning (final phase)
    finetune_max_epochs: int = 50
    finetune_patience: int = 5

    # Optimization
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Debug: cap the number of unlabeled samples.
    # None (default) uses all. Set to e.g. 200 for fast smoke tests.
    max_unlabeled_samples: Optional[int] = None

    # Device override: "cuda:0", "cuda:1", "cpu", or None (auto-detect)
    device: Optional[str] = None

    # Paths (base)
    project_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent)
    )
    data_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "data")
    )
    results_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "results")
    )

    # Auto-computed paths (set in __post_init__)
    labeled_path: str = field(init=False, default="")
    unlabeled_path: str = field(init=False, default="")
    dev_path: str = field(init=False, default="")
    test_path: str = field(init=False, default="")
    output_dir: str = field(init=False, default="")
    log_dir: str = field(init=False, default="")

    def __post_init__(self):
        task_dir = (
            Path(self.data_root) / "CrisisMMD" / "tasks"
            / self.task / self.modality
        )

        self.labeled_path = str(
            task_dir / f"labeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.unlabeled_path = str(
            task_dir / f"unlabeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.dev_path = str(task_dir / "dev.tsv")
        self.test_path = str(task_dir / "test.tsv")

        # Output path: no pseudo_label_source level (vanilla generates its own)
        output_base = Path(self.results_root) / "cotrain" / self.method
        if self.run_id is not None:
            output_base = output_base / self.run_id
        self.log_dir = str(output_base / self.task / self.modality)
        self.output_dir = str(
            output_base / self.task / self.modality
            / f"{self.budget}_set{self.seed_set}"
        )
