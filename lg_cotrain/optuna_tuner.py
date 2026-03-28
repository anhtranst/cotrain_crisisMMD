"""Global Optuna hyperparameter tuner for LG-CoTrain.

Runs a single Optuna study where each trial executes the full 3-phase pipeline
for a given task/modality (budget=50, seed=1 by default), optimizing
dev macro-F1.  The best hyperparameters can then be applied manually to
``run_experiment.py`` via CLI flags.

Usage::

    python -m lg_cotrain.optuna_tuner --n-trials 20
    python -m lg_cotrain.optuna_tuner --n-trials 20 --task informative --modality text_only
    python -m lg_cotrain.optuna_tuner --n-trials 20 --storage sqlite:///optuna.db
"""

import argparse
import logging
from typing import Dict, List, Optional

from .config import LGCoTrainConfig

logger = logging.getLogger("lg_cotrain")


def create_objective(
    task: str = "humanitarian",
    modality: str = "text_only",
    budget: int = 50,
    seed_set: int = 1,
    data_root: str = "/workspace/data",
    results_root: str = "/workspace/results/optuna",
    fixed_params: Optional[Dict] = None,
    _trainer_cls=None,
):
    """Return an Optuna objective function (closure).

    Each call to the returned function:

    1. Samples hyperparameters from the Optuna trial.
    2. Runs the full 3-phase pipeline for the given task/modality.
    3. Returns ``dev_macro_f1``.

    Parameters
    ----------
    task : str
        Classification task (informative, humanitarian, damage).
    modality : str
        Data modality (text_only, image_only, text_image).
    budget, seed_set : int
        Fixed experiment identifiers (default 50 / 1).
    data_root, results_root : str
        Base directories for input data and trial outputs.
    fixed_params : dict, optional
        Extra keyword arguments forwarded to ``LGCoTrainConfig``.
    _trainer_cls : class, optional
        Override the trainer class (for testing with mocks).
    """

    def objective(trial):
        # Sample hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        cotrain_epochs = trial.suggest_int("cotrain_epochs", 5, 20)
        finetune_patience = trial.suggest_int("finetune_patience", 4, 10)

        cfg = LGCoTrainConfig(
            task=task,
            modality=modality,
            budget=budget,
            seed_set=seed_set,
            lr=lr,
            batch_size=batch_size,
            cotrain_epochs=cotrain_epochs,
            finetune_patience=finetune_patience,
            data_root=data_root,
            results_root=results_root,
            **(fixed_params or {}),
        )

        if _trainer_cls is None:
            from .trainer import LGCoTrainer
            trainer = LGCoTrainer(cfg)
        else:
            trainer = _trainer_cls(cfg)

        result = trainer.run()
        return result["dev_macro_f1"]

    return objective


def run_study(
    n_trials: int = 20,
    task: str = "humanitarian",
    modality: str = "text_only",
    budget: int = 50,
    seed_set: int = 1,
    data_root: str = "/workspace/data",
    results_root: str = "/workspace/results/optuna",
    study_name: str = "lg_cotrain_global",
    storage: Optional[str] = None,
    fixed_params: Optional[Dict] = None,
    _trainer_cls=None,
):
    """Create and run a global Optuna study.

    Returns the completed ``optuna.Study`` object.
    """
    import optuna

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )

    objective = create_objective(
        task=task,
        modality=modality,
        budget=budget,
        seed_set=seed_set,
        data_root=data_root,
        results_root=results_root,
        fixed_params=fixed_params,
        _trainer_cls=_trainer_cls,
    )

    study.optimize(objective, n_trials=n_trials)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Study complete: {len(study.trials)} trials")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best dev macro-F1: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}")

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best dev F1: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study


def main():
    """CLI entry point: ``python -m lg_cotrain.optuna_tuner``."""
    parser = argparse.ArgumentParser(
        description="Global Optuna hyperparameter tuner for LG-CoTrain. "
        "Finds the best lr, batch_size, cotrain_epochs, and finetune_patience "
        "by maximizing dev macro-F1.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of Optuna trials to run (default: 20)",
    )
    parser.add_argument(
        "--task", type=str, default="humanitarian",
        choices=["informative", "humanitarian", "damage"],
        help="Classification task (default: humanitarian)",
    )
    parser.add_argument(
        "--modality", type=str, default="text_only",
        choices=["text_only", "image_only", "text_image"],
        help="Data modality (default: text_only)",
    )
    parser.add_argument(
        "--budget", type=int, default=50,
        help="Budget level for tuning experiments (default: 50)",
    )
    parser.add_argument(
        "--seed-set", type=int, default=1,
        help="Seed set for tuning experiments (default: 1)",
    )
    parser.add_argument(
        "--data-root", type=str, default="/workspace/data",
    )
    parser.add_argument(
        "--results-root", type=str, default="/workspace/results/optuna",
    )
    parser.add_argument(
        "--study-name", type=str, default="lg_cotrain_global",
        help="Optuna study name (default: lg_cotrain_global)",
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help="Optuna storage URL for persistence, e.g. sqlite:///optuna.db",
    )

    args = parser.parse_args()

    run_study(
        n_trials=args.n_trials,
        task=args.task,
        modality=args.modality,
        budget=args.budget,
        seed_set=args.seed_set,
        data_root=args.data_root,
        results_root=args.results_root,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
