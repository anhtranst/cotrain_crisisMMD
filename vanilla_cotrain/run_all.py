"""Batch runner: execute all 12 (budget x seed_set) experiments for one task/modality."""

import json
import logging
import statistics
import time
from pathlib import Path

from .config import VanillaCoTrainConfig

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

logger = logging.getLogger("vanilla_cotrain")


def run_all_experiments(
    task,
    modality,
    *,
    budgets=None,
    seed_sets=None,
    run_id=None,
    model_name="vinai/bertweet-base",
    image_model_name="openai/clip-vit-base-patch32",
    image_size=224,
    num_iterations=10,
    samples_per_class=5,
    train_epochs=5,
    finetune_max_epochs=50,
    finetune_patience=5,
    batch_size=32,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_length=128,
    data_root="/workspace/data",
    results_root="/workspace/results",
    _trainer_cls=None,
    _on_experiment_done=None,
):
    """Run all budget x seed_set combinations for a given *task* and *modality*.

    Returns a list of result dicts (or ``None`` for failed experiments).
    Experiments whose ``metrics.json`` already exists are loaded and skipped.
    """
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    _common_kwargs = dict(
        run_id=run_id,
        model_name=model_name,
        image_model_name=image_model_name,
        image_size=image_size,
        num_iterations=num_iterations,
        samples_per_class=samples_per_class,
        train_epochs=train_epochs,
        finetune_max_epochs=finetune_max_epochs,
        finetune_patience=finetune_patience,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        data_root=data_root,
        results_root=results_root,
    )

    if _trainer_cls is None:
        from .trainer import VanillaCoTrainer  # lazy import
        _trainer_cls = VanillaCoTrainer

    all_results = []
    total = len(budgets) * len(seed_sets)
    completed = skipped = failed = 0
    start_time = time.time()

    for budget in budgets:
        for seed_set in seed_sets:
            idx = completed + skipped + failed + 1

            # Build metrics path (no pseudo_label_source level)
            _metrics_base = Path(results_root) / "cotrain" / "vanilla-cotrain"
            if run_id is not None:
                _metrics_base = _metrics_base / run_id
            metrics_path = (
                _metrics_base / task / modality
                / f"{budget}_set{seed_set}" / "metrics.json"
            )

            # Resume: reuse existing results
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result = json.load(f)
                all_results.append(result)
                skipped += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- SKIPPED (exists)"
                )
                if _on_experiment_done is not None:
                    _on_experiment_done(task, modality, budget, seed_set, "skipped")
                continue

            print(f"[{idx}/{total}] budget={budget}, seed={seed_set} -- starting...")
            config = VanillaCoTrainConfig(
                task=task,
                modality=modality,
                budget=budget,
                seed_set=seed_set,
                **_common_kwargs,
            )

            try:
                trainer = _trainer_cls(config)
                result = trainer.run()
                all_results.append(result)
                completed += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- done (macro_f1={result['test_macro_f1']:.4f})"
                )
                if _on_experiment_done is not None:
                    _on_experiment_done(task, modality, budget, seed_set, "done")
            except Exception as e:
                logger.error(
                    f"Experiment budget={budget}, seed={seed_set} failed: {e}"
                )
                all_results.append(None)
                failed += 1
                if _on_experiment_done is not None:
                    _on_experiment_done(task, modality, budget, seed_set, "failed")

            # Free GPU memory between experiments
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    elapsed = time.time() - start_time
    print(
        f"\nBatch complete: {completed} ran, {skipped} skipped, {failed} failed"
        f" ({elapsed:.1f}s total)"
    )
    return all_results


def format_summary_table(all_results, task, modality, budgets=None, seed_sets=None):
    """Return a formatted summary table grouped by budget."""
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    lookup = {}
    for r in all_results:
        if r is not None:
            lookup[(r["budget"], r["seed_set"])] = r

    lines = []
    lines.append(f"=== Vanilla Co-Training Results for {task}/{modality} ===")
    lines.append("")

    seed_hdrs = "".join(f"  Seed {s:<13}" for s in seed_sets)
    lines.append(f"{'Budget':>6}  {seed_hdrs}  {'Mean':>8}  {'Std':>8}")

    sub_cells = "".join(f"  {'ErrR%':>6} {'MacF1':>6}" for _ in seed_sets)
    lines.append(f"{'':>6}  {sub_cells}  {'ErrR%':>8}  {'MacF1':>8}")
    lines.append("-" * len(lines[-1]))

    for budget in budgets:
        err_rates = []
        macro_f1s = []
        cells = ""
        for seed_set in seed_sets:
            r = lookup.get((budget, seed_set))
            if r is not None:
                cells += f"  {r['test_error_rate']:>6.2f} {r['test_macro_f1']:>6.4f}"
                err_rates.append(r["test_error_rate"])
                macro_f1s.append(r["test_macro_f1"])
            else:
                cells += f"  {'N/A':>6} {'N/A':>6}"

        if len(err_rates) >= 2:
            e_mean = statistics.mean(err_rates)
            e_std = statistics.stdev(err_rates)
            f_mean = statistics.mean(macro_f1s)
            f_std = statistics.stdev(macro_f1s)
            agg = f"  {e_mean:>5.2f}+/-{e_std:<5.2f}  {f_mean:.4f}+/-{f_std:.4f}"
        elif len(err_rates) == 1:
            agg = f"  {err_rates[0]:>8.2f}  {macro_f1s[0]:>8.4f}"
        else:
            agg = f"  {'N/A':>8}  {'N/A':>8}"

        lines.append(f"{budget:>6}  {cells}{agg}")

    return "\n".join(lines)
