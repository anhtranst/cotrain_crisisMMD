"""CLI entry point for running Vanilla Co-Training experiments.

Supports single experiments or batch mode with multiple budgets/seeds.
"""

import argparse

from .run_all import BUDGETS, SEED_SETS, format_summary_table, run_all_experiments


def main():
    parser = argparse.ArgumentParser(description="Vanilla Co-Training experiment runner")

    # Task and modality
    parser.add_argument(
        "--task", type=str, default="humanitarian",
        choices=["informative", "humanitarian"],
        help="Classification task (default: humanitarian)",
    )
    parser.add_argument(
        "--modality", type=str, default="text_only",
        choices=["text_only", "image_only", "text_image"],
        help="Data modality (default: text_only)",
    )

    # Budget(s)
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget", type=int, dest="budgets", nargs=1,
        help="Single budget value",
    )
    budget_group.add_argument(
        "--budgets", type=int, nargs="+", default=None,
        help="One or more budget values (default: all [5, 10, 25, 50])",
    )

    # Seed set(s)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-set", type=int, dest="seed_sets", nargs=1,
        help="Single seed set",
    )
    seed_group.add_argument(
        "--seed-sets", type=int, nargs="+", default=None,
        help="One or more seed sets (default: all [1, 2, 3])",
    )

    # Run ID and output
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (e.g. run-1). Inserted into output path.",
    )
    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Output folder for results (overrides --results-root).",
    )

    # Model hyperparameters
    parser.add_argument("--model-name", type=str, default="vinai/bertweet-base",
                        help="Text model (default: vinai/bertweet-base)")
    parser.add_argument("--image-model-name", type=str, default="openai/clip-vit-base-patch32",
                        help="Image model for image_only/text_image")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Image input size (default: 224)")

    # Vanilla co-training specific
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of co-training rounds (default: 10)")
    parser.add_argument("--samples-per-class", type=int, default=5,
                        help="Top-k samples per class per model per iteration (default: 5)")
    parser.add_argument("--train-epochs", type=int, default=5,
                        help="Epochs per model per iteration (default: 5)")
    parser.add_argument("--finetune-max-epochs", type=int, default=50,
                        help="Max fine-tuning epochs (default: 50)")
    parser.add_argument("--finetune-patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)

    # Paths
    parser.add_argument("--data-root", type=str, default="/workspace/data")
    parser.add_argument("--results-root", type=str, default="/workspace/results")

    args = parser.parse_args()

    budgets = args.budgets
    seed_sets = args.seed_sets
    results_root = args.output_folder or args.results_root

    hyperparams = dict(
        run_id=args.run_id,
        model_name=args.model_name,
        image_model_name=args.image_model_name,
        image_size=args.image_size,
        num_iterations=args.num_iterations,
        samples_per_class=args.samples_per_class,
        train_epochs=args.train_epochs,
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_patience=args.finetune_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
    )

    all_results = run_all_experiments(
        args.task,
        args.modality,
        budgets=budgets,
        seed_sets=seed_sets,
        data_root=args.data_root,
        results_root=results_root,
        **hyperparams,
    )

    # Single experiment: show inline results
    if (
        budgets is not None and len(budgets) == 1
        and seed_sets is not None and len(seed_sets) == 1
    ):
        result = all_results[0]
        if result is not None:
            print(f"\nFinal Results:")
            print(f"  Test Error Rate: {result['test_error_rate']:.2f}%")
            print(f"  Test Macro-F1:   {result['test_macro_f1']:.4f}")
            print(f"  Iterations:      {result['num_iterations_completed']}")
    else:
        print()
        print(format_summary_table(
            all_results, args.task, args.modality,
            budgets=budgets, seed_sets=seed_sets,
        ))


if __name__ == "__main__":
    main()
