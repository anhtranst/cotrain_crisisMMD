"""Tests for run_experiment.py — CLI entry point.

Pure-Python tests: no torch/numpy/transformers required.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace")


def _make_result(budget=5, seed_set=1):
    """Build a result dict matching trainer.run() output."""
    return {
        "task": "humanitarian",
        "modality": "text_only",
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": 40.0,
        "test_macro_f1": 0.50,
        "test_per_class_f1": [0.5] * 8,
        "dev_error_rate": 39.0,
        "dev_macro_f1": 0.51,
        "lambda1_mean": 0.7,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.5,
        "lambda2_std": 0.1,
    }


class TestSingleExperimentMode(unittest.TestCase):
    """Test that --task X --modality Y --budget Z --seed-set W runs a single experiment."""

    def test_single_experiment_calls_run_all(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--budget", "5", "--seed-set", "1",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], "humanitarian")
        self.assertEqual(args[1], "text_only")
        self.assertEqual(kwargs["budgets"], [5])
        self.assertEqual(kwargs["seed_sets"], [1])

    def test_pseudo_label_source_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--budget", "5", "--seed-set", "1",
            "--pseudo-label-source", "llama-3",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["pseudo_label_source"], "llama-3")

    def test_output_folder_overrides_results_root(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--budget", "5", "--seed-set", "1",
            "--output-folder", "/custom/output",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["results_root"], "/custom/output")


class TestBatchMode(unittest.TestCase):
    """Test that omitting --budget/--seed-set triggers batch mode."""

    def test_all_budgets_all_seeds_batch(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertIsNone(kwargs["budgets"])
        self.assertIsNone(kwargs["seed_sets"])

    def test_custom_budgets_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--budgets", "5", "10",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["budgets"], [5, 10])

    def test_custom_seed_sets_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--seed-sets", "1", "3",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["seed_sets"], [1, 3])


class TestCLIValidation(unittest.TestCase):
    """Test CLI argument validation."""

    def test_default_hyperparameters_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["model_name"], "vinai/bertweet-base")
        self.assertEqual(kwargs["lr"], 2e-5)
        self.assertEqual(kwargs["batch_size"], 32)
        self.assertEqual(kwargs["pseudo_label_source"], "gpt-4o")

    def test_phase1_seed_strategy_default(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["phase1_seed_strategy"], "last")

    def test_phase1_seed_strategy_best_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "humanitarian", "--modality", "text_only",
            "--phase1-seed-strategy", "best",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["phase1_seed_strategy"], "best")

    def test_different_task_and_modality(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment",
            "--task", "informative", "--modality", "image_only",
            "--budget", "10", "--seed-set", "2",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result(10, 2)]
                main()

        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], "informative")
        self.assertEqual(args[1], "image_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
