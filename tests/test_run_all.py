"""Tests for run_all.py — batch experiment runner.

Pure-Python tests: no torch/numpy/transformers required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace")

from lg_cotrain.run_all import BUDGETS, SEED_SETS, format_summary_table


def _make_result(budget, seed_set, error_rate=40.0, macro_f1=0.50):
    """Helper to build a result dict matching trainer.run() output."""
    return {
        "task": "humanitarian",
        "modality": "text_only",
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": error_rate,
        "test_macro_f1": macro_f1,
        "test_per_class_f1": [0.5] * 8,
        "dev_error_rate": error_rate - 1,
        "dev_macro_f1": macro_f1 + 0.01,
        "lambda1_mean": 0.7,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.5,
        "lambda2_std": 0.1,
    }


def _fake_trainer_cls(config):
    """Return a mock trainer whose .run() produces a valid result dict."""
    mock = MagicMock()
    mock.run.return_value = _make_result(config.budget, config.seed_set)
    return mock


class TestConstants(unittest.TestCase):
    def test_budgets(self):
        self.assertEqual(BUDGETS, [5, 10, 25, 50])

    def test_seed_sets(self):
        self.assertEqual(SEED_SETS, [1, 2, 3])

    def test_total_combinations(self):
        self.assertEqual(len(BUDGETS) * len(SEED_SETS), 12)


class TestFormatSummaryTable(unittest.TestCase):
    def test_complete_results(self):
        results = []
        for b in BUDGETS:
            for s in SEED_SETS:
                results.append(_make_result(b, s, error_rate=40.0 - b * 0.1 + s))
        table = format_summary_table(results, "humanitarian", "text_only")
        self.assertIn("humanitarian/text_only", table)
        for b in BUDGETS:
            self.assertIn(str(b), table)

    def test_partial_results_with_none(self):
        results = [
            _make_result(5, 1, error_rate=45.0, macro_f1=0.40),
            _make_result(5, 2, error_rate=43.0, macro_f1=0.42),
            None,  # budget=5, seed=3 failed
        ]
        results.extend([None] * 9)
        table = format_summary_table(results, "humanitarian", "text_only")
        self.assertIn("N/A", table)
        self.assertIn("0.41", table)  # mean of 0.40 and 0.42

    def test_single_seed_no_std(self):
        results = [_make_result(5, 1, error_rate=45.0, macro_f1=0.40)]
        results.extend([None] * 11)
        table = format_summary_table(results, "humanitarian", "text_only")
        self.assertIn("45.00", table)
        self.assertIn("0.4000", table)
        self.assertNotIn("+/-", table)

    def test_all_none_results(self):
        results = [None] * 12
        table = format_summary_table(results, "humanitarian", "text_only")
        self.assertIn("N/A", table)


class TestRunAllExperiments(unittest.TestCase):
    """Tests use _trainer_cls injection to avoid torch/numpy imports."""

    def test_iterates_all_12_combinations(self):
        from lg_cotrain.run_all import run_all_experiments

        seen = []

        def tracking_cls(config):
            seen.append((config.budget, config.seed_set))
            return _fake_trainer_cls(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=tracking_cls,
            )

        self.assertEqual(len(results), 12)
        expected = [(b, s) for b in BUDGETS for s in SEED_SETS]
        self.assertEqual(seen, expected)

    def test_skips_existing_metrics(self):
        from lg_cotrain.run_all import run_all_experiments

        call_count = [0]

        def counting_cls(config):
            call_count[0] += 1
            return _fake_trainer_cls(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create metrics.json for budget=5, seed=1
            out_dir = Path(tmpdir) / "cotrain" / "lg-cotrain" / "llama-3.2-11b" / "humanitarian" / "text_only" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = _make_result(5, 1, error_rate=99.0, macro_f1=0.99)
            (out_dir / "metrics.json").write_text(json.dumps(existing))

            results = run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=counting_cls,
            )

        self.assertEqual(len(results), 12)
        r_5_1 = [r for r in results if r["budget"] == 5 and r["seed_set"] == 1][0]
        self.assertEqual(r_5_1["test_error_rate"], 99.0)
        self.assertEqual(r_5_1["test_macro_f1"], 0.99)
        self.assertEqual(call_count[0], 11)

    def test_continues_on_failure(self):
        from lg_cotrain.run_all import run_all_experiments

        call_count = [0]

        def failing_cls(config):
            call_count[0] += 1
            mock = MagicMock()
            if config.budget == 10 and config.seed_set == 2:
                mock.run.side_effect = RuntimeError("OOM")
            else:
                mock.run.return_value = _make_result(config.budget, config.seed_set)
            return mock

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=failing_cls,
            )

        self.assertEqual(len(results), 12)
        self.assertIsNone(results[4])
        non_none = [r for r in results if r is not None]
        self.assertEqual(len(non_none), 11)
        self.assertEqual(call_count[0], 12)

    def test_hyperparameters_forwarded(self):
        from lg_cotrain.run_all import run_all_experiments

        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            return _fake_trainer_cls(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_all_experiments(
                "humanitarian", "text_only",
                lr=1e-4,
                batch_size=16,
                cotrain_epochs=5,
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=capturing_cls,
            )

        for cfg in configs_seen:
            self.assertEqual(cfg.lr, 1e-4)
            self.assertEqual(cfg.batch_size, 16)
            self.assertEqual(cfg.cotrain_epochs, 5)


class TestRunAllCLI(unittest.TestCase):
    def test_task_and_modality_required(self):
        from lg_cotrain.run_all import main

        with patch("sys.argv", ["run_all"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_default_hyperparameters(self):
        from lg_cotrain.run_all import main

        with patch("sys.argv", ["run_all", "--task", "humanitarian", "--modality", "text_only"]):
            with patch("lg_cotrain.run_all.run_all_experiments") as mock_run:
                mock_run.return_value = [None] * 12
                with patch("lg_cotrain.run_all.format_summary_table", return_value=""):
                    main()

                _, kwargs = mock_run.call_args
                self.assertEqual(kwargs["model_name"], "vinai/bertweet-base")
                self.assertEqual(kwargs["lr"], 2e-5)
                self.assertEqual(kwargs["batch_size"], 32)

    def test_custom_hyperparameters(self):
        from lg_cotrain.run_all import main

        with patch(
            "sys.argv",
            [
                "run_all",
                "--task", "humanitarian",
                "--modality", "text_only",
                "--lr", "1e-4",
                "--batch-size", "16",
                "--cotrain-epochs", "5",
            ],
        ):
            with patch("lg_cotrain.run_all.run_all_experiments") as mock_run:
                mock_run.return_value = [None] * 12
                with patch("lg_cotrain.run_all.format_summary_table", return_value=""):
                    main()

                _, kwargs = mock_run.call_args
                self.assertEqual(kwargs["lr"], 1e-4)
                self.assertEqual(kwargs["batch_size"], 16)
                self.assertEqual(kwargs["cotrain_epochs"], 5)


class TestCustomBudgetsAndSeedSets(unittest.TestCase):
    def test_custom_budgets_subset(self):
        from lg_cotrain.run_all import run_all_experiments

        seen = []

        def tracking_cls(config):
            seen.append((config.budget, config.seed_set))
            return _fake_trainer_cls(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "humanitarian", "text_only",
                budgets=[5, 10],
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=tracking_cls,
            )

        self.assertEqual(len(results), 6)
        budgets_seen = set(b for b, s in seen)
        self.assertEqual(budgets_seen, {5, 10})

    def test_single_budget_single_seed(self):
        from lg_cotrain.run_all import run_all_experiments

        seen = []

        def tracking_cls(config):
            seen.append((config.budget, config.seed_set))
            return _fake_trainer_cls(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "humanitarian", "text_only",
                budgets=[25],
                seed_sets=[2],
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=tracking_cls,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(seen, [(25, 2)])


class TestOnExperimentDoneCallback(unittest.TestCase):
    def test_callback_called_for_all_12(self):
        from lg_cotrain.run_all import run_all_experiments

        calls = []

        def recorder(task, modality, budget, seed_set, status):
            calls.append((task, modality, budget, seed_set, status))

        with tempfile.TemporaryDirectory() as tmpdir:
            run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=_fake_trainer_cls,
                _on_experiment_done=recorder,
            )

        self.assertEqual(len(calls), 12)
        for task, modality, budget, seed_set, status in calls:
            self.assertEqual(task, "humanitarian")
            self.assertEqual(modality, "text_only")
            self.assertEqual(status, "done")

    def test_callback_reports_skipped(self):
        from lg_cotrain.run_all import run_all_experiments

        calls = []

        def recorder(task, modality, budget, seed_set, status):
            calls.append((task, modality, budget, seed_set, status))

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "cotrain" / "lg-cotrain" / "llama-3.2-11b" / "humanitarian" / "text_only" / "5_set1"
            out_dir.mkdir(parents=True)
            existing = _make_result(5, 1)
            (out_dir / "metrics.json").write_text(json.dumps(existing))

            run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=_fake_trainer_cls,
                _on_experiment_done=recorder,
            )

        self.assertEqual(len(calls), 12)
        skipped = [c for c in calls if c[4] == "skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0], ("humanitarian", "text_only", 5, 1, "skipped"))

    def test_no_callback_is_fine(self):
        from lg_cotrain.run_all import run_all_experiments

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_all_experiments(
                "humanitarian", "text_only",
                data_root=tmpdir,
                results_root=tmpdir,
                _trainer_cls=_fake_trainer_cls,
            )

        self.assertEqual(len(results), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
