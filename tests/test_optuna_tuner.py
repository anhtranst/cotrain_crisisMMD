"""Tests for optuna_tuner.py — global Optuna hyperparameter tuner.

Pure-Python tests: no torch/numpy/transformers required.
Uses mock trainer classes injected via _trainer_cls.
"""

import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace")

import optuna

from lg_cotrain.config import LGCoTrainConfig
from lg_cotrain.optuna_tuner import create_objective, run_study


def _make_result(dev_macro_f1=0.55):
    """Helper to build a result dict matching trainer.run() output."""
    return {
        "task": "humanitarian",
        "modality": "text_only",
        "budget": 50,
        "seed_set": 1,
        "test_error_rate": 40.0,
        "test_macro_f1": 0.50,
        "test_per_class_f1": [0.5] * 8,
        "dev_error_rate": 39.0,
        "dev_macro_f1": dev_macro_f1,
        "lambda1_mean": 0.7,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.5,
        "lambda2_std": 0.1,
    }


def _fake_trainer_cls(config):
    """Return a mock trainer whose .run() produces a valid result dict."""
    mock = MagicMock()
    mock.run.return_value = _make_result()
    return mock


class TestCreateObjective(unittest.TestCase):
    def test_calls_trainer(self):
        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            mock = MagicMock()
            mock.run.return_value = _make_result(dev_macro_f1=0.60)
            return mock

        objective = create_objective(
            task="humanitarian",
            modality="text_only",
            _trainer_cls=capturing_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        self.assertEqual(len(configs_seen), 1)
        self.assertEqual(configs_seen[0].task, "humanitarian")
        self.assertEqual(configs_seen[0].modality, "text_only")

    def test_returns_dev_f1(self):
        def fixed_cls(config):
            mock = MagicMock()
            mock.run.return_value = _make_result(dev_macro_f1=0.75)
            return mock

        objective = create_objective(
            _trainer_cls=fixed_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        self.assertAlmostEqual(study.best_value, 0.75, places=6)

    def test_passes_sampled_hyperparams_to_config(self):
        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            mock = MagicMock()
            mock.run.return_value = _make_result()
            return mock

        objective = create_objective(
            _trainer_cls=capturing_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        cfg = configs_seen[0]
        trial = study.best_trial

        self.assertEqual(cfg.lr, trial.params["lr"])
        self.assertEqual(cfg.batch_size, trial.params["batch_size"])
        self.assertEqual(cfg.cotrain_epochs, trial.params["cotrain_epochs"])
        self.assertEqual(cfg.finetune_patience, trial.params["finetune_patience"])

    def test_hyperparams_within_search_space(self):
        objective = create_objective(
            _trainer_cls=_fake_trainer_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=5)

        for trial in study.trials:
            self.assertGreaterEqual(trial.params["lr"], 1e-5)
            self.assertLessEqual(trial.params["lr"], 1e-3)
            self.assertIn(trial.params["batch_size"], [8, 16, 32, 64])
            self.assertGreaterEqual(trial.params["cotrain_epochs"], 5)
            self.assertLessEqual(trial.params["cotrain_epochs"], 20)
            self.assertGreaterEqual(trial.params["finetune_patience"], 4)
            self.assertLessEqual(trial.params["finetune_patience"], 10)

    def test_budget_and_seed_forwarded(self):
        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            mock = MagicMock()
            mock.run.return_value = _make_result()
            return mock

        objective = create_objective(
            budget=25,
            seed_set=3,
            _trainer_cls=capturing_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        self.assertEqual(configs_seen[0].budget, 25)
        self.assertEqual(configs_seen[0].seed_set, 3)

    def test_fixed_params_forwarded(self):
        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            mock = MagicMock()
            mock.run.return_value = _make_result()
            return mock

        objective = create_objective(
            fixed_params={"stopping_strategy": "no_early_stopping"},
            _trainer_cls=capturing_cls,
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        self.assertEqual(configs_seen[0].stopping_strategy, "no_early_stopping")

    def test_results_root_forwarded(self):
        configs_seen = []

        def capturing_cls(config):
            configs_seen.append(config)
            mock = MagicMock()
            mock.run.return_value = _make_result()
            return mock

        with tempfile.TemporaryDirectory() as tmpdir:
            objective = create_objective(
                results_root=tmpdir,
                _trainer_cls=capturing_cls,
            )

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=1)

            self.assertEqual(configs_seen[0].results_root, tmpdir)


class TestRunStudy(unittest.TestCase):
    def test_completes_with_correct_trial_count(self):
        study = run_study(
            n_trials=3,
            _trainer_cls=_fake_trainer_cls,
        )

        self.assertEqual(len(study.trials), 3)

    def test_best_params_has_all_keys(self):
        study = run_study(
            n_trials=2,
            _trainer_cls=_fake_trainer_cls,
        )

        expected_keys = {"lr", "batch_size", "cotrain_epochs", "finetune_patience"}
        self.assertEqual(set(study.best_params.keys()), expected_keys)

    def test_best_value_is_valid(self):
        study = run_study(
            n_trials=2,
            _trainer_cls=_fake_trainer_cls,
        )

        self.assertIsInstance(study.best_value, float)
        self.assertGreater(study.best_value, 0.0)
        self.assertLessEqual(study.best_value, 1.0)

    def test_direction_is_maximize(self):
        study = run_study(
            n_trials=1,
            _trainer_cls=_fake_trainer_cls,
        )

        self.assertEqual(study.direction, optuna.study.StudyDirection.MAXIMIZE)

    def test_study_name(self):
        study = run_study(
            n_trials=1,
            study_name="test_study",
            _trainer_cls=_fake_trainer_cls,
        )

        self.assertEqual(study.study_name, "test_study")

    def test_storage_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"sqlite:///{tmpdir}/test.db"

            study1 = run_study(
                n_trials=2,
                storage=db_path,
                study_name="persist_test",
                _trainer_cls=_fake_trainer_cls,
            )
            self.assertEqual(len(study1.trials), 2)

            study2 = run_study(
                n_trials=2,
                storage=db_path,
                study_name="persist_test",
                _trainer_cls=_fake_trainer_cls,
            )
            self.assertEqual(len(study2.trials), 4)


class TestCLI(unittest.TestCase):
    def test_help_flag(self):
        from lg_cotrain.optuna_tuner import main

        with patch("sys.argv", ["optuna_tuner", "--help"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 0)

    def test_default_args(self):
        from lg_cotrain.optuna_tuner import main

        with patch("sys.argv", ["optuna_tuner"]):
            with patch("lg_cotrain.optuna_tuner.run_study") as mock_run:
                mock_run.return_value = MagicMock()
                main()

                _, kwargs = mock_run.call_args
                self.assertEqual(kwargs["n_trials"], 20)
                self.assertEqual(kwargs["task"], "humanitarian")
                self.assertEqual(kwargs["modality"], "text_only")
                self.assertEqual(kwargs["budget"], 50)
                self.assertEqual(kwargs["seed_set"], 1)
                self.assertEqual(kwargs["study_name"], "lg_cotrain_global")
                self.assertIsNone(kwargs["storage"])

    def test_custom_args(self):
        from lg_cotrain.optuna_tuner import main

        with patch(
            "sys.argv",
            [
                "optuna_tuner",
                "--n-trials", "10",
                "--task", "informative",
                "--modality", "image_only",
                "--budget", "25",
                "--seed-set", "2",
                "--study-name", "my_study",
                "--storage", "sqlite:///test.db",
            ],
        ):
            with patch("lg_cotrain.optuna_tuner.run_study") as mock_run:
                mock_run.return_value = MagicMock()
                main()

                _, kwargs = mock_run.call_args
                self.assertEqual(kwargs["n_trials"], 10)
                self.assertEqual(kwargs["task"], "informative")
                self.assertEqual(kwargs["modality"], "image_only")
                self.assertEqual(kwargs["budget"], 25)
                self.assertEqual(kwargs["seed_set"], 2)
                self.assertEqual(kwargs["study_name"], "my_study")
                self.assertEqual(kwargs["storage"], "sqlite:///test.db")


if __name__ == "__main__":
    unittest.main(verbosity=2)
