"""Tests for vanilla_cotrain/trainer.py.

Includes unit tests for sample selection and view dispatching,
plus a tiny end-to-end integration test.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from lg_cotrain.data_loading import CLASS_LABELS


class TestSelectTopKPerClass(unittest.TestCase):
    """Unit tests for VanillaCoTrainer._select_top_k_per_class."""

    def _make_trainer(self, num_labels=3, samples_per_class=2):
        """Create a minimal trainer with mocked config for testing selection."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        from vanilla_cotrain.config import VanillaCoTrainConfig
        from vanilla_cotrain.trainer import VanillaCoTrainer

        config = VanillaCoTrainConfig(
            num_labels=num_labels,
            samples_per_class=samples_per_class,
            device="cpu",
        )
        # We can't fully init the trainer without HuggingFace models,
        # so we create it and set minimal attributes directly
        trainer = object.__new__(VanillaCoTrainer)
        trainer.config = config
        trainer.device = "cpu"
        trainer.label2id = {f"class_{i}": i for i in range(num_labels)}
        trainer.id2label = {i: f"class_{i}" for i in range(num_labels)}
        return trainer

    def test_basic_selection(self):
        """Select top-2 per class from 3 classes."""
        trainer = self._make_trainer(num_labels=3, samples_per_class=2)

        # 9 samples: 3 per class
        pred_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Confidences: class 0=[0.9, 0.7, 0.5], class 1=[0.8, 0.6, 0.4], class 2=[0.95, 0.3, 0.2]
        probs = np.zeros((9, 3))
        confs = [0.9, 0.7, 0.5, 0.8, 0.6, 0.4, 0.95, 0.3, 0.2]
        for i, (c, lbl) in enumerate(zip(confs, pred_labels)):
            probs[i, lbl] = c

        selected = trainer._select_top_k_per_class(probs, pred_labels)

        # Should select top-2 from each class = 6 total
        self.assertEqual(len(selected), 6)
        # Check class 0: indices 0, 1 (highest confidence)
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        # Check class 1: indices 3, 4
        self.assertIn(3, selected)
        self.assertIn(4, selected)
        # Check class 2: index 6 (0.95) and one of 7 or 8
        self.assertIn(6, selected)

    def test_fewer_samples_than_k(self):
        """When a class has fewer samples than k, select all available."""
        trainer = self._make_trainer(num_labels=2, samples_per_class=5)

        pred_labels = np.array([0, 0, 1])  # 2 for class 0, 1 for class 1
        probs = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9]])

        selected = trainer._select_top_k_per_class(probs, pred_labels)
        # Class 0: 2 samples (< k=5), select all. Class 1: 1 sample, select all.
        self.assertEqual(len(selected), 3)

    def test_empty_class(self):
        """When a class has no predictions, it's skipped."""
        trainer = self._make_trainer(num_labels=3, samples_per_class=2)

        pred_labels = np.array([0, 0, 0])  # All class 0, none for 1 or 2
        probs = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])

        selected = trainer._select_top_k_per_class(probs, pred_labels)
        # Only class 0 has samples, top-2
        self.assertEqual(len(selected), 2)

    def test_empty_input(self):
        """Empty arrays return empty selection."""
        trainer = self._make_trainer(num_labels=2, samples_per_class=5)

        pred_labels = np.array([], dtype=int)
        probs = np.zeros((0, 2))

        selected = trainer._select_top_k_per_class(probs, pred_labels)
        self.assertEqual(len(selected), 0)


class TestViewDispatching(unittest.TestCase):
    """Test that _create_model_for_view creates correct model types."""

    def test_text_only_creates_bert_for_both_views(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("torch/transformers not available")

        from lg_cotrain.model import BertClassifier
        from vanilla_cotrain.config import VanillaCoTrainConfig
        from vanilla_cotrain.trainer import VanillaCoTrainer

        config = VanillaCoTrainConfig(
            modality="text_only",
            model_name="google/bert_uncased_L-2_H-128_A-2",
            num_labels=2,
            device="cpu",
        )
        trainer = VanillaCoTrainer(config)
        model_a = trainer._create_model_for_view("A")
        model_b = trainer._create_model_for_view("B")
        self.assertIsInstance(model_a, BertClassifier)
        self.assertIsInstance(model_b, BertClassifier)

    def test_image_only_creates_image_classifier_for_both_views(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("torch/transformers not available")

        from lg_cotrain.model import ImageClassifier
        from vanilla_cotrain.config import VanillaCoTrainConfig
        from vanilla_cotrain.trainer import VanillaCoTrainer

        config = VanillaCoTrainConfig(
            modality="image_only",
            num_labels=2,
            device="cpu",
        )
        trainer = VanillaCoTrainer(config)
        model_a = trainer._create_model_for_view("A")
        model_b = trainer._create_model_for_view("B")
        self.assertIsInstance(model_a, ImageClassifier)
        self.assertIsInstance(model_b, ImageClassifier)

    def test_text_image_creates_bert_and_image_classifier(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("torch/transformers not available")

        from lg_cotrain.model import BertClassifier, ImageClassifier
        from vanilla_cotrain.config import VanillaCoTrainConfig
        from vanilla_cotrain.trainer import VanillaCoTrainer

        config = VanillaCoTrainConfig(
            modality="text_image",
            model_name="google/bert_uncased_L-2_H-128_A-2",
            num_labels=2,
            device="cpu",
        )
        trainer = VanillaCoTrainer(config)
        model_a = trainer._create_model_for_view("A")
        model_b = trainer._create_model_for_view("B")
        self.assertIsInstance(model_a, BertClassifier)
        self.assertIsInstance(model_b, ImageClassifier)


class TestFullPipelineTiny(unittest.TestCase):
    """End-to-end pipeline on tiny synthetic data using bert-tiny.

    Requires: torch, transformers, pandas, sklearn.
    """

    def _make_tiny_data(self, tmp_path):
        """Create tiny synthetic data files for text_only modality."""
        classes = CLASS_LABELS
        labeled_rows = []
        for i, cls in enumerate(classes):
            for j in range(2):
                labeled_rows.append(
                    [str(100 + i * 10 + j), f"Labeled text {cls} {j}", cls]
                )

        unlabeled_rows = []
        for i, cls in enumerate(classes):
            for j in range(4):
                unlabeled_rows.append(
                    [str(500 + i * 10 + j), f"Unlabeled text {cls} {j}", cls]
                )

        dev_rows = [
            [str(900 + i), f"Dev text {cls}", cls] for i, cls in enumerate(classes)
        ]
        test_rows = [
            [str(950 + i), f"Test text {cls}", cls] for i, cls in enumerate(classes)
        ]

        paths = {}
        for name, rows in [
            ("labeled", labeled_rows),
            ("unlabeled", unlabeled_rows),
            ("dev", dev_rows),
            ("test", test_rows),
        ]:
            path = os.path.join(tmp_path, f"{name}.tsv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["tweet_id", "tweet_text", "class_label"])
                writer.writerows(rows)
            paths[name] = path
        return paths

    def test_full_pipeline(self):
        try:
            import torch
            import transformers
            import pandas
            import sklearn
        except ImportError:
            self.skipTest("torch/transformers/pandas/sklearn not available")

        from vanilla_cotrain.config import VanillaCoTrainConfig
        from vanilla_cotrain.trainer import VanillaCoTrainer

        import logging

        with tempfile.TemporaryDirectory() as tmp_path:
            paths = self._make_tiny_data(tmp_path)
            cfg = VanillaCoTrainConfig(
                task="humanitarian",
                modality="text_only",
                budget=5,
                seed_set=1,
                model_name="google/bert_uncased_L-2_H-128_A-2",
                num_labels=5,
                num_iterations=2,
                samples_per_class=1,
                train_epochs=1,
                finetune_max_epochs=2,
                finetune_patience=1,
                batch_size=8,
                max_seq_length=32,
                device="cpu",
            )
            cfg.labeled_path = paths["labeled"]
            cfg.unlabeled_path = paths["unlabeled"]
            cfg.dev_path = paths["dev"]
            cfg.test_path = paths["test"]
            cfg.output_dir = os.path.join(tmp_path, "results")
            cfg.log_dir = os.path.join(tmp_path, "logs")

            trainer = VanillaCoTrainer(cfg)
            result = trainer.run()

            # Check result dict has expected keys
            self.assertIn("test_macro_f1", result)
            self.assertIn("test_error_rate", result)
            self.assertIn("test_ece", result)
            self.assertIn("dev_macro_f1", result)
            self.assertIn("num_iterations_completed", result)
            self.assertIn("samples_added_to_A", result)
            self.assertIn("samples_added_to_B", result)
            self.assertIn("per_iteration_log", result)
            self.assertEqual(result["method"], "vanilla-cotrain")

            # Check metrics are in valid ranges
            self.assertGreaterEqual(result["test_macro_f1"], 0.0)
            self.assertLessEqual(result["test_macro_f1"], 1.0)
            self.assertGreaterEqual(result["test_error_rate"], 0.0)
            self.assertLessEqual(result["test_error_rate"], 100.0)
            self.assertGreaterEqual(result["num_iterations_completed"], 0)

            # Check metrics.json was saved
            metrics_path = os.path.join(tmp_path, "results", "metrics.json")
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["test_macro_f1"], result["test_macro_f1"])

            # Check prediction files were saved
            self.assertTrue(
                os.path.exists(os.path.join(tmp_path, "results", "test_predictions.tsv"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmp_path, "results", "dev_predictions.tsv"))
            )

            # Close logging file handlers so Windows can clean up temp dir
            for logger_name in ("vanilla_cotrain", "lg_cotrain"):
                lgr = logging.getLogger(logger_name)
                for handler in lgr.handlers[:]:
                    handler.close()
                    lgr.removeHandler(handler)


class TestRunAllVanilla(unittest.TestCase):
    """Test the batch runner with a mock trainer."""

    def test_run_all_with_mock_trainer(self):
        try:
            import pandas
        except ImportError:
            self.skipTest("pandas not available")

        from vanilla_cotrain.run_all import run_all_experiments

        class MockTrainer:
            def __init__(self, config):
                self.config = config

            def run(self):
                return {
                    "task": self.config.task,
                    "modality": self.config.modality,
                    "budget": self.config.budget,
                    "seed_set": self.config.seed_set,
                    "method": "vanilla-cotrain",
                    "test_error_rate": 30.0,
                    "test_macro_f1": 0.5,
                    "test_ece": 0.05,
                    "dev_macro_f1": 0.48,
                    "num_iterations_completed": 5,
                }

        with tempfile.TemporaryDirectory() as tmp_path:
            results = run_all_experiments(
                "humanitarian",
                "text_only",
                budgets=[5],
                seed_sets=[1],
                data_root=tmp_path,
                results_root=tmp_path,
                _trainer_cls=MockTrainer,
            )

            self.assertEqual(len(results), 1)
            self.assertIsNotNone(results[0])
            self.assertEqual(results[0]["test_macro_f1"], 0.5)
            self.assertEqual(results[0]["method"], "vanilla-cotrain")


class TestFormatSummaryTable(unittest.TestCase):
    """Test summary table formatting."""

    def test_single_result(self):
        from vanilla_cotrain.run_all import format_summary_table

        results = [{
            "task": "humanitarian",
            "modality": "text_only",
            "budget": 5,
            "seed_set": 1,
            "test_error_rate": 30.0,
            "test_macro_f1": 0.5,
        }]

        table = format_summary_table(
            results, "humanitarian", "text_only",
            budgets=[5], seed_sets=[1],
        )
        self.assertIn("Vanilla Co-Training", table)
        self.assertIn("humanitarian", table)
        self.assertIn("30.00", table)
        self.assertIn("0.5000", table)


if __name__ == "__main__":
    unittest.main(verbosity=2)
