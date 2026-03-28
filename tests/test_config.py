"""Tests for config.py — pure stdlib, no external deps."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lg_cotrain.config import LGCoTrainConfig


class TestConfigDefaults(unittest.TestCase):
    """Default hyperparameters."""

    def test_default_model(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.model_name, "vinai/bertweet-base")

    def test_default_num_labels(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.num_labels, 5)

    def test_default_task(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.task, "humanitarian")

    def test_default_modality(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.modality, "text_only")

    def test_default_batch_size(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.batch_size, 32)

    def test_default_lr(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.lr, 2e-5)

    def test_default_weight_decay(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.weight_decay, 0.01)

    def test_default_warmup_ratio(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.warmup_ratio, 0.1)

    def test_custom_weight_decay(self):
        cfg = LGCoTrainConfig(weight_decay=0.05)
        self.assertEqual(cfg.weight_decay, 0.05)

    def test_custom_warmup_ratio(self):
        cfg = LGCoTrainConfig(warmup_ratio=0.2)
        self.assertEqual(cfg.warmup_ratio, 0.2)

    def test_default_max_seq_length(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.max_seq_length, 128)

    def test_default_weight_gen_epochs(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.weight_gen_epochs, 7)

    def test_default_cotrain_epochs(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.cotrain_epochs, 10)

    def test_default_finetune_max_epochs(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.finetune_max_epochs, 100)

    def test_default_finetune_patience(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.finetune_patience, 5)


class TestConfigPathComputation(unittest.TestCase):
    """Auto-computed paths from task/modality/budget/seed_set."""

    def test_labeled_path(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertIn("CrisisMMD", cfg.labeled_path)
        self.assertIn("humanitarian", cfg.labeled_path)
        self.assertIn("text_only", cfg.labeled_path)
        self.assertTrue(cfg.labeled_path.endswith("labeled_5_set1.tsv"))

    def test_unlabeled_path(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_5_set1.tsv"))

    def test_pseudo_label_path(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertIn("pseudo-labelled", cfg.pseudo_label_path)
        self.assertIn("gpt-4o", cfg.pseudo_label_path)
        self.assertTrue(cfg.pseudo_label_path.endswith("train_pred.csv"))

    def test_dev_path(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertTrue(cfg.dev_path.endswith("dev.tsv"))

    def test_test_path(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertTrue(cfg.test_path.endswith("test.tsv"))

    def test_output_dir(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        self.assertTrue(cfg.output_dir.endswith(str(Path("humanitarian/text_only/5_set1"))))


class TestConfigVariousTasks(unittest.TestCase):
    """Paths change correctly for different task/modality/budget/seed."""

    def test_different_task_modality(self):
        cfg = LGCoTrainConfig(task="informative", modality="image_only", budget=25, seed_set=3)
        self.assertIn("informative", cfg.labeled_path)
        self.assertIn("image_only", cfg.labeled_path)
        self.assertTrue(cfg.labeled_path.endswith("labeled_25_set3.tsv"))
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_25_set3.tsv"))
        self.assertTrue(cfg.output_dir.endswith(str(Path("informative/image_only/25_set3"))))

    def test_budget_50_seed_2(self):
        cfg = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=50, seed_set=2)
        self.assertTrue(cfg.labeled_path.endswith("labeled_50_set2.tsv"))
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_50_set2.tsv"))
        self.assertTrue(cfg.output_dir.endswith(str(Path("humanitarian/text_only/50_set2"))))

    def test_pseudo_label_path_independent_of_budget(self):
        """Pseudo-label path depends only on task/modality, not budget/seed."""
        cfg1 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        cfg2 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=50, seed_set=3)
        self.assertEqual(cfg1.pseudo_label_path, cfg2.pseudo_label_path)


class TestConfigPseudoLabelSource(unittest.TestCase):
    """Tests for the configurable pseudo_label_source field."""

    def test_default_is_gpt4o(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.pseudo_label_source, "gpt-4o")

    def test_default_path_contains_gpt4o(self):
        cfg = LGCoTrainConfig()
        self.assertIn("gpt-4o", cfg.pseudo_label_path)

    def test_custom_source_changes_path(self):
        cfg = LGCoTrainConfig(pseudo_label_source="llama-3")
        self.assertIn("llama-3", cfg.pseudo_label_path)
        self.assertNotIn("gpt-4o", cfg.pseudo_label_path)

    def test_custom_source_preserves_task_and_filename(self):
        cfg = LGCoTrainConfig(
            task="humanitarian", modality="text_only", pseudo_label_source="llama-3"
        )
        self.assertIn("humanitarian", cfg.pseudo_label_path)
        self.assertTrue(cfg.pseudo_label_path.endswith("train_pred.csv"))

    def test_source_independent_of_budget_and_seed(self):
        cfg1 = LGCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1,
            pseudo_label_source="custom-model",
        )
        cfg2 = LGCoTrainConfig(
            task="humanitarian", modality="text_only", budget=50, seed_set=3,
            pseudo_label_source="custom-model",
        )
        self.assertEqual(cfg1.pseudo_label_path, cfg2.pseudo_label_path)


class TestConfigDevice(unittest.TestCase):
    """Device override field."""

    def test_default_device_is_none(self):
        cfg = LGCoTrainConfig()
        self.assertIsNone(cfg.device)

    def test_custom_device(self):
        cfg = LGCoTrainConfig(device="cuda:1")
        self.assertEqual(cfg.device, "cuda:1")

    def test_device_does_not_affect_paths(self):
        cfg1 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        cfg2 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1,
                                device="cuda:1")
        self.assertEqual(cfg1.labeled_path, cfg2.labeled_path)
        self.assertEqual(cfg1.output_dir, cfg2.output_dir)


class TestConfigPhase1SeedStrategy(unittest.TestCase):
    """phase1_seed_strategy config field."""

    def test_default_is_last(self):
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.phase1_seed_strategy, "last")

    def test_custom_best(self):
        cfg = LGCoTrainConfig(phase1_seed_strategy="best")
        self.assertEqual(cfg.phase1_seed_strategy, "best")

    def test_does_not_affect_paths(self):
        cfg1 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1)
        cfg2 = LGCoTrainConfig(task="humanitarian", modality="text_only", budget=5, seed_set=1,
                                phase1_seed_strategy="best")
        self.assertEqual(cfg1.labeled_path, cfg2.labeled_path)
        self.assertEqual(cfg1.output_dir, cfg2.output_dir)


class TestConfigCustomRoots(unittest.TestCase):
    """Custom data_root and results_root."""

    def test_custom_data_root(self):
        cfg = LGCoTrainConfig(data_root="/custom/data")
        self.assertIn("custom", cfg.labeled_path)

    def test_custom_results_root(self):
        cfg = LGCoTrainConfig(results_root="/custom/results")
        self.assertIn("custom", cfg.output_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
