"""Tests for vanilla_cotrain/config.py — pure stdlib, no external deps."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vanilla_cotrain.config import VanillaCoTrainConfig


class TestVanillaConfigDefaults(unittest.TestCase):
    """Default hyperparameters."""

    def test_default_task(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.task, "humanitarian")

    def test_default_modality(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.modality, "text_only")

    def test_default_method(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.method, "vanilla-cotrain")

    def test_default_model(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.model_name, "vinai/bertweet-base")

    def test_default_image_model(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.image_model_name, "openai/clip-vit-base-patch32")

    def test_default_num_iterations(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.num_iterations, 10)

    def test_default_samples_per_class(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.samples_per_class, 5)

    def test_default_train_epochs(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.train_epochs, 5)

    def test_default_finetune_max_epochs(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.finetune_max_epochs, 50)

    def test_default_finetune_patience(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.finetune_patience, 5)

    def test_default_batch_size(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.batch_size, 32)

    def test_default_lr(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.lr, 2e-5)

    def test_default_weight_decay(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.weight_decay, 0.01)

    def test_default_warmup_ratio(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.warmup_ratio, 0.1)


class TestVanillaConfigPathComputation(unittest.TestCase):
    """Auto-computed paths from task/modality/budget/seed_set."""

    def test_labeled_path(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        self.assertIn("CrisisMMD", cfg.labeled_path)
        self.assertIn("humanitarian", cfg.labeled_path)
        self.assertIn("text_only", cfg.labeled_path)
        self.assertTrue(cfg.labeled_path.endswith("labeled_5_set1.tsv"))

    def test_unlabeled_path(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_5_set1.tsv"))

    def test_no_pseudo_label_path(self):
        """Vanilla co-training has no pseudo_label_path field."""
        cfg = VanillaCoTrainConfig()
        self.assertFalse(hasattr(cfg, "pseudo_label_path"))

    def test_dev_path(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        self.assertTrue(cfg.dev_path.endswith("dev.tsv"))

    def test_test_path(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        self.assertTrue(cfg.test_path.endswith("test.tsv"))

    def test_output_dir_no_pseudo_source(self):
        """Output path should NOT include pseudo_label_source."""
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        expected_suffix = str(
            Path("cotrain/vanilla-cotrain/humanitarian/text_only/5_set1")
        )
        self.assertTrue(cfg.output_dir.endswith(expected_suffix))
        # Explicitly check no llama or pseudo source in the path
        self.assertNotIn("llama", cfg.output_dir)
        self.assertNotIn("qwen", cfg.output_dir)

    def test_output_dir_with_run_id(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1,
            run_id="run-1",
        )
        expected_suffix = str(
            Path("cotrain/vanilla-cotrain/run-1/humanitarian/text_only/5_set1")
        )
        self.assertTrue(cfg.output_dir.endswith(expected_suffix))


class TestVanillaConfigVariousTasks(unittest.TestCase):
    """Paths change correctly for different task/modality/budget/seed."""

    def test_informative_image_only(self):
        cfg = VanillaCoTrainConfig(
            task="informative", modality="image_only", budget=25, seed_set=3
        )
        self.assertIn("informative", cfg.labeled_path)
        self.assertIn("image_only", cfg.labeled_path)
        self.assertTrue(cfg.labeled_path.endswith("labeled_25_set3.tsv"))
        expected_suffix = str(
            Path("cotrain/vanilla-cotrain/informative/image_only/25_set3")
        )
        self.assertTrue(cfg.output_dir.endswith(expected_suffix))

    def test_text_image_modality(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_image", budget=10, seed_set=2
        )
        self.assertIn("text_image", cfg.labeled_path)
        self.assertTrue(cfg.labeled_path.endswith("labeled_10_set2.tsv"))

    def test_budget_50_seed_2(self):
        cfg = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=50, seed_set=2
        )
        self.assertTrue(cfg.labeled_path.endswith("labeled_50_set2.tsv"))
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_50_set2.tsv"))


class TestVanillaConfigDevice(unittest.TestCase):
    """Device override field."""

    def test_default_device_is_none(self):
        cfg = VanillaCoTrainConfig()
        self.assertIsNone(cfg.device)

    def test_custom_device(self):
        cfg = VanillaCoTrainConfig(device="cuda:1")
        self.assertEqual(cfg.device, "cuda:1")

    def test_device_does_not_affect_paths(self):
        cfg1 = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1
        )
        cfg2 = VanillaCoTrainConfig(
            task="humanitarian", modality="text_only", budget=5, seed_set=1,
            device="cuda:1",
        )
        self.assertEqual(cfg1.labeled_path, cfg2.labeled_path)
        self.assertEqual(cfg1.output_dir, cfg2.output_dir)


class TestVanillaConfigCustomRoots(unittest.TestCase):
    """Custom data_root and results_root."""

    def test_custom_data_root(self):
        cfg = VanillaCoTrainConfig(data_root="/custom/data")
        self.assertIn("custom", cfg.labeled_path)

    def test_custom_results_root(self):
        cfg = VanillaCoTrainConfig(results_root="/custom/results")
        self.assertIn("custom", cfg.output_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
