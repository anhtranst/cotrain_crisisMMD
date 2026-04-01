"""Tests for scripts/dashboard.py — CrisisMMD HTML dashboard generator.

Pure-Python tests: no torch/numpy/transformers required.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.dashboard import (
    collect_all_metrics,
    collect_dataset_stats,
    collect_event_stats,
    collect_zeroshot_results,
    generate_html,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(path, fieldnames, rows):
    """Write a TSV file with given fieldnames and row dicts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _make_dataset_tree(tmpdir, task="informative", modality="text_only"):
    """Create a minimal preprocessed CrisisMMD task tree under tmpdir/CrisisMMD/tasks/."""
    base = Path(tmpdir) / "CrisisMMD" / "tasks" / task / modality
    fieldnames = ["tweet_id", "tweet_text", "class_label"]
    _write_tsv(base / "train.tsv", fieldnames, [
        {"tweet_id": "1", "tweet_text": "flood", "class_label": "informative"},
        {"tweet_id": "2", "tweet_text": "hello", "class_label": "not_informative"},
        {"tweet_id": "3", "tweet_text": "help", "class_label": "informative"},
    ])
    _write_tsv(base / "dev.tsv", fieldnames, [
        {"tweet_id": "4", "tweet_text": "rain", "class_label": "informative"},
    ])
    _write_tsv(base / "test.tsv", fieldnames, [
        {"tweet_id": "5", "tweet_text": "sun", "class_label": "not_informative"},
    ])
    return base


def _make_zeroshot_metrics(tmpdir, model="llama-3.2-11b", task="informative",
                           modality="text_only", split="test", **overrides):
    """Write a metrics.json under results/zeroshot/{model}/{task}/{modality}/{split}/."""
    metrics = {
        "task": task,
        "modality": modality,
        "split": split,
        "model_id": f"org/{model}",
        "num_samples": 100,
        "num_unparseable": 0,
        "accuracy": 0.85,
        "weighted_precision": 0.84,
        "weighted_recall": 0.85,
        "weighted_f1": 0.84,
        "macro_precision": 0.83,
        "macro_recall": 0.82,
        "macro_f1": 0.82,
        "per_class_f1": {"informative": 0.90, "not_informative": 0.74},
    }
    metrics.update(overrides)
    out_dir = Path(tmpdir) / "zeroshot" / model / task / modality / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    return metrics


def _make_cotrain_metrics(tmpdir, task="humanitarian", modality="text_only",
                          budget=5, seed_set=1, **overrides):
    """Write a co-training metrics.json under cotrain/lg-cotrain/...."""
    metrics = {
        "task": task,
        "modality": modality,
        "method": "lg-cotrain",
        "pseudo_label_source": "llama-3.2-11b",
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": 35.0,
        "test_macro_f1": 0.55,
        "test_ece": 0.12,
        "dev_macro_f1": 0.56,
    }
    metrics.update(overrides)
    out_dir = (Path(tmpdir) / "cotrain" / "lg-cotrain" / "llama-3.2-11b"
               / task / modality / f"{budget}_set{seed_set}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Tests: collect_dataset_stats
# ---------------------------------------------------------------------------

class TestCollectDatasetStats(unittest.TestCase):
    def test_returns_empty_if_no_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_dataset_stats(tmpdir)
        # All tasks present but empty
        for task in ["informative", "humanitarian"]:
            self.assertEqual(result[task], {})

    def test_collects_train_dev_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_dataset_tree(tmpdir, "informative", "text_only")
            result = collect_dataset_stats(tmpdir)
        stats = result["informative"]["text_only"]
        self.assertEqual(stats["train"]["informative"], 2)
        self.assertEqual(stats["train"]["not_informative"], 1)
        self.assertEqual(stats["dev"]["informative"], 1)
        self.assertEqual(stats["test"]["not_informative"], 1)

    def test_multiple_modalities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_dataset_tree(tmpdir, "informative", "text_only")
            _make_dataset_tree(tmpdir, "informative", "image_only")
            result = collect_dataset_stats(tmpdir)
        self.assertIn("text_only", result["informative"])
        self.assertIn("image_only", result["informative"])


# ---------------------------------------------------------------------------
# Tests: collect_event_stats
# ---------------------------------------------------------------------------

class TestCollectEventStats(unittest.TestCase):
    def test_returns_empty_if_no_original_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_event_stats(tmpdir)
        for task in result.values():
            self.assertEqual(task, {})

    def test_counts_events_from_original(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Path(tmpdir) / "CrisisMMD" / "original"
            orig.mkdir(parents=True)
            _write_tsv(orig / "task_informative_text_img_train.tsv",
                       ["event_name", "tweet_text"],
                       [{"event_name": "hurricane_harvey", "tweet_text": "a"},
                        {"event_name": "hurricane_harvey", "tweet_text": "b"},
                        {"event_name": "hurricane_irma", "tweet_text": "c"}])
            result = collect_event_stats(tmpdir)
        self.assertEqual(result["informative"]["train"]["hurricane_harvey"], 2)
        self.assertEqual(result["informative"]["train"]["hurricane_irma"], 1)


# ---------------------------------------------------------------------------
# Tests: collect_all_metrics
# ---------------------------------------------------------------------------

class TestCollectAllMetrics(unittest.TestCase):
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_all_metrics(tmpdir)
        self.assertEqual(result, [])

    def test_finds_cotrain_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_cotrain_metrics(tmpdir)
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["task"], "humanitarian")

    def test_skips_zeroshot_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zeroshot_metrics(tmpdir)
            result = collect_all_metrics(tmpdir)
        self.assertEqual(result, [])

    def test_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_cotrain_metrics(tmpdir, seed_set=1)
            bad_path = (Path(tmpdir) / "cotrain" / "lg-cotrain" / "llama-3.2-11b"
                        / "humanitarian" / "text_only" / "5_set2" / "metrics.json")
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text("{invalid json")
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)

    def test_multiple_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for s in [1, 2, 3]:
                _make_cotrain_metrics(tmpdir, seed_set=s)
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# Tests: collect_zeroshot_results
# ---------------------------------------------------------------------------

class TestCollectZeroshotResults(unittest.TestCase):
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_zeroshot_results(tmpdir)
        self.assertEqual(result, {})

    def test_finds_zeroshot_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zeroshot_metrics(tmpdir, model="llama-3.2-11b",
                                   task="informative", modality="text_only", split="test")
            result = collect_zeroshot_results(tmpdir)
        self.assertIn("informative", result)
        self.assertEqual(len(result["informative"]), 1)
        self.assertEqual(result["informative"][0]["model_slug"], "llama-3.2-11b")

    def test_multiple_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zeroshot_metrics(tmpdir, model="llama-3.2-11b",
                                   task="informative", split="test")
            _make_zeroshot_metrics(tmpdir, model="qwen2.5-vl-7b",
                                   task="informative", split="test", accuracy=0.70)
            result = collect_zeroshot_results(tmpdir)
        self.assertEqual(len(result["informative"]), 2)
        slugs = {m["model_slug"] for m in result["informative"]}
        self.assertEqual(slugs, {"llama-3.2-11b", "qwen2.5-vl-7b"})

    def test_multiple_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_zeroshot_metrics(tmpdir, task="informative", split="test")
            _make_zeroshot_metrics(tmpdir, task="humanitarian", split="test",
                                   macro_f1=0.63)
            result = collect_zeroshot_results(tmpdir)
        self.assertIn("informative", result)
        self.assertIn("humanitarian", result)


# ---------------------------------------------------------------------------
# Tests: generate_html
# ---------------------------------------------------------------------------

class TestGenerateHtml(unittest.TestCase):
    def test_returns_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html = generate_html(tmpdir, tmpdir)
        self.assertTrue(html.strip().startswith("<!DOCTYPE html>"))

    def test_contains_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html = generate_html(tmpdir, tmpdir)
        self.assertIn("LG-CoTrain", html)
        self.assertIn("CrisisMMD", html)

    def test_contains_dataset_tab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html = generate_html(tmpdir, tmpdir)
        self.assertIn("Dataset Exploration", html)

    def test_contains_cotrain_tab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            html = generate_html(tmpdir, tmpdir)
        self.assertIn("Co-Training Results", html)

    def test_zeroshot_single_tab_with_subtabs(self):
        """Zero-shot results create one top-level tab with sub-tabs per task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            _make_zeroshot_metrics(str(results_dir), task="informative")
            _make_zeroshot_metrics(str(results_dir), task="humanitarian", macro_f1=0.63)
            html = generate_html(tmpdir, str(results_dir))
        # Single top-level Zero-Shot tab
        self.assertIn("tab-zeroshot", html)
        self.assertIn("Zero-Shot", html)
        # Sub-tabs for each task
        self.assertIn("sub-zs-informative", html)
        self.assertIn("sub-zs-humanitarian", html)
        self.assertIn("sub-tab-bar", html)

    def test_per_model_summary_cards(self):
        """Per-model summary cards show test-set averages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            _make_zeroshot_metrics(str(results_dir), model="llama-3.2-11b",
                                   task="informative", split="test",
                                   avg_confidence=0.85, avg_entropy=0.30)
            _make_zeroshot_metrics(str(results_dir), model="qwen2.5-vl-7b",
                                   task="informative", split="test", accuracy=0.70,
                                   avg_confidence=0.75, avg_entropy=0.40)
            html = generate_html(tmpdir, str(results_dir))
        self.assertIn("llama-3.2-11b", html)
        self.assertIn("qwen2.5-vl-7b", html)
        self.assertIn("Test Set Averages", html)
        self.assertIn("Avg Confidence", html)

    def test_modality_section_headers(self):
        """Results tables are grouped by modality with section headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            _make_zeroshot_metrics(str(results_dir), task="informative",
                                   modality="text_only", split="test")
            _make_zeroshot_metrics(str(results_dir), task="informative",
                                   modality="image_only", split="test")
            html = generate_html(tmpdir, str(results_dir))
        self.assertIn("Text Only", html)
        self.assertIn("Image Only", html)

    def test_sortable_table_headers(self):
        """Table headers have sortable class and onclick."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            _make_zeroshot_metrics(str(results_dir), task="informative", split="test")
            html = generate_html(tmpdir, str(results_dir))
        self.assertIn("sortable", html)
        self.assertIn("sortTable", html)

    def test_dataset_stats_rendered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_dataset_tree(tmpdir, "informative", "text_only")
            html = generate_html(tmpdir, tmpdir)
        self.assertIn("informative", html.lower())


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------

class TestDashboardCLI(unittest.TestCase):
    def test_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["dashboard",
                                     "--results-root", tmpdir,
                                     "--data-root", tmpdir]):
                main()
            self.assertTrue((Path(tmpdir) / "dashboard.html").exists())

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "custom.html")
            with patch("sys.argv", ["dashboard",
                                     "--results-root", tmpdir,
                                     "--data-root", tmpdir,
                                     "--output", output]):
                main()
            self.assertTrue(Path(output).exists())

    def test_writes_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["dashboard",
                                     "--results-root", tmpdir,
                                     "--data-root", tmpdir]):
                main()
            content = (Path(tmpdir) / "dashboard.html").read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("LG-CoTrain", content)


if __name__ == "__main__":
    unittest.main()
