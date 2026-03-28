"""Tests for dashboard.py — HTML results dashboard generator.

Pure-Python tests: no torch/numpy/transformers required.
"""

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/workspace")

from lg_cotrain.dashboard import (
    DEFAULT_EVENTS,
    EVENTS,
    _render_data_tab,
    _render_optuna_tab,
    build_lambda_pivot,
    build_overall_means,
    build_pivot_data,
    collect_all_metrics,
    collect_data_stats,
    compute_summary_cards,
    count_expected_experiments,
    discover_events,
    discover_result_sets,
    format_event_name,
    generate_html,
    generate_html_multi,
    get_event_class_count,
    load_optuna_results,
)
from lg_cotrain.run_all import BUDGETS, SEED_SETS


def _make_metric(event="california_wildfires_2018", budget=5, seed_set=1,
                 macro_f1=0.55, error_rate=35.0, num_classes=10,
                 test_ece=0.12, dev_ece=0.10):
    """Build a metrics dict matching trainer.run() output."""
    return {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": error_rate,
        "test_macro_f1": macro_f1,
        "test_ece": test_ece,
        "test_per_class_f1": [macro_f1] * num_classes,
        "dev_error_rate": error_rate - 1.0,
        "dev_macro_f1": macro_f1 + 0.01,
        "dev_ece": dev_ece,
        "lambda1_mean": 1.08,
        "lambda1_std": 0.21,
        "lambda2_mean": 0.56,
        "lambda2_std": 0.17,
    }


def _write_metric(tmpdir, metric):
    """Write a metric dict to the proper path under tmpdir."""
    path = (Path(tmpdir) / metric["event"]
            / f"{metric['budget']}_set{metric['seed_set']}" / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metric))


def _write_tsv(path, rows):
    """Write a minimal TSV with tweet_id, tweet_text, class_label columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["tweet_id\ttweet_text\tclass_label"]
    for i, row in enumerate(rows):
        lines.append(f"{i}\ttext_{i}\t{row['class_label']}")
    path.write_text("\n".join(lines))


class TestFormatEventName(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            format_event_name("california_wildfires_2018"),
            "California Wildfires 2018",
        )

    def test_multi_word(self):
        self.assertEqual(
            format_event_name("hurricane_harvey_2017"),
            "Hurricane Harvey 2017",
        )


class TestDiscoverEvents(unittest.TestCase):
    def test_discovers_from_metrics(self):
        metrics = [
            _make_metric(event="b_event"),
            _make_metric(event="a_event"),
            _make_metric(event="b_event", seed_set=2),
        ]
        events = discover_events(metrics)
        self.assertEqual(events, ["a_event", "b_event"])

    def test_empty_returns_empty(self):
        self.assertEqual(discover_events([]), [])


class TestDiscoverResultSets(unittest.TestCase):
    def test_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_result_sets(tmpdir)
        self.assertEqual(result, {})

    def test_three_level_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "gpt-4o" / "test" / "run-1"
            _write_metric(str(base), _make_metric())
            result = discover_result_sets(tmpdir)
        self.assertIn("gpt-4o", result)
        self.assertIn("test", result["gpt-4o"])
        names = [name for name, _ in result["gpt-4o"]["test"]]
        self.assertIn("run-1", names)

    def test_multiple_experiments_in_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for exp in ["run-1", "run-2"]:
                base = Path(tmpdir) / "gpt-4o" / "test" / exp
                _write_metric(str(base), _make_metric())
            result = discover_result_sets(tmpdir)
        names = [name for name, _ in result["gpt-4o"]["test"]]
        self.assertIn("run-1", names)
        self.assertIn("run-2", names)

    def test_multiple_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "quick-stop" / "baseline"), _make_metric())
            result = discover_result_sets(tmpdir)
        self.assertIn("test", result["gpt-4o"])
        self.assertIn("quick-stop", result["gpt-4o"])

    def test_empty_type_dir_included(self):
        """An experiment type dir with no experiments is included with empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            (Path(tmpdir) / "gpt-4o" / "stop").mkdir(parents=True)
            result = discover_result_sets(tmpdir)
        self.assertIn("stop", result["gpt-4o"])
        self.assertEqual(result["gpt-4o"]["stop"], [])

    def test_non_result_dirs_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            (Path(tmpdir) / "gpt-4o" / "test" / "empty-dir").mkdir(parents=True)
            result = discover_result_sets(tmpdir)
        names = [name for name, _ in result["gpt-4o"]["test"]]
        self.assertIn("run-1", names)
        self.assertNotIn("empty-dir", names)


class TestCollectAllMetrics(unittest.TestCase):
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_all_metrics(tmpdir)
        self.assertEqual(result, [])

    def test_single_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            m = _make_metric()
            _write_metric(tmpdir, m)
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event"], "california_wildfires_2018")

    def test_multiple_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for s in SEED_SETS:
                _write_metric(tmpdir, _make_metric(seed_set=s))
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 3)

    def test_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric(seed_set=1))
            bad_path = (Path(tmpdir) / "california_wildfires_2018"
                        / "5_set2" / "metrics.json")
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text("{invalid json")
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)

    def test_discovers_unknown_events(self):
        """Events NOT in DEFAULT_EVENTS are still found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric(event="custom_disaster_2025"))
            result = collect_all_metrics(tmpdir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["event"], "custom_disaster_2025")

    def test_returns_dicts_with_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric())
            result = collect_all_metrics(tmpdir)
        for key in ["event", "budget", "seed_set", "test_macro_f1"]:
            self.assertIn(key, result[0])


class TestCountExpectedExperiments(unittest.TestCase):
    def test_default_count_is_120(self):
        self.assertEqual(count_expected_experiments(), 120)

    def test_custom_events(self):
        self.assertEqual(count_expected_experiments(["a", "b"]), 2 * 4 * 3)


class TestGetEventClassCount(unittest.TestCase):
    def test_from_metrics(self):
        metrics = [_make_metric(num_classes=8)]
        counts = get_event_class_count(metrics)
        self.assertEqual(counts["california_wildfires_2018"], 8)

    def test_missing_event_defaults_to_10(self):
        counts = get_event_class_count([], events=DEFAULT_EVENTS)
        for event in DEFAULT_EVENTS:
            self.assertEqual(counts[event], 10)

    def test_mixed_class_counts(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", num_classes=10),
            _make_metric(event="canada_wildfires_2016", num_classes=8),
        ]
        counts = get_event_class_count(metrics)
        self.assertEqual(counts["california_wildfires_2018"], 10)
        self.assertEqual(counts["canada_wildfires_2016"], 8)


class TestBuildPivotData(unittest.TestCase):
    def test_empty_metrics(self):
        pivot = build_pivot_data([], events=DEFAULT_EVENTS)
        for event in DEFAULT_EVENTS:
            for budget in BUDGETS:
                self.assertIsNone(pivot[event][budget]["f1_mean"])
                self.assertEqual(pivot[event][budget]["count"], 0)

    def test_single_seed_no_std(self):
        metrics = [_make_metric(budget=5, seed_set=1, macro_f1=0.60)]
        pivot = build_pivot_data(metrics)
        entry = pivot["california_wildfires_2018"][5]
        self.assertAlmostEqual(entry["f1_mean"], 0.60)
        self.assertIsNone(entry["f1_std"])
        self.assertEqual(entry["count"], 1)

    def test_three_seeds_mean_std(self):
        f1_vals = [0.50, 0.55, 0.60]
        metrics = [
            _make_metric(budget=10, seed_set=s, macro_f1=f)
            for s, f in zip(SEED_SETS, f1_vals)
        ]
        pivot = build_pivot_data(metrics)
        entry = pivot["california_wildfires_2018"][10]
        self.assertAlmostEqual(entry["f1_mean"], statistics.mean(f1_vals))
        self.assertAlmostEqual(entry["f1_std"], statistics.stdev(f1_vals))
        self.assertEqual(entry["count"], 3)

    def test_multiple_events(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", budget=5, macro_f1=0.60),
            _make_metric(event="canada_wildfires_2016", budget=5, macro_f1=0.50),
        ]
        pivot = build_pivot_data(metrics)
        self.assertAlmostEqual(pivot["california_wildfires_2018"][5]["f1_mean"], 0.60)
        self.assertAlmostEqual(pivot["canada_wildfires_2016"][5]["f1_mean"], 0.50)

    def test_ece_aggregated(self):
        metrics = [_make_metric(budget=5, test_ece=0.15)]
        pivot = build_pivot_data(metrics)
        entry = pivot["california_wildfires_2018"][5]
        self.assertAlmostEqual(entry["ece_mean"], 0.15)

    def test_ece_missing_graceful(self):
        """Old metrics without test_ece should still work."""
        m = _make_metric()
        del m["test_ece"]
        pivot = build_pivot_data([m])
        entry = pivot["california_wildfires_2018"][5]
        self.assertIsNone(entry["ece_mean"])


class TestBuildOverallMeans(unittest.TestCase):
    def test_single_event(self):
        metrics = [_make_metric(budget=5, macro_f1=0.60, error_rate=30.0)]
        pivot = build_pivot_data(metrics)
        overall = build_overall_means(pivot)
        self.assertAlmostEqual(overall[5]["f1_mean"], 0.60)
        self.assertAlmostEqual(overall[5]["err_mean"], 30.0)

    def test_multiple_events_averaged(self):
        metrics = [
            _make_metric(event="california_wildfires_2018", budget=5, macro_f1=0.60, error_rate=30.0),
            _make_metric(event="canada_wildfires_2016", budget=5, macro_f1=0.40, error_rate=50.0),
        ]
        pivot = build_pivot_data(metrics)
        overall = build_overall_means(pivot)
        self.assertAlmostEqual(overall[5]["f1_mean"], 0.50)
        self.assertAlmostEqual(overall[5]["err_mean"], 40.0)

    def test_missing_budget_is_none(self):
        pivot = build_pivot_data([], events=DEFAULT_EVENTS)
        overall = build_overall_means(pivot)
        for budget in BUDGETS:
            self.assertIsNone(overall[budget]["f1_mean"])


class TestBuildLambdaPivot(unittest.TestCase):
    def test_lambda_values_extracted(self):
        metrics = [_make_metric()]
        pivot = build_lambda_pivot(metrics)
        entry = pivot["california_wildfires_2018"][5]
        self.assertAlmostEqual(entry["l1_mean"], 1.08)
        self.assertAlmostEqual(entry["l2_mean"], 0.56)

    def test_averaged_across_seeds(self):
        metrics = [
            _make_metric(seed_set=1),
            _make_metric(seed_set=2),
            _make_metric(seed_set=3),
        ]
        pivot = build_lambda_pivot(metrics)
        self.assertAlmostEqual(pivot["california_wildfires_2018"][5]["l1_mean"], 1.08)

    def test_missing_is_none(self):
        pivot = build_lambda_pivot([], events=DEFAULT_EVENTS)
        self.assertIsNone(pivot["california_wildfires_2018"][5]["l1_mean"])


class TestComputeSummaryCards(unittest.TestCase):
    def test_experiment_count(self):
        metrics = [_make_metric()]
        s = compute_summary_cards(metrics, events=DEFAULT_EVENTS)
        self.assertEqual(s["completed"], 1)
        self.assertEqual(s["total"], 120)

    def test_custom_events_total(self):
        metrics = [_make_metric(event="a")]
        s = compute_summary_cards(metrics, events=["a"])
        self.assertEqual(s["total"], 12)

    def test_percentage(self):
        metrics = [_make_metric() for _ in range(12)]
        s = compute_summary_cards(metrics, events=DEFAULT_EVENTS)
        self.assertAlmostEqual(s["pct"], 10.0)

    def test_avg_f1(self):
        metrics = [
            _make_metric(macro_f1=0.50),
            _make_metric(macro_f1=0.60, seed_set=2),
        ]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["avg_f1"], 0.55)

    def test_avg_error_rate(self):
        metrics = [
            _make_metric(error_rate=30.0),
            _make_metric(error_rate=40.0, seed_set=2),
        ]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["avg_err"], 35.0)

    def test_disaster_count(self):
        metrics = [
            _make_metric(event="california_wildfires_2018"),
            _make_metric(event="canada_wildfires_2016"),
        ]
        s = compute_summary_cards(metrics)
        self.assertEqual(s["disasters_done"], 2)

    def test_avg_ece(self):
        metrics = [
            _make_metric(test_ece=0.10),
            _make_metric(test_ece=0.20, seed_set=2),
        ]
        s = compute_summary_cards(metrics)
        self.assertAlmostEqual(s["avg_ece"], 0.15)

    def test_avg_ece_missing_graceful(self):
        """Metrics without test_ece should not break summary."""
        m = _make_metric()
        del m["test_ece"]
        s = compute_summary_cards([m])
        self.assertIsNone(s["avg_ece"])

    def test_empty_metrics(self):
        s = compute_summary_cards([])
        self.assertEqual(s["completed"], 0)
        self.assertIsNone(s["avg_f1"])
        self.assertIsNone(s["avg_err"])
        self.assertIsNone(s["avg_ece"])
        self.assertEqual(s["disasters_done"], 0)


class TestGenerateHtml(unittest.TestCase):
    def test_returns_string(self):
        html = generate_html([], "/tmp/fake")
        self.assertIsInstance(html, str)

    def test_contains_doctype(self):
        html = generate_html([], "/tmp/fake")
        self.assertTrue(html.strip().startswith("<!DOCTYPE html>"))

    def test_contains_title(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("LG-CoTrain", html)

    def test_contains_summary_cards(self):
        metrics = [_make_metric(macro_f1=0.616, error_rate=28.82)]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("0.616", html)

    def test_contains_pivot_tables(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("Macro-F1 by Disaster", html)
        self.assertIn("Error Rate", html)
        self.assertIn("ECE", html)
        self.assertIn("Lambda Weights", html)

    def test_contains_all_results_div(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("All Experiment Results", html)

    def test_contains_toggle_buttons(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("Pivot Summary", html)
        self.assertIn("All Results", html)
        self.assertIn("showView", html)

    def test_partial_results_no_error(self):
        metrics = [_make_metric()]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("California Wildfires 2018", html)

    def test_empty_results_no_error(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("N/A", html)

    def test_color_classes_present(self):
        html = generate_html([], "/tmp/fake")
        self.assertIn("cell-pending", html)

    def test_event_names_formatted(self):
        metrics = [_make_metric(event="hurricane_harvey_2017")]
        html = generate_html(metrics, "/tmp/fake")
        self.assertIn("Hurricane Harvey 2017", html)


class TestGenerateHtmlMulti(unittest.TestCase):
    def _make_hierarchy(self, tmpdir, model="gpt-4o", exp_type="test", exp_name="run-1"):
        """Helper: write metric under model/type/experiment and return discover result."""
        base = Path(tmpdir) / model / exp_type / exp_name
        _write_metric(str(base), _make_metric())
        return discover_result_sets(tmpdir)

    def test_returns_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_sets = self._make_hierarchy(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertTrue(html.strip().startswith("<!DOCTYPE html>"))

    def test_has_level_1_tabs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_sets = self._make_hierarchy(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertIn("level-1", html)
        self.assertIn("Data Analysis", html)
        self.assertIn("gpt-4o", html)
        self.assertIn("showL1Tab", html)

    def test_has_level_2_tabs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_sets = self._make_hierarchy(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertIn("level-2", html)
        self.assertIn("test", html)
        self.assertIn("showL2Tab", html)

    def test_has_level_3_tabs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for exp in ["run-1", "run-2"]:
                _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / exp), _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertIn("level-3", html)
        self.assertIn("run-1", html)
        self.assertIn("run-2", html)
        self.assertIn("showL3Tab", html)

    def test_tab_content_has_pivot_and_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_sets = self._make_hierarchy(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertIn("Pivot Summary", html)
        self.assertIn("All Results", html)
        self.assertIn("Macro-F1 by Disaster", html)

    def test_empty_type_shows_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_hierarchy(tmpdir)
            (Path(tmpdir) / "gpt-4o" / "stop").mkdir(parents=True)
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(result_sets)
        self.assertIn("stop (empty)", html)
        self.assertIn("No experiments yet", html)


class TestDashboardCLI(unittest.TestCase):
    def test_default_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir]):
                from lg_cotrain.dashboard import main
                main()
            self.assertTrue((Path(tmpdir) / "dashboard.html").exists())

    def test_custom_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = str(Path(tmpdir) / "custom.html")
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir,
                                     "--output", output]):
                from lg_cotrain.dashboard import main
                main()
            self.assertTrue(Path(output).exists())

    def test_writes_valid_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(tmpdir, _make_metric())
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir]):
                from lg_cotrain.dashboard import main
                main()
            content = (Path(tmpdir) / "dashboard.html").read_text()
            self.assertIn("<!DOCTYPE html>", content)
            self.assertIn("California Wildfires 2018", content)

    def test_multi_tab_cli(self):
        """CLI with 3-level structure produces multi-tab dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-2"), _make_metric())
            with patch("sys.argv", ["dashboard", "--results-root", tmpdir]):
                from lg_cotrain.dashboard import main
                main()
            content = (Path(tmpdir) / "dashboard.html").read_text()
            self.assertIn("gpt-4o", content)
            self.assertIn("run-1", content)
            self.assertIn("run-2", content)
            self.assertIn("level-1", content)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure EVENTS alias and old function signatures still work."""

    def test_events_alias(self):
        self.assertEqual(EVENTS, DEFAULT_EVENTS)

    def test_count_expected_no_args(self):
        self.assertEqual(count_expected_experiments(), 120)


# ---------------------------------------------------------------------------
# Tests for Data Analysis tab
# ---------------------------------------------------------------------------

class TestCollectDataStats(unittest.TestCase):
    """Tests for collect_data_stats()."""

    def test_returns_empty_if_no_data_dir(self):
        """Missing data_root returns {}, no crash."""
        result = collect_data_stats("/nonexistent/path/that/does/not/exist")
        self.assertEqual(result, {})

    def test_returns_empty_if_no_original_subdir(self):
        """data_root exists but has no original/ subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = collect_data_stats(tmpdir)
        self.assertEqual(result, {})

    def test_collects_train_dev_test_counts(self):
        """Standard train/dev/test files are loaded and class counts are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Path(tmpdir) / "original" / "test_event"
            orig.mkdir(parents=True)
            _write_tsv(orig / "test_event_train.tsv", [
                {"class_label": "not_humanitarian"},
                {"class_label": "not_humanitarian"},
                {"class_label": "injured_or_dead_people"},
            ])
            _write_tsv(orig / "test_event_dev.tsv", [
                {"class_label": "sympathy_and_support"},
            ])
            _write_tsv(orig / "test_event_test.tsv", [
                {"class_label": "not_humanitarian"},
            ])
            result = collect_data_stats(tmpdir)
        self.assertIn("test_event", result)
        event = result["test_event"]
        self.assertEqual(event["train"]["not_humanitarian"], 2)
        self.assertEqual(event["train"]["injured_or_dead_people"], 1)
        self.assertEqual(event["dev"]["sympathy_and_support"], 1)
        self.assertEqual(event["test"]["not_humanitarian"], 1)

    def test_collects_labeled_for_all_budgets(self):
        """labeled_{budget}_set1 keys are present for every budget in BUDGETS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Path(tmpdir) / "original" / "test_event"
            orig.mkdir(parents=True)
            for budget in BUDGETS:
                _write_tsv(orig / f"labeled_{budget}_set1.tsv", [
                    {"class_label": "not_humanitarian"} for _ in range(budget)
                ])
            result = collect_data_stats(tmpdir)
        event = result["test_event"]
        for budget in BUDGETS:
            key = f"labeled_{budget}_set1"
            self.assertIn(key, event)
            self.assertEqual(event[key]["not_humanitarian"], budget)

    def test_missing_file_gives_empty_dict_for_key(self):
        """A missing TSV file produces {} for that key — no KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Path(tmpdir) / "original" / "test_event"
            orig.mkdir(parents=True)
            _write_tsv(orig / "test_event_train.tsv", [
                {"class_label": "not_humanitarian"},
            ])
            # dev, test, labeled_*, unlabeled_* are all absent
            result = collect_data_stats(tmpdir)
        event = result["test_event"]
        self.assertEqual(event["dev"], {})
        self.assertEqual(event["test"], {})
        self.assertEqual(event["labeled_5_set1"], {})
        self.assertEqual(event["unlabeled_50_set1"], {})

    def test_multiple_events_all_present(self):
        """Two event directories both appear in the output dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for event_name in ["event_alpha", "event_beta"]:
                orig = Path(tmpdir) / "original" / event_name
                orig.mkdir(parents=True)
                _write_tsv(orig / f"{event_name}_train.tsv", [
                    {"class_label": "not_humanitarian"},
                ])
            result = collect_data_stats(tmpdir)
        self.assertIn("event_alpha", result)
        self.assertIn("event_beta", result)

    def test_graceful_without_pandas(self):
        """collect_data_stats uses pure-Python fallback when pandas is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig = Path(tmpdir) / "original" / "test_event"
            orig.mkdir(parents=True)
            _write_tsv(orig / "test_event_train.tsv", [
                {"class_label": "not_humanitarian"},
                {"class_label": "not_humanitarian"},
            ])
            with patch.dict("sys.modules", {"pandas": None}):
                result = collect_data_stats(tmpdir)
        # Pure-Python path still returns data — no longer returns {}
        self.assertIn("test_event", result)
        self.assertEqual(result["test_event"]["train"]["not_humanitarian"], 2)


class TestRenderDataTab(unittest.TestCase):
    """Tests for _render_data_tab()."""

    # Helper to build a minimal data_stats dict for one event
    def _minimal_stats(self, train_counts=None):
        file_stats = {
            "train":               train_counts or {"not_humanitarian": 10},
            "dev":                 {},
            "test":                {},
            "labeled_5_set1":      {},
            "labeled_10_set1":     {},
            "labeled_25_set1":     {},
            "labeled_50_set1":     {},
            "unlabeled_5_set1":    {},
            "unlabeled_10_set1":   {},
            "unlabeled_25_set1":   {},
            "unlabeled_50_set1":   {},
        }
        return {"test_event": file_stats}

    def test_returns_string(self):
        self.assertIsInstance(_render_data_tab({}), str)

    def test_empty_stats_returns_no_data_message(self):
        html = _render_data_tab({})
        self.assertIn("No data found", html)

    def test_renders_event_name_formatted(self):
        stats = {"california_wildfires_2018": self._minimal_stats()["test_event"]}
        html = _render_data_tab(stats)
        self.assertIn("California Wildfires 2018", html)

    def test_renders_class_labels_as_rows(self):
        stats = self._minimal_stats({"not_humanitarian": 10, "caution_and_advice": 5})
        html = _render_data_tab(stats)
        self.assertIn("not_humanitarian", html)
        self.assertIn("caution_and_advice", html)

    def test_renders_total_row(self):
        html = _render_data_tab(self._minimal_stats())
        self.assertIn("Total", html)

    def test_all_column_headers_present(self):
        html = _render_data_tab(self._minimal_stats())
        for header in ["Train", "Dev", "Test", "L5", "L10", "L25", "L50",
                       "U5", "U10", "U25", "U50"]:
            self.assertIn(header, html)

    def test_only_present_classes_rendered(self):
        """A class absent from all files for an event must not appear in output."""
        stats = self._minimal_stats({"not_humanitarian": 3})
        html = _render_data_tab(stats)
        self.assertNotIn("injured_or_dead_people", html)

    def test_event_with_no_files_shows_graceful_message(self):
        """An event where all file_stats values are {} shows a graceful fallback."""
        empty_stats = {key: {} for key in [
            "train", "dev", "test",
            "labeled_5_set1", "labeled_10_set1", "labeled_25_set1", "labeled_50_set1",
            "unlabeled_5_set1", "unlabeled_10_set1", "unlabeled_25_set1", "unlabeled_50_set1",
        ]}
        html = _render_data_tab({"test_event": empty_stats})
        self.assertIn("No data files found", html)


# New methods for existing TestGenerateHtmlMulti
class TestGenerateHtmlMultiDataTab(unittest.TestCase):
    """Data Analysis tab integration tests for generate_html_multi()."""

    def test_data_analysis_is_first_tab(self):
        """'Data Analysis' button must appear before any model tab in the HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(result_sets, data_root="/nonexistent/path")
        da_pos = html.find("Data Analysis")
        model_pos = html.find("gpt-4o")
        self.assertGreater(da_pos, -1, "Data Analysis tab not found")
        self.assertLess(da_pos, model_pos, "Data Analysis must precede model tabs")

    def test_data_tab_id_is_l1_data_analysis(self):
        """The data tab div must have id='l1-data-analysis'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"), _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(result_sets, data_root="/nonexistent/path")
        self.assertIn('id="l1-data-analysis"', html)


class TestGenerateHtmlDataTab(unittest.TestCase):
    """Data Analysis tab integration tests for generate_html()."""

    def test_data_analysis_tab_present(self):
        """generate_html() includes a Data Analysis tab."""
        html = generate_html([], "/tmp/fake", data_root="/nonexistent/path")
        self.assertIn("Data Analysis", html)
        self.assertIn('id="tab-data-analysis"', html)

    def test_data_tab_comes_before_results_tab(self):
        """Data Analysis tab button precedes the Results tab button in HTML."""
        html = generate_html([], "/tmp/fake", data_root="/nonexistent/path")
        da_pos = html.find("Data Analysis")
        results_pos = html.find(">Results<")
        self.assertGreater(da_pos, -1, "Data Analysis tab not found")
        self.assertGreater(results_pos, -1, "Results tab not found")
        self.assertLess(da_pos, results_pos)


class TestDashboardCLIDataRoot(unittest.TestCase):
    """Tests for --data-root CLI argument."""

    def test_cli_accepts_data_root_argument(self):
        """--data-root argument is accepted and dashboard is written without crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", [
                "dashboard",
                "--results-root", tmpdir,
                "--data-root", "/nonexistent/data/path",
            ]):
                from lg_cotrain.dashboard import main
                main()
            self.assertTrue((Path(tmpdir) / "dashboard.html").exists())


def _make_optuna_results():
    """Build a minimal optuna_results.json dict for testing."""
    return {
        "study_name": "lg_cotrain_global",
        "n_trials": 2,
        "best_trial": {
            "number": 1,
            "mean_dev_macro_f1": 0.6234,
            "params": {
                "lr": 3e-5,
                "batch_size": 16,
                "cotrain_epochs": 15,
                "finetune_patience": 7,
            },
        },
        "paper_defaults": {
            "lr": 2e-5,
            "batch_size": 32,
            "cotrain_epochs": 10,
            "finetune_patience": 5,
        },
        "search_space": {
            "lr": "1e-5 to 1e-3 (log-uniform)",
            "batch_size": [8, 16, 32, 64],
            "cotrain_epochs": "5 to 20",
            "finetune_patience": "4 to 10",
        },
        "trials": [
            {
                "number": 0,
                "state": "COMPLETE",
                "params": {"lr": 1e-4, "batch_size": 32,
                           "cotrain_epochs": 10, "finetune_patience": 5},
                "mean_dev_macro_f1": 0.55,
                "duration_seconds": 123.4,
            },
            {
                "number": 1,
                "state": "COMPLETE",
                "params": {"lr": 3e-5, "batch_size": 16,
                           "cotrain_epochs": 15, "finetune_patience": 7},
                "mean_dev_macro_f1": 0.6234,
                "duration_seconds": 456.7,
            },
        ],
    }


class TestLoadOptunaResults(unittest.TestCase):
    """Tests for load_optuna_results()."""

    def test_returns_none_when_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(load_optuna_results(tmpdir))

    def test_returns_none_when_dir_exists_but_no_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "optuna").mkdir()
            self.assertIsNone(load_optuna_results(tmpdir))

    def test_returns_none_for_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "optuna"
            d.mkdir()
            (d / "optuna_results.json").write_text("{invalid")
            self.assertIsNone(load_optuna_results(tmpdir))

    def test_returns_none_for_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "optuna"
            d.mkdir()
            (d / "optuna_results.json").write_text('{"foo": "bar"}')
            self.assertIsNone(load_optuna_results(tmpdir))

    def test_loads_valid_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir) / "optuna"
            d.mkdir()
            data = _make_optuna_results()
            (d / "optuna_results.json").write_text(json.dumps(data))
            result = load_optuna_results(tmpdir)
        self.assertIsNotNone(result)
        self.assertEqual(result["study_name"], "lg_cotrain_global")
        self.assertEqual(len(result["trials"]), 2)


class TestRenderOptunaTab(unittest.TestCase):
    """Tests for _render_optuna_tab()."""

    def test_returns_string(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIsInstance(html, str)

    def test_contains_study_name(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("lg_cotrain_global", html)

    def test_contains_best_trial_info(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("#1", html)
        self.assertIn("0.6234", html)

    def test_contains_paper_defaults(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("Paper Default", html)

    def test_contains_search_space(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("Search Space", html)
        self.assertIn("log-uniform", html)

    def test_contains_trial_states(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("COMPLETE", html)

    def test_pruned_trial_shown(self):
        data = _make_optuna_results()
        data["trials"].append({
            "number": 2, "state": "PRUNED",
            "params": {"lr": 5e-4, "batch_size": 64,
                       "cotrain_epochs": 5, "finetune_patience": 4},
        })
        data["n_trials"] = 3
        html = _render_optuna_tab(data)
        self.assertIn("PRUNED", html)
        self.assertIn("1 pruned", html)

    def test_sortable_columns(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn("sortAllTable('optuna'", html)
        self.assertIn("all-tbody-optuna", html)

    def test_highlights_best_trial_row(self):
        html = _render_optuna_tab(_make_optuna_results())
        self.assertIn('class="mean-row"', html)

    def test_highlights_param_differences(self):
        html = _render_optuna_tab(_make_optuna_results())
        # lr, batch_size, cotrain_epochs, finetune_patience all differ from defaults
        self.assertIn('background:#fff3cd', html)


class TestDiscoverResultSetsOptunaFiltered(unittest.TestCase):
    """Ensure optuna directory is excluded from model result sets."""

    def test_optuna_dir_not_in_result_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Normal model hierarchy
            base = Path(tmpdir) / "gpt-4o" / "test" / "run-1"
            _write_metric(str(base), _make_metric())
            # Optuna directory
            opt = Path(tmpdir) / "optuna" / "california_wildfires_2018" / "50_set1"
            opt.mkdir(parents=True)
            (opt / "metrics.json").write_text(json.dumps(_make_metric()))
            result = discover_result_sets(tmpdir)
        self.assertNotIn("optuna", result)
        self.assertIn("gpt-4o", result)


class TestGenerateHtmlMultiOptunaTab(unittest.TestCase):
    """Optuna tab integration tests for generate_html_multi()."""

    def test_optuna_tab_not_present_without_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"),
                          _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(result_sets, data_root="/nonexistent")
        self.assertNotIn('id="l1-optuna"', html)

    def test_optuna_tab_present_with_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"),
                          _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(
                result_sets, data_root="/nonexistent",
                optuna_data=_make_optuna_results(),
            )
        self.assertIn('id="l1-optuna"', html)
        self.assertIn(">Optuna<", html)

    def test_optuna_tab_between_data_and_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"),
                          _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(
                result_sets, data_root="/nonexistent",
                optuna_data=_make_optuna_results(),
            )
        da_pos = html.find("Data Analysis")
        optuna_pos = html.find(">Optuna<")
        model_pos = html.find("gpt-4o")
        self.assertGreater(da_pos, -1)
        self.assertGreater(optuna_pos, -1)
        self.assertGreater(model_pos, -1)
        self.assertLess(da_pos, optuna_pos)
        self.assertLess(optuna_pos, model_pos)

    def test_optuna_content_in_tab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_metric(str(Path(tmpdir) / "gpt-4o" / "test" / "run-1"),
                          _make_metric())
            result_sets = discover_result_sets(tmpdir)
            html = generate_html_multi(
                result_sets, data_root="/nonexistent",
                optuna_data=_make_optuna_results(),
            )
        self.assertIn("lg_cotrain_global", html)
        self.assertIn("Best Hyperparameters", html)
        self.assertIn("All Trials", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)
