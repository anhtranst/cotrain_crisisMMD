"""Generate an interactive HTML dashboard for CrisisMMD v2.0 experiments.

Tab 1: Dataset Exploration — class distributions, event breakdown, budget splits.
Tab 2+: Experiment Results — pivot tables and all-results (when metrics exist).
"""

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lg_cotrain.data_loading import TASK_LABELS
    from lg_cotrain.run_all import BUDGETS, SEED_SETS
else:
    from .data_loading import TASK_LABELS
    from .run_all import BUDGETS, SEED_SETS

TASKS = ["informative", "humanitarian", "damage"]
MODALITIES = ["text_only", "image_only", "text_image"]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_dataset_stats(data_root: str) -> dict:
    """Scan preprocessed CrisisMMD data and collect class distributions.

    Returns nested dict: {task: {modality: {split: {label: count}}}}
    """
    base = Path(data_root) / "CrisisMMD_v2.0" / "tasks"
    stats = {}
    for task in TASKS:
        stats[task] = {}
        for modality in MODALITIES:
            mod_dir = base / task / modality
            if not mod_dir.is_dir():
                continue
            stats[task][modality] = {}
            for split in ["train", "dev", "test"]:
                tsv = mod_dir / f"{split}.tsv"
                if tsv.exists():
                    stats[task][modality][split] = _count_labels(tsv)
            # Budget splits (seed 1 only for display)
            for budget in BUDGETS:
                for prefix in ["labeled", "unlabeled"]:
                    key = f"{prefix}_{budget}"
                    tsv = mod_dir / f"{prefix}_{budget}_set1.tsv"
                    if tsv.exists():
                        stats[task][modality][key] = _count_labels(tsv)
    return stats


def collect_event_stats(data_root: str) -> dict:
    """Count samples per event from the original source files.

    Returns {task: {split: {event: count}}}
    """
    base = Path(data_root) / "CrisisMMD_v2.0" / "original"
    prefixes = {
        "informative": "task_informative_text_img",
        "humanitarian": "task_humanitarian_text_img",
        "damage": "task_damage_text_img",
    }
    result = {}
    for task, prefix in prefixes.items():
        result[task] = {}
        for split in ["train", "dev", "test"]:
            tsv = base / f"{prefix}_{split}.tsv"
            if tsv.exists():
                result[task][split] = _count_column(tsv, "event_name")
    return result


def _count_labels(path: Path) -> dict:
    """Count occurrences of each class_label in a TSV."""
    counts = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("class_label", "")
            counts[label] = counts.get(label, 0) + 1
    return counts


def _count_column(path: Path, col: str) -> dict:
    """Count occurrences of each value in a specific TSV column."""
    counts = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            val = row.get(col, "")
            counts[val] = counts.get(val, 0) + 1
    return counts


def collect_all_metrics(results_root: str) -> list:
    """Recursively find all metrics.json files under results_root."""
    metrics = []
    root = Path(results_root)
    if root.exists():
        for p in sorted(root.rglob("metrics.json")):
            try:
                with open(p) as f:
                    metrics.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass
    return metrics


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --bg: #f5f6fa; --card-bg: #fff; --header-bg: #1a1a2e;
    --accent: #4361ee; --text: #2b2d42; --muted: #8d99ae;
    --border: #dee2e6; --high: #2ecc71; --mid: #f1c40f;
    --low: #e67e22; --vlow: #e74c3c; --pending: #bdc3c7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: var(--bg); color: var(--text); font-size: 14px; }
header { background: linear-gradient(135deg, var(--header-bg), #16213e);
         color: #fff; padding: 24px 32px; }
header h1 { font-size: 22px; font-weight: 600; }
header p { color: #adb5bd; font-size: 13px; margin-top: 4px; }
.tab-bar { display: flex; gap: 2px; background: #e9ecef; padding: 4px 8px 0; }
.tab-bar button { padding: 8px 18px; border: none; background: transparent;
    cursor: pointer; font-size: 13px; font-weight: 500; color: var(--muted);
    border-bottom: 2px solid transparent; transition: all .2s; }
.tab-bar button:hover { color: var(--text); }
.tab-bar button.active { color: var(--accent); border-bottom-color: var(--accent);
    background: var(--card-bg); }
.tab-content { display: none; padding: 24px 32px; }
.tab-content.active { display: block; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.card { background: var(--card-bg); border-radius: 8px; padding: 16px 20px;
        min-width: 150px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.card .value { font-size: 24px; font-weight: 700; color: var(--accent); }
.card .label { font-size: 12px; color: var(--muted); margin-top: 2px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 20px;
        background: var(--card-bg); border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.06); }
th { background: #f8f9fa; padding: 10px 12px; text-align: left;
     font-size: 12px; font-weight: 600; color: var(--muted);
     text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid var(--border); }
td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
tr:last-child td { border-bottom: none; }
.total-row td { font-weight: 600; background: #f8f9fa; }
.section-title { font-size: 16px; font-weight: 600; margin: 24px 0 12px; }
.sub-title { font-size: 14px; font-weight: 500; color: var(--muted); margin: 16px 0 8px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }
.badge-high { background: #d4edda; color: #155724; }
.badge-mid { background: #fff3cd; color: #856404; }
.badge-low { background: #f8d7da; color: #721c24; }
.heat { padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }
.heat-5 { background: rgba(67,97,238,.10); }
.heat-4 { background: rgba(67,97,238,.08); }
.heat-3 { background: rgba(67,97,238,.05); }
.heat-2 { background: rgba(67,97,238,.03); }
.heat-1 { background: rgba(67,97,238,.01); }
.note { background: #fff3cd; padding: 12px 16px; border-radius: 6px;
        border-left: 4px solid #ffc107; margin: 16px 0; font-size: 13px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
@media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
"""

# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS = """
function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-bar button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.target.classList.add('active');
}
"""

# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def _heat_class(count, max_count):
    """Return a heat-map CSS class based on count relative to max."""
    if max_count == 0:
        return "heat-1"
    ratio = count / max_count
    if ratio > 0.6:
        return "heat-5"
    if ratio > 0.3:
        return "heat-4"
    if ratio > 0.15:
        return "heat-3"
    if ratio > 0.05:
        return "heat-2"
    return "heat-1"


def _fmt(n):
    """Format a number with comma separator."""
    return f"{n:,}"


def _pct(n, total):
    """Format as percentage."""
    if total == 0:
        return "0%"
    return f"{n/total*100:.1f}%"


# ---------------------------------------------------------------------------
# Dataset Exploration Tab
# ---------------------------------------------------------------------------

def _render_dataset_overview(ds_stats, event_stats):
    """Render the Dataset Exploration tab HTML."""
    parts = []

    # Summary cards
    total_train = 0
    for task in TASKS:
        for mod in MODALITIES:
            if mod in ds_stats.get(task, {}):
                total_train += sum(ds_stats[task][mod].get("train", {}).values())
    total_classes = sum(len(v) for v in TASK_LABELS.values())
    n_events = len(set().union(*(
        event_stats.get(t, {}).get("train", {}).keys() for t in TASKS
    )))

    parts.append(f"""
    <div class="cards">
        <div class="card"><div class="value">3</div><div class="label">Tasks</div></div>
        <div class="card"><div class="value">3</div><div class="label">Modalities</div></div>
        <div class="card"><div class="value">{n_events}</div><div class="label">Disaster Events</div></div>
        <div class="card"><div class="value">{total_classes}</div><div class="label">Total Classes (across tasks)</div></div>
    </div>
    """)

    # Event distribution
    parts.append('<div class="section-title">Event Distribution (Humanitarian Task)</div>')
    hum_train_events = event_stats.get("humanitarian", {}).get("train", {})
    if hum_train_events:
        total_evt = sum(hum_train_events.values())
        max_evt = max(hum_train_events.values()) if hum_train_events else 1
        parts.append("<table><tr><th>Event</th><th>Train Samples</th><th>% of Total</th><th>Bar</th></tr>")
        for evt in sorted(hum_train_events, key=lambda e: -hum_train_events[e]):
            count = hum_train_events[evt]
            pct = count / total_evt * 100
            bar_w = count / max_evt * 200
            name = evt.replace("_", " ").title()
            parts.append(
                f'<tr><td>{name}</td><td class="heat {_heat_class(count, max_evt)}">{_fmt(count)}</td>'
                f'<td>{pct:.1f}%</td>'
                f'<td><div style="background:var(--accent);height:14px;width:{bar_w:.0f}px;border-radius:3px;opacity:.6"></div></td></tr>'
            )
        parts.append(f'<tr class="total-row"><td>Total</td><td>{_fmt(total_evt)}</td><td>100%</td><td></td></tr>')
        parts.append("</table>")

    # Per-task class distribution tables
    for task in TASKS:
        labels = TASK_LABELS.get(task, [])
        parts.append(f'<div class="section-title">Task: {task.title()} ({len(labels)} classes)</div>')

        for modality in MODALITIES:
            mod_stats = ds_stats.get(task, {}).get(modality, {})
            if not mod_stats:
                continue

            parts.append(f'<div class="sub-title">{modality.replace("_", " ").title()}</div>')

            # Build column headers
            cols = ["train", "dev", "test"]
            budget_cols = []
            for b in BUDGETS:
                if f"labeled_{b}" in mod_stats:
                    budget_cols.append(f"labeled_{b}")
                    budget_cols.append(f"unlabeled_{b}")
            all_cols = cols + budget_cols

            # Header row
            parts.append("<table><tr><th>Class</th>")
            for c in cols:
                parts.append(f"<th>{c.title()}</th>")
            for b in BUDGETS:
                if f"labeled_{b}" in mod_stats:
                    parts.append(f"<th>L{b}</th><th>U{b}</th>")
            parts.append("</tr>")

            # Find max for heat-map
            all_counts = []
            for c in all_cols:
                all_counts.extend(mod_stats.get(c, {}).values())
            max_count = max(all_counts) if all_counts else 1

            # Data rows
            totals = {c: 0 for c in all_cols}
            for label in labels:
                parts.append(f"<tr><td><code>{label}</code></td>")
                for c in all_cols:
                    count = mod_stats.get(c, {}).get(label, 0)
                    totals[c] += count
                    cls = _heat_class(count, max_count)
                    parts.append(f'<td class="heat {cls}">{_fmt(count)}</td>')
                parts.append("</tr>")

            # Total row
            parts.append('<tr class="total-row"><td>Total</td>')
            for c in all_cols:
                parts.append(f"<td>{_fmt(totals[c])}</td>")
            parts.append("</tr></table>")

    # Notes
    parts.append("""
    <div class="note">
        <strong>Label source notes:</strong>
        For informative and humanitarian tasks, <code>text_only</code> and <code>text_image</code> use
        <code>label_text</code> (text annotation), <code>image_only</code> uses <code>label_image</code>
        (image annotation). For the damage task, all modalities use the single <code>label</code> column.
        Budget columns (L5, U5, etc.) show seed=1 splits. L = labeled, U = unlabeled.
    </div>
    """)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Results Tab (placeholder for when experiments run)
# ---------------------------------------------------------------------------

def _render_results_tab(metrics):
    """Render experiment results tab."""
    if not metrics:
        return '<div class="note">No experiment results found yet. Run experiments with <code>python -m lg_cotrain.run_experiment</code> to populate this tab.</div>'

    parts = []

    # Summary cards
    n = len(metrics)
    f1s = [m["test_macro_f1"] for m in metrics if "test_macro_f1" in m]
    errs = [m["test_error_rate"] for m in metrics if "test_error_rate" in m]
    eces = [m.get("test_ece", 0) for m in metrics if "test_ece" in m]

    parts.append(f"""
    <div class="cards">
        <div class="card"><div class="value">{n}</div><div class="label">Experiments</div></div>
        <div class="card"><div class="value">{statistics.mean(f1s):.4f}</div><div class="label">Avg Macro-F1</div></div>
        <div class="card"><div class="value">{statistics.mean(errs):.1f}%</div><div class="label">Avg Error Rate</div></div>
        <div class="card"><div class="value">{statistics.mean(eces):.4f}</div><div class="label">Avg ECE</div></div>
    </div>
    """)

    # Results table
    parts.append('<div class="section-title">All Results</div>')
    parts.append("""<table>
        <tr><th>Task</th><th>Modality</th><th>Budget</th><th>Seed</th>
        <th>Test F1</th><th>Test Err%</th><th>Test ECE</th>
        <th>Dev F1</th><th>Strategy</th></tr>""")

    for m in sorted(metrics, key=lambda x: (x.get("task",""), x.get("modality",""), x.get("budget",0), x.get("seed_set",0))):
        task = m.get("task", "?")
        mod = m.get("modality", "?")
        budget = m.get("budget", "?")
        seed = m.get("seed_set", "?")
        f1 = m.get("test_macro_f1", 0)
        err = m.get("test_error_rate", 0)
        ece = m.get("test_ece", 0)
        dev_f1 = m.get("dev_macro_f1", 0)
        strat = m.get("stopping_strategy", "?")

        # Color based on F1
        if f1 >= 0.7:
            badge = "badge-high"
        elif f1 >= 0.4:
            badge = "badge-mid"
        else:
            badge = "badge-low"

        parts.append(
            f'<tr><td>{task}</td><td>{mod}</td><td>{budget}</td><td>{seed}</td>'
            f'<td><span class="badge {badge}">{f1:.4f}</span></td>'
            f'<td>{err:.2f}%</td><td>{ece:.4f}</td>'
            f'<td>{dev_f1:.4f}</td><td>{strat}</td></tr>'
        )
    parts.append("</table>")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main HTML generation
# ---------------------------------------------------------------------------

def generate_html(data_root: str, results_root: str) -> str:
    """Generate the full dashboard HTML."""
    ds_stats = collect_dataset_stats(data_root)
    event_stats = collect_event_stats(data_root)
    metrics = collect_all_metrics(results_root)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_html = _render_dataset_overview(ds_stats, event_stats)
    results_html = _render_results_tab(metrics)

    has_results = len(metrics) > 0
    results_badge = f' ({len(metrics)})' if has_results else ''

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG-CoTrain CrisisMMD Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<header>
    <h1>LG-CoTrain &mdash; CrisisMMD v2.0 Dashboard</h1>
    <p>Generated {now} &bull; 3 tasks &bull; 3 modalities &bull; {len(metrics)} experiment results</p>
</header>

<nav class="tab-bar">
    <button class="active" onclick="showTab('tab-data')">Dataset Exploration</button>
    <button onclick="showTab('tab-results')">Experiment Results{results_badge}</button>
</nav>

<div id="tab-data" class="tab-content active">
    {dataset_html}
</div>

<div id="tab-results" class="tab-content">
    {results_html}
</div>

<script>{_JS}</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_repo_root():
    """Walk up from cwd looking for a directory containing lg_cotrain/."""
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "lg_cotrain").is_dir():
            return p
    return cwd


def main():
    parser = argparse.ArgumentParser(description="Generate CrisisMMD experiment dashboard")
    parser.add_argument("--results-root", type=str, default=None,
                        help="Results directory (default: {repo}/results)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Data directory (default: {repo}/data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HTML path (default: {results-root}/dashboard.html)")
    args = parser.parse_args()

    repo = _find_repo_root()
    results_root = args.results_root or str(repo / "results")
    data_root = args.data_root or str(repo / "data")
    output = args.output or str(Path(results_root) / "dashboard.html")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    html = generate_html(data_root, results_root)

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    metrics = collect_all_metrics(results_root)
    print(f"Dashboard written to: {output}")
    print(f"  Dataset: CrisisMMD v2.0 ({len(TASKS)} tasks, {len(MODALITIES)} modalities)")
    print(f"  Experiments: {len(metrics)} results found")


if __name__ == "__main__":
    main()
