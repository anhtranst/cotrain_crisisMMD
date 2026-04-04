"""Generate an interactive HTML dashboard for CrisisMMD experiments.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lg_cotrain.data_loading import TASK_LABELS
from lg_cotrain.run_all import BUDGETS, SEED_SETS

TASKS = ["informative", "humanitarian"]
MODALITIES = ["text_only", "image_only", "text_image"]

TASK_ICONS = {"informative": "&#x1F4CB;", "humanitarian": "&#x1F6D1;", "damage": "&#x1F3DA;"}
MODALITY_ICONS = {"text_only": "&#x1F4DD;", "image_only": "&#x1F5BC;", "text_image": "&#x1F4F7;"}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_dataset_stats(data_root: str) -> dict:
    """Scan preprocessed CrisisMMD data and collect class distributions."""
    base = Path(data_root) / "CrisisMMD" / "tasks"
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
            for budget in BUDGETS:
                for prefix in ["labeled", "unlabeled"]:
                    key = f"{prefix}_{budget}"
                    tsv = mod_dir / f"{prefix}_{budget}_set1.tsv"
                    if tsv.exists():
                        stats[task][modality][key] = _count_labels(tsv)
    return stats


def collect_event_stats(data_root: str) -> dict:
    """Count samples per event from the original source files."""
    base = Path(data_root) / "CrisisMMD" / "original"
    prefixes = {
        "informative": "task_informative_text_img_agreed_lab",
        "humanitarian": "task_humanitarian_text_img_agreed_lab",
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
    counts = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("class_label", "")
            counts[label] = counts.get(label, 0) + 1
    return counts


def _count_column(path: Path, col: str) -> dict:
    counts = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            val = row.get(col, "")
            counts[val] = counts.get(val, 0) + 1
    return counts


def collect_all_metrics(results_root: str) -> list:
    """Collect co-training metrics, enriching each with method/model/run_id from path.

    Expected path structure:
        results/cotrain/{method}/{pseudo_source}/[{run_id}/]{task}/{modality}/{budget}_set{seed}/metrics.json
    """
    metrics = []
    cotrain_dir = Path(results_root) / "cotrain"
    if cotrain_dir.exists():
        for p in sorted(cotrain_dir.rglob("metrics.json")):
            if "backup" in str(p).lower():
                continue
            try:
                with open(p) as f:
                    m = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Extract hierarchy from path by finding "cotrain" directory.
            # Expected: .../cotrain/{method}/{pseudo_source}/[{run_id}/]{task}/{modality}/{budget_seed}/metrics.json
            p_parts = p.parts
            try:
                # Find the last "cotrain" component (skip "cotrain_smoke_test" etc.)
                cotrain_idx = None
                for idx in range(len(p_parts) - 1, -1, -1):
                    if p_parts[idx] == "cotrain":
                        cotrain_idx = idx
                        break
                if cotrain_idx is not None:
                    after = p_parts[cotrain_idx + 1:]  # e.g. ('lg-cotrain', 'llama-3.2-11b', 'run-1', 'informative', 'text_only', '5_set1', 'metrics.json')
                    if len(after) >= 6:
                        m.setdefault("method", after[0])
                        m.setdefault("pseudo_source", after[1])
                        if after[2].startswith("run-") or after[2].startswith("run_"):
                            m.setdefault("run_id", after[2])
                        else:
                            m.setdefault("run_id", "default")
            except (IndexError, ValueError):
                pass

            metrics.append(m)
    return metrics


def collect_zeroshot_results(results_root: str) -> dict:
    """Scan results/zeroshot/{model}/{task}/{modality}/{split}/metrics.json.

    Returns {task: [metrics_dict, ...]}.
    """
    base = Path(results_root) / "zeroshot"
    results = {}
    if not base.exists():
        return results
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            if task not in results:
                results[task] = []
            for mod_dir in sorted(task_dir.iterdir()):
                if not mod_dir.is_dir():
                    continue
                for split_dir in sorted(mod_dir.iterdir()):
                    if not split_dir.is_dir():
                        continue
                    metrics_path = split_dir / "metrics.json"
                    if metrics_path.exists():
                        try:
                            with open(metrics_path) as f:
                                m = json.load(f)
                                m["model_slug"] = model_dir.name
                                results[task].append(m)
                        except (json.JSONDecodeError, OSError):
                            pass
    return results


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = r"""
:root {
    --bg: #f0f2f5;
    --card: #ffffff;
    --header-from: #0f0c29;
    --header-via: #302b63;
    --header-to: #24243e;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --accent-bg: rgba(99,102,241,.08);
    --text: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,.05), 0 2px 4px rgba(0,0,0,.03);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Header */
header {
    background: linear-gradient(135deg, var(--header-from), var(--header-via), var(--header-to));
    color: #fff; padding: 32px 40px; position: relative; overflow: hidden;
}
header::after {
    content: ''; position: absolute; top: -50%; right: -10%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,.15) 0%, transparent 70%);
    border-radius: 50%;
}
header h1 { font-size: 26px; font-weight: 700; letter-spacing: -.5px; position: relative; }
header .subtitle { color: rgba(255,255,255,.6); font-size: 13px; margin-top: 6px; position: relative; }
header .subtitle span { color: rgba(255,255,255,.85); font-weight: 500; }

/* Tab bar */
.tab-bar {
    display: flex; gap: 0; background: var(--card); padding: 0 40px;
    border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10;
    box-shadow: var(--shadow);
}
.tab-bar button {
    padding: 14px 24px; border: none; background: transparent; cursor: pointer;
    font-size: 14px; font-weight: 500; color: var(--text-secondary);
    border-bottom: 2px solid transparent; transition: all .2s ease;
    position: relative;
}
.tab-bar button:hover { color: var(--text); background: var(--accent-bg); }
.tab-bar button.active {
    color: var(--accent); border-bottom-color: var(--accent); font-weight: 600;
}
.tab-content { display: none; padding: 32px 40px; max-width: 1400px; }
.tab-content.active { display: block; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }
.card {
    background: var(--card); border-radius: var(--radius); padding: 20px 24px;
    box-shadow: var(--shadow); border: 1px solid var(--border);
    transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.card .icon { font-size: 20px; margin-bottom: 8px; }
.card .value { font-size: 28px; font-weight: 800; color: var(--accent); line-height: 1.1; }
.card .label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: .5px; }

/* Sections */
.section {
    background: var(--card); border-radius: var(--radius); padding: 24px 28px;
    margin-bottom: 24px; box-shadow: var(--shadow); border: 1px solid var(--border);
}
.section-highlight {
    border-left: 4px solid var(--accent);
    background: linear-gradient(135deg, var(--card) 0%, var(--accent-bg) 100%);
    box-shadow: var(--shadow-md);
}
.run-notes {
    border-left: 4px solid var(--accent);
    background: var(--accent-bg);
    padding: 14px 20px;
    border-radius: var(--radius);
    margin-bottom: 20px;
    font-size: 13px;
    line-height: 1.5;
    color: var(--text);
}
.run-notes strong { color: var(--accent); font-size: 14px; }
.run-notes table {
    margin-top: 10px; border-collapse: collapse; font-size: 12px;
    background: transparent; box-shadow: none;
}
.run-notes td {
    padding: 2px 12px 2px 0; border: none; vertical-align: top;
}
.run-notes td:first-child {
    font-weight: 600; color: var(--text-secondary); white-space: nowrap;
}
.section-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border);
}
.section-header h2 { font-size: 18px; font-weight: 700; }
.section-header .tag {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
    background: var(--accent-bg); color: var(--accent);
}

/* Collapsible task/modality sections */
.collapse-toggle {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; background: #f8fafc; border-radius: 8px;
    cursor: pointer; margin-bottom: 8px; border: 1px solid var(--border);
    transition: background .15s;
}
.collapse-toggle:hover { background: #f1f5f9; }
.collapse-toggle h3 { font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.collapse-toggle .arrow { transition: transform .2s; font-size: 12px; color: var(--text-secondary); }
.collapse-toggle.open .arrow { transform: rotate(90deg); }
.collapse-body { display: none; padding: 16px 0 8px; }
.collapse-body.open { display: block; }

/* Tables */
table { border-collapse: collapse; width: 100%; font-size: 13px; }
thead th {
    background: #f8fafc; padding: 10px 14px; text-align: left;
    font-size: 11px; font-weight: 700; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: .6px;
    border-bottom: 2px solid var(--border); position: sticky; top: 0;
}
thead th.num { text-align: right; }
tbody td { padding: 9px 14px; border-bottom: 1px solid #f1f5f9; }
tbody td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: 'SF Mono', 'Cascadia Code', monospace; font-size: 12.5px; }
tbody tr:hover { background: #fafbfe; }
tbody tr:last-child td { border-bottom: none; }
.total-row td { font-weight: 700; background: #f8fafc; border-top: 2px solid var(--border); }
td code { font-size: 12px; padding: 2px 6px; background: #f1f5f9; border-radius: 4px; color: #475569; }

/* Bar inside table */
.bar-cell { padding: 9px 14px; }
.bar-outer { background: #f1f5f9; border-radius: 6px; height: 22px; overflow: hidden; position: relative; min-width: 60px; }
.bar-inner { height: 100%; border-radius: 6px; transition: width .3s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 6px; }
.bar-inner span { font-size: 10px; font-weight: 700; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,.2); }
.bar-purple { background: linear-gradient(90deg, #818cf8, #6366f1); }
.bar-blue { background: linear-gradient(90deg, #60a5fa, #3b82f6); }
.bar-green { background: linear-gradient(90deg, #34d399, #10b981); }

/* Heat map cells */
.hm { text-align: right; font-variant-numeric: tabular-nums; font-family: 'SF Mono', monospace; font-size: 12px; }
.hm-5 { background: rgba(99,102,241,.14); color: #4338ca; font-weight: 600; }
.hm-4 { background: rgba(99,102,241,.09); color: #4f46e5; }
.hm-3 { background: rgba(99,102,241,.05); }
.hm-2 { background: rgba(99,102,241,.025); }
.hm-1 { color: #94a3b8; }
.hm-0 { color: #cbd5e1; }

/* Badges */
.badge { display: inline-flex; align-items: center; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-success { background: #d1fae5; color: #065f46; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.badge-info { background: var(--accent-bg); color: var(--accent); }

/* Notes */
.note {
    display: flex; gap: 12px; padding: 16px 20px; border-radius: 8px;
    border: 1px solid #e2e8f0; margin: 20px 0; font-size: 13px;
    background: linear-gradient(135deg, #fffbeb, #fef9c3); border-left: 4px solid var(--warning);
}
.note-icon { font-size: 18px; flex-shrink: 0; }
.note-body { line-height: 1.7; }
.note-body strong { color: #92400e; }
.note-body code { font-size: 12px; padding: 1px 5px; background: rgba(0,0,0,.06); border-radius: 3px; }

/* Empty state */
.empty-state {
    text-align: center; padding: 60px 20px; color: var(--text-secondary);
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: .5; }
.empty-state h3 { font-size: 18px; font-weight: 600; color: var(--text); margin-bottom: 8px; }
.empty-state p { max-width: 400px; margin: 0 auto; }

/* Sub-tabs (within a tab) */
.sub-tab-bar {
    display: flex; gap: 0; margin: 0 0 16px; border-bottom: 2px solid var(--border);
}
.sub-tab-bar button {
    padding: 8px 20px; border: none; background: none; cursor: pointer;
    font-size: 14px; font-weight: 500; color: var(--text-secondary);
    border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all .15s;
}
.sub-tab-bar button:hover { background: var(--accent-bg); }
.sub-tab-bar button.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
.sub-tab-content { display: none; }
.sub-tab-content.active { display: block; }

/* Model header for per-model card rows */
.model-header {
    font-size: 14px; font-weight: 600; color: var(--text-secondary);
    margin: 20px 0 8px; padding: 4px 0;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
}
.model-header:first-child { margin-top: 0; }
.model-header code { font-size: 13px; color: var(--accent); background: var(--accent-bg); padding: 2px 8px; border-radius: 4px; }
.model-header .tag { font-size: 11px; color: var(--text-secondary); font-weight: 400; }

/* Sortable table headers */
th.sortable { cursor: pointer; user-select: none; position: relative; }
th.sortable:hover { background: var(--accent-bg); }
th.sortable::after { content: ' \2195'; font-size: 10px; opacity: .4; }
th.sortable.sort-asc::after { content: ' \25B2'; opacity: .8; }
th.sortable.sort-desc::after { content: ' \25BC'; opacity: .8; }

/* Best/worst cell highlighting */
.cell-best { background: #d1fae5; font-weight: 600; }
.cell-worst { background: #fee2e2; }

/* Responsive */
@media (max-width: 768px) {
    header, .tab-bar, .tab-content { padding-left: 20px; padding-right: 20px; }
    .cards { grid-template-columns: repeat(2, 1fr); }
}
"""


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

_JS = r"""
function showTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
}

function showSubTab(parentId, tabId, btn) {
    var parent = document.getElementById(parentId);
    // Only toggle direct child sub-tab-content and sibling buttons (not nested ones)
    Array.from(parent.children).forEach(function(el) {
        if (el.classList.contains('sub-tab-content')) el.classList.remove('active');
    });
    var bar = btn.parentElement;
    bar.querySelectorAll('button').forEach(function(b) { b.classList.remove('active'); });
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
}

function toggleCollapse(id) {
    const body = document.getElementById(id);
    const toggle = body.previousElementSibling;
    body.classList.toggle('open');
    toggle.classList.toggle('open');
}

function sortTable(tableId, colIdx) {
    var table = document.getElementById(tableId);
    var tbody = table.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var th = table.querySelectorAll('thead th')[colIdx];

    // Determine sort direction
    var asc = !th.classList.contains('sort-asc');
    table.querySelectorAll('thead th').forEach(function(h) {
        h.classList.remove('sort-asc', 'sort-desc');
    });
    th.classList.add(asc ? 'sort-asc' : 'sort-desc');

    rows.sort(function(a, b) {
        var aVal = a.cells[colIdx].getAttribute('data-val') || a.cells[colIdx].textContent.trim();
        var bVal = b.cells[colIdx].getAttribute('data-val') || b.cells[colIdx].textContent.trim();
        var aNum = parseFloat(aVal), bNum = parseFloat(bVal);
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return asc ? aNum - bNum : bNum - aNum;
        }
        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
    });
    rows.forEach(function(row) { tbody.appendChild(row); });
    highlightBestWorst(tableId);
}

function highlightBestWorst(tableId) {
    var table = document.getElementById(tableId);
    var tbody = table.querySelector('tbody');
    var rows = Array.from(tbody.querySelectorAll('tr'));
    var numCols = rows.length > 0 ? rows[0].cells.length : 0;

    // Clear existing highlights
    rows.forEach(function(r) {
        Array.from(r.cells).forEach(function(c) {
            c.classList.remove('cell-best', 'cell-worst');
        });
    });

    // Group rows by split (column index 1)
    var splitGroups = {};
    rows.forEach(function(row) {
        var split = row.cells[1].textContent.trim();
        if (!splitGroups[split]) splitGroups[split] = [];
        splitGroups[split].push(row);
    });

    // For each numeric column (index 3+), find best/worst per split
    for (var col = 3; col < numCols; col++) {
        for (var split in splitGroups) {
            var group = splitGroups[split];
            if (group.length < 2) continue;
            var vals = group.map(function(r) {
                return { row: r, val: parseFloat(r.cells[col].getAttribute('data-val') || r.cells[col].textContent) };
            }).filter(function(v) { return !isNaN(v.val); });
            if (vals.length < 2) continue;
            var best = vals.reduce(function(a, b) { return a.val > b.val ? a : b; });
            var worst = vals.reduce(function(a, b) { return a.val < b.val ? a : b; });
            if (best.val !== worst.val) {
                best.row.cells[col].classList.add('cell-best');
                worst.row.cells[col].classList.add('cell-worst');
            }
        }
    }
}

// Run highlighting on page load for all zeroshot tables
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('table[id^="zs-table-"]').forEach(function(t) {
        highlightBestWorst(t.id);
    });
});
"""


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _hm_class(count, max_count):
    """Return a heat-map CSS class."""
    if count == 0:
        return "hm hm-0"
    if max_count == 0:
        return "hm hm-1"
    ratio = count / max_count
    if ratio > 0.5:
        return "hm hm-5"
    if ratio > 0.25:
        return "hm hm-4"
    if ratio > 0.10:
        return "hm hm-3"
    if ratio > 0.03:
        return "hm hm-2"
    return "hm hm-1"


def _fmt(n):
    return f"{n:,}"


def _modality_label(m):
    return m.replace("_", " ").title()


def _collapse_id():
    _collapse_id.counter = getattr(_collapse_id, 'counter', 0) + 1
    return f"collapse-{_collapse_id.counter}"


# ---------------------------------------------------------------------------
# Dataset Exploration Tab
# ---------------------------------------------------------------------------

def _render_dataset_overview(ds_stats, event_stats):
    parts = []

    # Compute summary numbers
    n_events = len(set().union(*(
        event_stats.get(t, {}).get("train", {}).keys() for t in TASKS
    )))
    train_counts = {}
    for task in TASKS:
        for mod in MODALITIES:
            t = sum(ds_stats.get(task, {}).get(mod, {}).get("train", {}).values())
            if t:
                train_counts[f"{task}/{mod}"] = t

    total_samples = sum(
        sum(ds_stats.get(t, {}).get(m, {}).get(s, {}).values())
        for t in TASKS for m in MODALITIES for s in ["train", "dev", "test"]
        if m in ds_stats.get(t, {})
    )

    # Summary cards
    parts.append(f"""
    <div class="cards">
        <div class="card">
            <div class="icon">&#x1F30D;</div>
            <div class="value">7</div>
            <div class="label">Combined Events</div>
        </div>
        <div class="card">
            <div class="icon">&#x1F4CA;</div>
            <div class="value">{len(TASKS)}</div>
            <div class="label">Annotation Tasks</div>
        </div>
        <div class="card">
            <div class="icon">&#x1F50D;</div>
            <div class="value">{len(MODALITIES)}</div>
            <div class="label">Modalities</div>
        </div>
        <div class="card">
            <div class="icon">&#x1F3F7;</div>
            <div class="value">{sum(len(v) for v in TASK_LABELS.values())}</div>
            <div class="label">Total Classes</div>
        </div>
        <div class="card">
            <div class="icon">&#x1F4E6;</div>
            <div class="value">{_fmt(total_samples)}</div>
            <div class="label">Total Samples</div>
        </div>
    </div>
    """)

    # --- Event Distribution ---
    parts.append('<div class="section">')
    parts.append('<div class="section-header"><h2>&#x1F30D; Event Distribution</h2>'
                 '<div class="tag">Humanitarian Task &mdash; Training Set</div></div>')
    hum_train = event_stats.get("humanitarian", {}).get("train", {})
    if hum_train:
        total_evt = sum(hum_train.values())
        max_evt = max(hum_train.values())
        parts.append('<table><thead><tr><th>Event</th><th class="num">Samples</th>'
                     '<th class="num">Share</th><th style="min-width:250px">Distribution</th></tr></thead><tbody>')
        colors = ["bar-purple", "bar-purple", "bar-purple", "bar-blue", "bar-blue", "bar-blue", "bar-green"]
        for i, evt in enumerate(sorted(hum_train, key=lambda e: -hum_train[e])):
            count = hum_train[evt]
            pct = count / total_evt * 100
            bar_w = count / max_evt * 100
            name = evt.replace("_", " ").title()
            color = colors[i % len(colors)]
            parts.append(
                f'<tr><td><strong>{name}</strong></td>'
                f'<td class="num">{_fmt(count)}</td>'
                f'<td class="num">{pct:.1f}%</td>'
                f'<td class="bar-cell"><div class="bar-outer">'
                f'<div class="bar-inner {color}" style="width:{bar_w:.1f}%">'
                f'<span>{pct:.0f}%</span></div></div></td></tr>'
            )
        parts.append(f'<tr class="total-row"><td>Total</td><td class="num">{_fmt(total_evt)}</td>'
                     f'<td class="num">100%</td><td></td></tr>')
        parts.append('</tbody></table>')
    parts.append('</div>')

    # --- Per-task sections ---
    for task in TASKS:
        labels = TASK_LABELS.get(task, [])
        icon = TASK_ICONS.get(task, "")

        parts.append(f'<div class="section">')
        parts.append(f'<div class="section-header">'
                     f'<h2>{icon} {task.title()} Task</h2>'
                     f'<div class="tag">{len(labels)} classes</div></div>')

        # Task overview: show train/dev/test totals per modality
        parts.append('<table><thead><tr><th>Modality</th><th class="num">Train</th>'
                     '<th class="num">Dev</th><th class="num">Test</th>'
                     '<th class="num">Total</th></tr></thead><tbody>')
        for mod in MODALITIES:
            ms = ds_stats.get(task, {}).get(mod, {})
            if not ms:
                continue
            tr = sum(ms.get("train", {}).values())
            dv = sum(ms.get("dev", {}).values())
            te = sum(ms.get("test", {}).values())
            mi = MODALITY_ICONS.get(mod, "")
            parts.append(f'<tr><td>{mi} {_modality_label(mod)}</td>'
                         f'<td class="num">{_fmt(tr)}</td>'
                         f'<td class="num">{_fmt(dv)}</td>'
                         f'<td class="num">{_fmt(te)}</td>'
                         f'<td class="num"><strong>{_fmt(tr+dv+te)}</strong></td></tr>')
        parts.append('</tbody></table>')

        # Collapsible per-modality class distribution
        for mod in MODALITIES:
            mod_stats = ds_stats.get(task, {}).get(mod, {})
            if not mod_stats:
                continue

            cid = _collapse_id()
            mi = MODALITY_ICONS.get(mod, "")
            parts.append(
                f'<div class="collapse-toggle" onclick="toggleCollapse(\'{cid}\')">'
                f'<h3>{mi} {_modality_label(mod)} &mdash; Class Distribution</h3>'
                f'<span class="arrow">&#x25B6;</span></div>'
                f'<div class="collapse-body" id="{cid}">'
            )

            # Columns
            cols = ["train", "dev", "test"]
            budget_cols = []
            for b in BUDGETS:
                if f"labeled_{b}" in mod_stats:
                    budget_cols.append(f"labeled_{b}")
                    budget_cols.append(f"unlabeled_{b}")
            all_cols = cols + budget_cols

            # Max for heat-map
            all_counts = []
            for c in all_cols:
                all_counts.extend(mod_stats.get(c, {}).values())
            max_count = max(all_counts) if all_counts else 1

            parts.append('<table><thead><tr><th>Class</th>')
            for c in cols:
                parts.append(f'<th class="num">{c.title()}</th>')
            for b in BUDGETS:
                if f"labeled_{b}" in mod_stats:
                    parts.append(f'<th class="num">L{b}</th><th class="num">U{b}</th>')
            parts.append('</tr></thead><tbody>')

            totals = {c: 0 for c in all_cols}
            for label in labels:
                parts.append(f'<tr><td><code>{label}</code></td>')
                for c in all_cols:
                    count = mod_stats.get(c, {}).get(label, 0)
                    totals[c] += count
                    # Warn if unlabeled count is less than the budget
                    warn = ""
                    if c.startswith("unlabeled_"):
                        budget_val = int(c.split("_")[1])
                        labeled_count = mod_stats.get(f"labeled_{budget_val}", {}).get(label, 0)
                        if count > 0 and count < budget_val:
                            warn = ' <span style="color:var(--danger);font-size:10px" title="Unlabeled samples fewer than budget">&#x26A0;</span>'
                    parts.append(f'<td class="{_hm_class(count, max_count)}">{_fmt(count)}{warn}</td>')
                parts.append('</tr>')

            parts.append('<tr class="total-row"><td><strong>Total</strong></td>')
            for c in all_cols:
                parts.append(f'<td class="num">{_fmt(totals[c])}</td>')
            parts.append('</tr></tbody></table>')

            parts.append('</div>')  # collapse-body

        parts.append('</div>')  # section

    # Notes
    parts.append("""
    <div class="note">
        <div class="note-icon">&#x1F4DD;</div>
        <div class="note-body">
            <strong>Dataset notes:</strong>
            This dataset uses the <strong>CrisisMMD agreed-label subset</strong> where text and image
            annotators agreed on the label. Labels are from <code>class_label</code> (unified column
            after preprocessing). Two tasks: informative (2 classes) and humanitarian (5 classes),
            across 7 combined disaster events.
            Budget columns (L5, U5, etc.) show seed=1 splits. <strong>L</strong> = labeled,
            <strong>U</strong> = unlabeled. &#x26A0; marks unlabeled splits where the remaining
            samples are fewer than the budget (e.g., only 20 unlabeled samples at budget=50).
        </div>
    </div>
    """)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Zero-Shot Results Tab (per-task)
# ---------------------------------------------------------------------------

_table_counter = 0

def _next_table_id():
    global _table_counter
    _table_counter += 1
    return f"zs-table-{_table_counter}"


def _render_zeroshot_tab(task, task_results):
    """Render a zero-shot results sub-tab for one task."""
    if not task_results:
        return f"""
        <div class="empty-state">
            <div class="icon">&#x1F4CA;</div>
            <h3>No Zero-Shot Results for {task.title()}</h3>
            <p>Run the notebook or CLI to generate results.</p>
        </div>"""

    parts = []

    # --- Summary table: per-model, per-split averages ---
    models = sorted(set(m.get("model_slug", "?") for m in task_results))
    summary_table_id = _next_table_id()
    parts.append('<div class="section section-highlight">')
    parts.append('<div class="section-header"><h2>&#x1F4CA; Summary Averages (across modalities)</h2></div>')
    parts.append(f'<table id="{summary_table_id}"><thead><tr>')
    for ci, h in enumerate(["Model", "Split", "Experiments", "Avg Accuracy", "Avg Weighted F1", "Avg Macro F1", "Avg Confidence"]):
        cls = "num " if ci >= 2 else ""
        parts.append(f'<th class="{cls}sortable" onclick="sortTable(\'{summary_table_id}\',{ci})">{h}</th>')
    parts.append('</tr></thead><tbody>')

    for model in models:
        for split in ["test", "train"]:
            split_metrics = [m for m in task_results
                            if m.get("model_slug") == model and m.get("split") == split]
            if not split_metrics:
                continue
            n = len(split_metrics)
            avg_acc = statistics.mean([m["accuracy"] for m in split_metrics])
            avg_wf1 = statistics.mean([m["weighted_f1"] for m in split_metrics])
            avg_mf1 = statistics.mean([m["macro_f1"] for m in split_metrics])
            conf_vals = [m["avg_confidence"] for m in split_metrics if "avg_confidence" in m]
            avg_conf = statistics.mean(conf_vals) if conf_vals else None
            conf_cell = f'{avg_conf:.4f}' if avg_conf is not None else '-'

            parts.append(
                f'<tr><td><code>{model}</code></td><td>{split}</td>'
                f'<td class="num">{n}</td>'
                f'<td class="num" data-val="{avg_acc:.6f}">{avg_acc:.4f}</td>'
                f'<td class="num" data-val="{avg_wf1:.6f}">{avg_wf1:.4f}</td>'
                f'<td class="num" data-val="{avg_mf1:.6f}">{avg_mf1:.4f}</td>'
                f'<td class="num">{conf_cell}</td></tr>'
            )

    parts.append('</tbody></table></div>')

    # --- Results tables grouped by modality ---
    modality_order = ["text_only", "image_only", "text_image"]
    for modality in modality_order:
        mod_results = [m for m in task_results if m.get("modality") == modality]
        if not mod_results:
            continue

        mi = MODALITY_ICONS.get(modality, "")
        table_id = _next_table_id()
        parts.append('<div class="section">')
        parts.append(f'<div class="section-header"><h2>{mi} {_modality_label(modality)}</h2></div>')
        parts.append(f'<table id="{table_id}"><thead><tr>')
        headers = ["Model", "Split", "Samples", "Unparseable", "Accuracy", "W. Precision", "W. Recall", "W. F1", "Macro F1"]
        for ci, h in enumerate(headers):
            parts.append(f'<th class="{"num " if ci >= 2 else ""}sortable" onclick="sortTable(\'{table_id}\',{ci})">{h}</th>')
        parts.append('</tr></thead><tbody>')

        for m in sorted(mod_results, key=lambda x: (x.get("split", ""), x.get("model_slug", ""))):
            acc = m.get("accuracy", 0) * 100
            prec = m.get("weighted_precision", 0) * 100
            rec = m.get("weighted_recall", 0) * 100
            wf1 = m.get("weighted_f1", 0) * 100
            mf1 = m.get("macro_f1", 0) * 100
            unparse = m.get("num_unparseable", 0)
            parts.append(
                f'<tr>'
                f'<td><code>{m.get("model_slug","?")}</code></td>'
                f'<td data-val="{m.get("split","")}">{m.get("split","?")}</td>'
                f'<td class="num" data-val="{m.get("num_samples",0)}">{m.get("num_samples", 0):,}</td>'
                f'<td class="num" data-val="{unparse}">{unparse}</td>'
                f'<td class="num" data-val="{acc:.2f}">{acc:.2f}</td>'
                f'<td class="num" data-val="{prec:.2f}">{prec:.2f}</td>'
                f'<td class="num" data-val="{rec:.2f}">{rec:.2f}</td>'
                f'<td class="num" data-val="{wf1:.2f}">{wf1:.2f}</td>'
                f'<td class="num" data-val="{mf1:.2f}">{mf1:.2f}</td>'
                f'</tr>'
            )
        parts.append("</tbody></table></div>")

    # --- Per-class F1 table (test sets only) ---
    test_results = [m for m in task_results if m.get("split") == "test" and m.get("per_class_f1")]
    if test_results:
        labels = sorted(test_results[0]["per_class_f1"].keys())
        parts.append('<div class="section">')
        parts.append('<div class="section-header"><h2>&#x1F3F7; Per-Class F1 (Test Set)</h2></div>')
        parts.append('<table><thead><tr><th>Class</th>')
        for m in sorted(test_results, key=lambda x: (modality_order.index(x.get("modality","")) if x.get("modality","") in modality_order else 99, x.get("model_slug",""))):
            mi = MODALITY_ICONS.get(m.get("modality", ""), "")
            slug = m.get("model_slug", "")
            parts.append(f'<th class="num">{slug}<br>{mi} {_modality_label(m.get("modality",""))}</th>')
        parts.append('</tr></thead><tbody>')

        for label in labels:
            parts.append(f'<tr><td><code>{label}</code></td>')
            for m in sorted(test_results, key=lambda x: (modality_order.index(x.get("modality","")) if x.get("modality","") in modality_order else 99, x.get("model_slug",""))):
                f1 = m["per_class_f1"].get(label, 0)
                if f1 >= 0.5:
                    cls = "hm hm-5"
                elif f1 >= 0.3:
                    cls = "hm hm-4"
                elif f1 >= 0.1:
                    cls = "hm hm-3"
                elif f1 > 0:
                    cls = "hm hm-2"
                else:
                    cls = "hm hm-0"
                parts.append(f'<td class="{cls}">{f1:.4f}</td>')
            parts.append('</tr>')
        parts.append('</tbody></table></div>')

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Co-Training Results Tab
# ---------------------------------------------------------------------------

def _render_cotrain_summary_cards(metrics):
    """Render summary cards for a set of co-training metrics."""
    n = len(metrics)
    f1s = [m["test_macro_f1"] for m in metrics if "test_macro_f1" in m]
    errs = [m["test_error_rate"] for m in metrics if "test_error_rate" in m]
    eces = [m.get("test_ece", 0) for m in metrics if "test_ece" in m]
    if not f1s:
        return ""
    return f"""
    <div class="cards">
        <div class="card"><div class="icon">&#x1F9EA;</div><div class="value">{n}</div><div class="label">Experiments</div></div>
        <div class="card"><div class="icon">&#x1F3AF;</div><div class="value">{statistics.mean(f1s):.4f}</div><div class="label">Avg Macro-F1</div></div>
        <div class="card"><div class="icon">&#x274C;</div><div class="value">{statistics.mean(errs):.1f}%</div><div class="label">Avg Error Rate</div></div>
        <div class="card"><div class="icon">&#x1F4CF;</div><div class="value">{statistics.mean(eces):.4f}</div><div class="label">Avg ECE</div></div>
    </div>"""


def _render_cotrain_task_tables(task, task_metrics, zs_results=None, pseudo_source=None):
    """Render tables for one task: one table per modality, rows = budgets (averaged across seeds).

    Zero-shot baselines are shown as reference rows at the top of each table.
    The baseline matching pseudo_source is highlighted.
    """
    parts = []
    modality_order = ["text_only", "image_only", "text_image"]
    zs_results = zs_results or []

    for modality in modality_order:
        mod_metrics = [m for m in task_metrics if m.get("modality") == modality]
        if not mod_metrics:
            continue

        icon = MODALITY_ICONS.get(modality, "")
        parts.append(f'<div class="section">')
        parts.append(f'<div class="section-header"><h3>{icon} {modality}</h3></div>')

        # Group by budget, average across seeds
        by_budget = {}
        for m in mod_metrics:
            b = m.get("budget", 0)
            by_budget.setdefault(b, []).append(m)

        parts.append("""<table><thead>
            <tr><th>Method</th><th class="num">Budget</th><th class="num">Seeds</th>
            <th class="num">Test Macro-F1</th><th class="num">Test W. F1</th>
            <th class="num">Test Err%</th>
            <th class="num">Test ECE</th><th class="num">Dev Macro-F1</th></tr></thead><tbody>""")

        # Zero-shot baseline rows (test split only)
        zs_mod = [m for m in zs_results
                  if m.get("modality") == modality and m.get("split") == "test"]

        # Extract the highlighted baseline's F1 values for comparison
        baseline_macro_f1 = None
        baseline_wf1 = None
        for m in zs_mod:
            if pseudo_source and m.get("model_slug") == pseudo_source:
                baseline_macro_f1 = m.get("macro_f1")
                baseline_wf1 = m.get("weighted_f1")
                break

        for m in sorted(zs_mod, key=lambda x: x.get("model_slug", "")):
            zs_f1 = m.get("macro_f1", 0)
            zs_wf1 = m.get("weighted_f1")
            zs_err = (1.0 - m.get("accuracy", 0)) * 100
            wf1_cell = f"{zs_wf1:.4f}" if zs_wf1 is not None else "-"
            slug = m.get("model_slug", "?")
            # Highlight the baseline that was used to generate pseudo-labels
            is_source = pseudo_source and slug == pseudo_source
            if is_source:
                row_style = 'style="background:var(--accent-bg); font-weight:700; border-left:4px solid var(--accent);"'
                label = f"ZS: {slug} &#x2B50;"
            else:
                row_style = 'style="background:var(--accent-bg); font-style:italic; opacity:0.7;"'
                label = f"ZS: {slug}"
            wf1_hm = _hm(zs_wf1) if zs_wf1 is not None else ""
            parts.append(
                f'<tr {row_style}>'
                f'<td>{label}</td>'
                f'<td class="num">-</td><td class="num">-</td>'
                f'<td class="num" {_hm(zs_f1)}>{zs_f1:.4f}</td>'
                f'<td class="num" {wf1_hm}>{wf1_cell}</td>'
                f'<td class="num" {_hm(zs_err, inverted=True, lo=10, hi=30)}>{zs_err:.2f}%</td>'
                f'<td class="num">-</td><td class="num">-</td></tr>'
            )

        # Co-training rows
        def _compare_badge(val, baseline):
            """Return badge class comparing val against the highlighted baseline."""
            if baseline is None:
                return "badge-warning"
            if val > baseline + 1e-6:
                return "badge-success"
            elif val >= baseline - 1e-6:
                return "badge-warning"
            else:
                return "badge-danger"

        for budget in sorted(by_budget.keys()):
            runs = by_budget[budget]
            n_seeds = len(runs)
            avg_f1 = statistics.mean(m["test_macro_f1"] for m in runs)
            std_f1 = statistics.stdev(m["test_macro_f1"] for m in runs) if n_seeds > 1 else 0
            avg_wf1_vals = [m.get("test_weighted_f1") for m in runs if m.get("test_weighted_f1") is not None]
            avg_wf1 = statistics.mean(avg_wf1_vals) if avg_wf1_vals else None
            avg_err = statistics.mean(m["test_error_rate"] for m in runs)
            avg_ece = statistics.mean(m.get("test_ece", 0) for m in runs)
            avg_dev_f1 = statistics.mean(m["dev_macro_f1"] for m in runs)

            f1_badge = _compare_badge(avg_f1, baseline_macro_f1)
            f1_display = f"{avg_f1:.4f}" if n_seeds == 1 else f"{avg_f1:.4f} &plusmn; {std_f1:.4f}"

            if avg_wf1 is not None:
                wf1_badge = _compare_badge(avg_wf1, baseline_wf1)
                std_wf1 = statistics.stdev(v for v in avg_wf1_vals) if len(avg_wf1_vals) > 1 else 0
                wf1_text = f"{avg_wf1:.4f}" if len(avg_wf1_vals) <= 1 else f"{avg_wf1:.4f} &plusmn; {std_wf1:.4f}"
                wf1_display = f'<span class="badge {wf1_badge}">{wf1_text}</span>'
                wf1_hm = _hm(avg_wf1)
            else:
                wf1_display = "-"
                wf1_hm = ""

            parts.append(
                f'<tr><td>Co-Train</td><td class="num">{budget}</td><td class="num">{n_seeds}</td>'
                f'<td class="num" {_hm(avg_f1)}><span class="badge {f1_badge}">{f1_display}</span></td>'
                f'<td class="num" {wf1_hm}>{wf1_display}</td>'
                f'<td class="num" {_hm(avg_err, inverted=True, lo=10, hi=30)}>{avg_err:.2f}%</td>'
                f'<td class="num" {_hm(avg_ece, inverted=True, lo=0.04, hi=0.20)}>{avg_ece:.4f}</td>'
                f'<td class="num" {_hm(avg_dev_f1)}>{avg_dev_f1:.4f}</td></tr>'
            )

        parts.append("</tbody></table></div>")

    return "\n".join(parts)


def _f1_to_heatmap_color(f1):
    """Map an F1 score (0-1) to a heatmap background color.

    Scale: red (0.6) -> yellow (0.725) -> green (0.85).
    Tightened to the typical co-training F1 range for better contrast.
    Returns CSS rgb() string with transparency for readability.
    """
    f1 = max(0.6, min(0.85, f1))  # clamp
    if f1 < 0.725:
        # red -> yellow
        t = (f1 - 0.6) / 0.125
        r, g, b = int(255 - t * 30), int(80 + t * 150), int(80 - t * 20)
    else:
        # yellow -> green
        t = (f1 - 0.725) / 0.125
        r, g, b = int(225 - t * 165), int(230 - t * 50), int(60 + t * 15)
    return f"rgba({r},{g},{b},0.35)"


def _f1_to_heatmap_inverted(val, lo=10.0, hi=30.0):
    """Map a lower-is-better metric to heatmap color (inverted: low=green, high=red).

    Args:
        val: The metric value (e.g., error rate % or ECE).
        lo: Value at the green end.
        hi: Value at the red end.
    """
    t = max(0.0, min(1.0, (val - lo) / (hi - lo)))  # 0=green (low), 1=red (high)
    # Reverse: map t=0 to green, t=1 to red
    return _f1_to_heatmap_color(0.85 - t * 0.25)  # maps to 0.85 (green) -> 0.6 (red)


def _hm(val, inverted=False, lo=None, hi=None):
    """Return heatmap style attribute for a cell value."""
    if inverted:
        lo = lo if lo is not None else 10.0
        hi = hi if hi is not None else 30.0
        color = _f1_to_heatmap_inverted(val, lo, hi)
    else:
        color = _f1_to_heatmap_color(val)
    return f'style="background:{color}"'


def _render_comparison_tab(method_hierarchy, zs_results_by_task):
    """Render cross-model comparison tables with heatmap coloring."""
    parts = []
    models = sorted(method_hierarchy.keys())

    if not models:
        return ""

    # Pick latest run per model, flatten metrics
    all_metrics_by_model = {}
    for source in models:
        runs = method_hierarchy[source]
        latest_run = max(sorted(runs.keys()))
        all_metrics_by_model[source] = runs[latest_run]

    modality_order = ["text_only", "image_only", "text_image"]

    for task in TASKS:
        task_icon = TASK_ICONS.get(task, "")
        task_has_data = False

        for modality in modality_order:
            mod_icon = MODALITY_ICONS.get(modality, "")

            # Build data: budget -> model -> {avg_f1, std_f1, n_seeds}
            budget_data = {}
            for source in models:
                mod_metrics = [m for m in all_metrics_by_model[source]
                               if m.get("task") == task and m.get("modality") == modality]
                if not mod_metrics:
                    continue

                by_budget = {}
                for m in mod_metrics:
                    by_budget.setdefault(m.get("budget", 0), []).append(m)

                for budget, runs in by_budget.items():
                    if budget not in budget_data:
                        budget_data[budget] = {}
                    n = len(runs)
                    avg_f1 = statistics.mean(m["test_macro_f1"] for m in runs)
                    std_f1 = statistics.stdev(m["test_macro_f1"] for m in runs) if n > 1 else 0
                    wf1_vals = [m.get("test_weighted_f1") for m in runs if m.get("test_weighted_f1") is not None]
                    avg_wf1 = statistics.mean(wf1_vals) if wf1_vals else None
                    std_wf1 = statistics.stdev(wf1_vals) if len(wf1_vals) > 1 else 0
                    budget_data[budget][source] = {"avg_f1": avg_f1, "std_f1": std_f1,
                                                   "avg_wf1": avg_wf1, "std_wf1": std_wf1, "n_seeds": n}

            if not budget_data:
                continue

            if not task_has_data:
                parts.append(
                    f'<h2 style="font-size:20px; font-weight:700; margin:28px 0 12px; '
                    f'padding-bottom:8px; border-bottom:2px solid var(--accent); '
                    f'color:var(--accent);">{task_icon} {task.title()}</h2>'
                )
                task_has_data = True

            # Render table
            parts.append(f'<div class="section">')
            parts.append(f'<div class="section-header"><h3>{mod_icon} {modality}</h3></div>')

            # Header — grouped by metric, sub-columns are models
            n_models = len(models)
            parts.append('<table><thead>')
            parts.append(f'<tr><th rowspan="2" style="border-right:3px solid var(--border);">Labels/class</th>')
            parts.append(f'<th class="num" colspan="{n_models}" style="text-align:center; border-bottom:none; border-right:3px solid var(--border);">Macro F1</th>')
            parts.append(f'<th class="num" colspan="{n_models}" style="text-align:center; border-bottom:none;">Weighted F1</th>')
            parts.append('</tr><tr style="border-top:3px solid var(--border);">')
            for i, source in enumerate(models):
                border = ' border-right:3px solid var(--border);' if i == n_models - 1 else ''
                parts.append(f'<th class="num" style="{border}">{source}</th>')
            for source in models:
                parts.append(f'<th class="num">{source}</th>')
            parts.append('</tr></thead><tbody>')

            # Zero-shot row — Macro F1 group first, then W. F1 group
            zs_task = zs_results_by_task.get(task, [])
            zs_by_model = {}
            for source in models:
                zs_match = [m for m in zs_task
                            if m.get("model_slug") == source
                            and m.get("modality") == modality
                            and m.get("split") == "test"]
                zs_by_model[source] = zs_match[0] if zs_match else None

            parts.append('<tr style="background:var(--accent-bg); font-style:italic;">')
            parts.append('<td style="border-right:3px solid var(--border);">Zero-Shot</td>')
            # Macro F1 columns
            for i, source in enumerate(models):
                last = i == n_models - 1
                m = zs_by_model[source]
                if m:
                    zs_f1 = m.get("macro_f1", 0)
                    hm_color = _f1_to_heatmap_color(zs_f1)
                    border_css = f"background:{hm_color}; border-right:3px solid var(--border);" if last else f"background:{hm_color};"
                    parts.append(f'<td class="num" style="{border_css}">{zs_f1:.4f}</td>')
                else:
                    border_css = "border-right:3px solid var(--border);" if last else ""
                    parts.append(f'<td class="num" style="{border_css}">-</td>')
            # W. F1 columns
            for source in models:
                m = zs_by_model[source]
                if m and m.get("weighted_f1") is not None:
                    zs_wf1 = m["weighted_f1"]
                    parts.append(f'<td class="num" {_hm(zs_wf1)}>{zs_wf1:.4f}</td>')
                else:
                    parts.append('<td class="num">-</td>')
            parts.append('</tr>')

            # Budget rows — Macro F1 group first, then W. F1 group
            for budget in sorted(budget_data.keys()):
                row_data = budget_data[budget]
                # Find best Macro F1 and best W. F1 in this row
                row_f1s = {s: d["avg_f1"] for s, d in row_data.items()}
                best_f1 = max(row_f1s.values()) if row_f1s else 0
                row_wf1s = {s: d["avg_wf1"] for s, d in row_data.items() if d.get("avg_wf1") is not None}
                best_wf1 = max(row_wf1s.values()) if row_wf1s else 0

                parts.append('<tr>')
                parts.append(f'<td class="num" style="border-right:3px solid var(--border);">LG-CoTrain ({budget})</td>')
                # Macro F1 columns
                for i, source in enumerate(models):
                    border = ' border-right:3px solid var(--border);' if i == n_models - 1 else ''
                    if source in row_data:
                        d = row_data[source]
                        f1_text = f"{d['avg_f1']:.4f}"
                        if d["n_seeds"] > 1:
                            f1_text += f" &plusmn; {d['std_f1']:.4f}"
                        if d["avg_f1"] >= best_f1 - 1e-6:
                            f1_text = f"<strong>{f1_text}</strong>"
                        hm_style = _hm(d["avg_f1"]).replace('style="', f'style="{border} ')
                        parts.append(f'<td class="num" {hm_style}>{f1_text}</td>')
                    else:
                        parts.append(f'<td class="num" style="{border}">-</td>')
                # W. F1 columns
                for source in models:
                    if source in row_data and row_data[source].get("avg_wf1") is not None:
                        d = row_data[source]
                        wf1_text = f"{d['avg_wf1']:.4f}"
                        if d["n_seeds"] > 1 and d["std_wf1"] > 0:
                            wf1_text += f" &plusmn; {d['std_wf1']:.4f}"
                        if d["avg_wf1"] >= best_wf1 - 1e-6:
                            wf1_text = f"<strong>{wf1_text}</strong>"
                        parts.append(f'<td class="num" {_hm(d["avg_wf1"])}>{wf1_text}</td>')
                    else:
                        parts.append('<td class="num">-</td>')
                parts.append('</tr>')

            parts.append('</tbody></table></div>')

    return "\n".join(parts)


def _render_results_tab(metrics, zeroshot=None, results_root=None):
    """Render co-training tab with nested subtabs: method -> model -> run_id -> task."""
    if not metrics:
        return """
        <div class="empty-state">
            <div class="icon">&#x1F4CA;</div>
            <h3>No Experiment Results Yet</h3>
            <p>Run experiments to populate this tab:<br>
            <code>python -m lg_cotrain.run_experiment --task humanitarian --modality text_only --budget 5 --seed-set 1</code></p>
        </div>"""

    # Build hierarchy: method -> pseudo_source -> run_id -> list of metrics
    from collections import defaultdict
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for m in metrics:
        method = m.get("method", "lg-cotrain")
        source = m.get("pseudo_source", "unknown")
        run_id = m.get("run_id", "default")
        hierarchy[method][source][run_id].append(m)

    parts = []

    # Level 1: Method subtabs
    method_buttons = []
    method_contents = []
    first_method = True

    for method in sorted(hierarchy.keys()):
        method_id = f"ct-method-{method}"
        active = " active" if first_method else ""
        method_n = sum(len(v) for src in hierarchy[method].values() for v in src.values())
        method_buttons.append(
            f'<button class="{active.strip()}" onclick="showSubTab(\'tab-results\',\'{method_id}\',this)">'
            f'{method} ({method_n})</button>'
        )

        method_parts = []

        # Level 2: Model (pseudo_source) subtabs
        model_buttons = []
        model_contents = []
        first_model = True

        # Insert comparison overview tab as first model-level tab
        if len(hierarchy[method]) > 1:
            overview_id = f"ct-{method}-overview".replace(".", "-")
            model_buttons.append(
                f'<button class="active" onclick="showSubTab(\'{method_id}\',\'{overview_id}\',this)">'
                f'&#x1F4CA; Comparison</button>'
            )
            overview_html = _render_comparison_tab(hierarchy[method], zeroshot or {})
            model_contents.append(
                f'<div id="{overview_id}" class="sub-tab-content active">\n'
                + overview_html + '\n</div>'
            )
            first_model = False

        for source in sorted(hierarchy[method].keys()):
            model_id = f"ct-{method}-{source}".replace(".", "-")
            active_m = " active" if first_model else ""
            model_n = sum(len(v) for v in hierarchy[method][source].values())
            model_buttons.append(
                f'<button class="{active_m.strip()}" onclick="showSubTab(\'{method_id}\',\'{model_id}\',this)">'
                f'{source} ({model_n})</button>'
            )

            model_parts = []

            # Level 3: Run subtabs
            runs = hierarchy[method][source]
            run_buttons = []
            run_contents = []
            first_run = True

            for run_id in sorted(runs.keys()):
                run_metrics = runs[run_id]
                run_tab_id = f"{model_id}-{run_id}".replace(".", "-")
                active_r = " active" if first_run else ""
                run_buttons.append(
                    f'<button class="{active_r.strip()}" onclick="showSubTab(\'{model_id}\',\'{run_tab_id}\',this)">'
                    f'{run_id} ({len(run_metrics)})</button>'
                )

                # Render run content: notes + summary cards + per-task tables
                run_parts = []

                # Look for run_notes.txt
                if results_root:
                    notes_path = (
                        Path(results_root) / "cotrain" / method / source / run_id / "run_notes.tsv"
                    )
                    if notes_path.exists():
                        import csv as _csv
                        with open(notes_path, encoding="utf-8") as _f:
                            reader = _csv.DictReader(_f, delimiter="\t")
                            notes_rows = list(reader)

                        title = ""
                        items = []
                        for row in notes_rows:
                            if row.get("key", "").lower() == "title":
                                title = row.get("value", "")
                            else:
                                items.append((row.get("key", ""), row.get("value", "")))

                        if title or items:
                            run_parts.append(f'<div class="run-notes">')
                            if title:
                                run_parts.append(f'<strong>{title}</strong>')
                            if items:
                                run_parts.append('<table><tbody>')
                                for i in range(0, len(items), 2):
                                    run_parts.append('<tr>')
                                    for j in range(2):
                                        if i + j < len(items):
                                            k, v = items[i + j]
                                            run_parts.append(f'<td>{k}:</td><td>{v}</td>')
                                        else:
                                            run_parts.append('<td></td><td></td>')
                                    run_parts.append('</tr>')
                                run_parts.append('</tbody></table>')
                            run_parts.append('</div>')

                run_parts.append(_render_cotrain_summary_cards(run_metrics))

                for task in TASKS:
                    task_metrics = [m for m in run_metrics if m.get("task") == task]
                    if not task_metrics:
                        continue
                    icon = TASK_ICONS.get(task, "")
                    run_parts.append(
                        f'<h2 style="font-size:20px; font-weight:700; margin:28px 0 12px; '
                        f'padding-bottom:8px; border-bottom:2px solid var(--accent); '
                        f'color:var(--accent);">{icon} {task.title()}</h2>'
                    )
                    zs_for_task = (zeroshot or {}).get(task, [])
                    run_parts.append(_render_cotrain_task_tables(task, task_metrics, zs_for_task, pseudo_source=source))

                run_contents.append(
                    f'<div id="{run_tab_id}" class="sub-tab-content{active_r}">\n'
                    + "\n".join(run_parts)
                    + '\n</div>'
                )
                first_run = False

            # Assemble model content
            if len(runs) == 1:
                # Single run: skip run subtabs, show content directly
                model_parts.append(run_contents[0].replace(f'class="sub-tab-content"', 'class="sub-tab-content active"'))
            else:
                run_bar = '<nav class="sub-tab-bar">' + "".join(run_buttons) + '</nav>'
                model_parts.append(run_bar)
                model_parts.extend(run_contents)

            model_contents.append(
                f'<div id="{model_id}" class="sub-tab-content{active_m}">\n'
                + "\n".join(model_parts)
                + '\n</div>'
            )
            first_model = False

        # Assemble method content — always show model subtabs
        model_bar = '<nav class="sub-tab-bar">' + "".join(model_buttons) + '</nav>'
        method_parts.append(model_bar)
        method_parts.extend(model_contents)

        method_contents.append(
            f'<div id="{method_id}" class="sub-tab-content{active}">\n'
            + "\n".join(method_parts)
            + '\n</div>'
        )
        first_method = False

    # Assemble top level
    if len(hierarchy) == 1:
        # Single method: skip method subtabs
        parts.extend(method_contents)
    else:
        method_bar = '<nav class="sub-tab-bar">' + "".join(method_buttons) + '</nav>'
        parts.append(method_bar)
        parts.extend(method_contents)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main HTML generation
# ---------------------------------------------------------------------------

def generate_html(data_root: str, results_root: str) -> str:
    ds_stats = collect_dataset_stats(data_root)
    event_stats = collect_event_stats(data_root)
    metrics = collect_all_metrics(results_root)
    zeroshot = collect_zeroshot_results(results_root)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_zs = sum(len(v) for v in zeroshot.values())
    total_exp = len(metrics) + total_zs

    dataset_html = _render_dataset_overview(ds_stats, event_stats)
    results_html = _render_results_tab(metrics, zeroshot, results_root=results_root)

    # Build single Zero-Shot tab with sub-tabs for each task
    zs_tab_button = ""
    zs_tab_content = ""
    if total_zs > 0:
        zs_badge = f' <span style="background:var(--accent);color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px">{total_zs}</span>'
        zs_tab_button = f'    <button onclick="showTab(\'tab-zeroshot\', this)">&#x1F50E; Zero-Shot{zs_badge}</button>\n'

        # Build sub-tab bar and content
        sub_buttons = []
        sub_contents = []
        first = True
        for task in TASKS:
            task_results = zeroshot.get(task, [])
            if not task_results:
                continue
            sub_id = f"sub-zs-{task}"
            icon = TASK_ICONS.get(task, "")
            active = " active" if first else ""
            sub_buttons.append(
                f'<button class="{active.strip()}" onclick="showSubTab(\'tab-zeroshot\',\'{sub_id}\',this)">'
                f'{icon} {task.title()}</button>'
            )
            sub_contents.append(
                f'<div id="{sub_id}" class="sub-tab-content{active}">\n'
                f'{_render_zeroshot_tab(task, task_results)}\n</div>'
            )
            first = False

        sub_bar = '<nav class="sub-tab-bar">' + "".join(sub_buttons) + '</nav>'
        zs_tab_content = (
            f'\n<div id="tab-zeroshot" class="tab-content">\n'
            f'{sub_bar}\n'
            + "\n".join(sub_contents)
            + '\n</div>\n'
        )

    cotrain_badge = f' <span style="background:var(--accent);color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px">{len(metrics)}</span>' if metrics else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG-CoTrain &mdash; CrisisMMD Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<header>
    <h1>LG-CoTrain &mdash; CrisisMMD</h1>
    <p class="subtitle">
        Generated <span>{now}</span> &bull;
        <span>{len(TASKS)} tasks</span> &bull;
        <span>{len(MODALITIES)} modalities</span> &bull;
        <span>{total_exp} experiments</span>
    </p>
</header>

<nav class="tab-bar">
    <button class="active" onclick="showTab('tab-data', this)">&#x1F50D; Dataset Exploration</button>
{zs_tab_button}    <button onclick="showTab('tab-results', this)">&#x1F504; Co-Training Results{cotrain_badge}</button>
</nav>

<div id="tab-data" class="tab-content active">
{dataset_html}
</div>
{zs_tab_content}
<div id="tab-results" class="tab-content">
{results_html}
</div>

<script>{_JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_repo_root():
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / "lg_cotrain").is_dir():
            return p
    return cwd


def main():
    parser = argparse.ArgumentParser(description="Generate CrisisMMD experiment dashboard")
    parser.add_argument("--results-root", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
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
    print(f"  Dataset: CrisisMMD ({len(TASKS)} tasks, {len(MODALITIES)} modalities)")
    print(f"  Experiments: {len(metrics)} results found")


if __name__ == "__main__":
    main()
