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

TASK_ICONS = {"informative": "&#x1F4CB;", "humanitarian": "&#x1F6D1;", "damage": "&#x1F3DA;"}
MODALITY_ICONS = {"text_only": "&#x1F4DD;", "image_only": "&#x1F5BC;", "text_image": "&#x1F4F7;"}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_dataset_stats(data_root: str) -> dict:
    """Scan preprocessed CrisisMMD data and collect class distributions."""
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
            for budget in BUDGETS:
                for prefix in ["labeled", "unlabeled"]:
                    key = f"{prefix}_{budget}"
                    tsv = mod_dir / f"{prefix}_{budget}_set1.tsv"
                    if tsv.exists():
                        stats[task][modality][key] = _count_labels(tsv)
    return stats


def collect_event_stats(data_root: str) -> dict:
    """Count samples per event from the original source files."""
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

function toggleCollapse(id) {
    const body = document.getElementById(id);
    const toggle = body.previousElementSibling;
    body.classList.toggle('open');
    toggle.classList.toggle('open');
}
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
            <div class="value">{n_events}</div>
            <div class="label">Disaster Events</div>
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
                train_count = mod_stats.get("train", {}).get(label, 0)
                total_train = sum(mod_stats.get("train", {}).values()) or 1
                share = train_count / total_train * 100
                if share < 1:
                    warn = ' <span style="color:var(--danger);font-size:10px" title="Rare class">&#x26A0;</span>'
                else:
                    warn = ""
                parts.append(f'<tr><td><code>{label}</code>{warn}</td>')
                for c in all_cols:
                    count = mod_stats.get(c, {}).get(label, 0)
                    totals[c] += count
                    parts.append(f'<td class="{_hm_class(count, max_count)}">{_fmt(count)}</td>')
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
            <strong>Label source notes:</strong>
            For informative and humanitarian tasks, <code>text_only</code> and <code>text_image</code>
            use <code>label_text</code> (text annotation), while <code>image_only</code> uses
            <code>label_image</code> (image annotation). For the damage task, all modalities use the
            single <code>label</code> column.
            Budget columns (L5, U5, etc.) show seed=1 splits. <strong>L</strong> = labeled,
            <strong>U</strong> = unlabeled. &#x26A0; marks classes with &lt;1% of training data.
        </div>
    </div>
    """)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Results Tab
# ---------------------------------------------------------------------------

def _render_results_tab(metrics):
    if not metrics:
        return """
        <div class="empty-state">
            <div class="icon">&#x1F4CA;</div>
            <h3>No Experiment Results Yet</h3>
            <p>Run experiments to populate this tab:<br>
            <code>python -m lg_cotrain.run_experiment --task humanitarian --modality text_only --budget 5 --seed-set 1</code></p>
        </div>"""

    parts = []
    n = len(metrics)
    f1s = [m["test_macro_f1"] for m in metrics if "test_macro_f1" in m]
    errs = [m["test_error_rate"] for m in metrics if "test_error_rate" in m]
    eces = [m.get("test_ece", 0) for m in metrics if "test_ece" in m]

    parts.append(f"""
    <div class="cards">
        <div class="card"><div class="icon">&#x1F9EA;</div><div class="value">{n}</div><div class="label">Experiments</div></div>
        <div class="card"><div class="icon">&#x1F3AF;</div><div class="value">{statistics.mean(f1s):.4f}</div><div class="label">Avg Macro-F1</div></div>
        <div class="card"><div class="icon">&#x274C;</div><div class="value">{statistics.mean(errs):.1f}%</div><div class="label">Avg Error Rate</div></div>
        <div class="card"><div class="icon">&#x1F4CF;</div><div class="value">{statistics.mean(eces):.4f}</div><div class="label">Avg ECE</div></div>
    </div>
    """)

    parts.append('<div class="section">')
    parts.append('<div class="section-header"><h2>&#x1F4CB; All Experiment Results</h2></div>')
    parts.append("""<table><thead>
        <tr><th>Task</th><th>Modality</th><th class="num">Budget</th><th class="num">Seed</th>
        <th class="num">Test F1</th><th class="num">Test Err%</th><th class="num">Test ECE</th>
        <th class="num">Dev F1</th><th>Strategy</th></tr></thead><tbody>""")

    for m in sorted(metrics, key=lambda x: (x.get("task",""), x.get("modality",""), x.get("budget",0), x.get("seed_set",0))):
        f1 = m.get("test_macro_f1", 0)
        if f1 >= 0.7:
            badge = "badge-success"
        elif f1 >= 0.4:
            badge = "badge-warning"
        else:
            badge = "badge-danger"
        parts.append(
            f'<tr><td>{m.get("task","?")}</td><td>{m.get("modality","?")}</td>'
            f'<td class="num">{m.get("budget","?")}</td><td class="num">{m.get("seed_set","?")}</td>'
            f'<td class="num"><span class="badge {badge}">{f1:.4f}</span></td>'
            f'<td class="num">{m.get("test_error_rate",0):.2f}%</td>'
            f'<td class="num">{m.get("test_ece",0):.4f}</td>'
            f'<td class="num">{m.get("dev_macro_f1",0):.4f}</td>'
            f'<td>{m.get("stopping_strategy","?")}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main HTML generation
# ---------------------------------------------------------------------------

def generate_html(data_root: str, results_root: str) -> str:
    ds_stats = collect_dataset_stats(data_root)
    event_stats = collect_event_stats(data_root)
    metrics = collect_all_metrics(results_root)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_html = _render_dataset_overview(ds_stats, event_stats)
    results_html = _render_results_tab(metrics)
    results_badge = f' <span style="background:var(--accent);color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px">{len(metrics)}</span>' if metrics else ''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LG-CoTrain &mdash; CrisisMMD v2.0 Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<header>
    <h1>LG-CoTrain &mdash; CrisisMMD v2.0</h1>
    <p class="subtitle">
        Generated <span>{now}</span> &bull;
        <span>{len(TASKS)} tasks</span> &bull;
        <span>{len(MODALITIES)} modalities</span> &bull;
        <span>{len(metrics)} experiments</span>
    </p>
</header>

<nav class="tab-bar">
    <button class="active" onclick="showTab('tab-data', this)">&#x1F50D; Dataset Exploration</button>
    <button onclick="showTab('tab-results', this)">&#x1F4CA; Experiment Results{results_badge}</button>
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
    print(f"  Dataset: CrisisMMD v2.0 ({len(TASKS)} tasks, {len(MODALITIES)} modalities)")
    print(f"  Experiments: {len(metrics)} results found")


if __name__ == "__main__":
    main()
