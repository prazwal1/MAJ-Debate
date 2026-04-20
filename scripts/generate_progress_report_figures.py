#!/usr/bin/env python3
"""
Generate the missing progress-report figures from saved experiment outputs.

Outputs:
  - progress_report/fig_ablation_acc.png|pdf
  - progress_report/fig_stage3_verdicts.png|pdf
  - progress_report/fig_stage2_labels.png|pdf
  - progress_report/fig_persuasion_correctness.png|pdf

Usage:
  .venv\\Scripts\\python.exe scripts\\generate_progress_report_figures.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "outputs").exists() and (p / "progress_report").exists():
            return p
    return cwd


ROOT = find_project_root()
PROGRESS_DIR = ROOT / "progress_report"

DDO_ABLATION = ROOT / "outputs" / "ablations" / "ddo_sample" / "ablation_table.json"
LOGIC_ABLATION = ROOT / "outputs" / "ablations" / "logic_test" / "ablation_table.json"
DDO_STAGE2_ZERO = ROOT / "outputs" / "stage2" / "ddo_sample" / "stage2_relations.json"
DDO_STAGE2_TARGETED = ROOT / "outputs" / "stage2" / "ddo_sample_targeted_attacks" / "stage2_relations.json"
DDO_STAGE3 = ROOT / "outputs" / "stage3" / "ddo_sample" / "stage3_graphs.json"
LOGIC_STAGE3 = ROOT / "outputs" / "stage3" / "logic_test" / "stage3_graphs.json"


COLORS = {
    "persuasion": "#355070",
    "correctness": "#6D597A",
    "Attack": "#C0392B",
    "Support": "#2A9D8F",
    "Neutral": "#E9C46A",
    "None": "#9AA0A6",
    "PRO": "#3A86FF",
    "CON": "#FF006E",
    "TIE": "#ADB5BD",
    "gt": "#264653",
    "empty": "#F4A261",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_output_dir() -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


def ci95_half_width(acc_pct: float, n: int) -> float:
    p = acc_pct / 100.0
    if not n:
        return 0.0
    sigma = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return 100.0 * 1.96 * sigma


def load_ablation_rows(path: Path) -> list[dict]:
    return read_json(path)["rows"]


def stage3_metrics(path: Path) -> dict:
    graphs = read_json(path).get("graphs", [])
    n = len(graphs)
    verdicts = Counter(g.get("graph_verdict", {}).get("winner", "TIE") for g in graphs)
    labels = Counter(g.get("benchmark_label", "TIE") for g in graphs)
    empty_grounded = sum(1 for g in graphs if not g.get("grounded_extension"))
    return {
        "n": n,
        "pred_pct": {k: 100.0 * verdicts.get(k, 0) / n for k in ["PRO", "CON", "TIE"]},
        "gold_pct": {k: 100.0 * labels.get(k, 0) / n for k in ["PRO", "CON", "TIE"]},
        "empty_grounded_pct": 100.0 * empty_grounded / n if n else 0.0,
    }


def stage2_label_distribution(path: Path) -> dict:
    summary = read_json(path).get("summary", {})
    counts = summary.get("label_counts", {})
    total = summary.get("total_relations", sum(counts.values()))
    kept = summary.get("total_kept_relations")
    pct = {k: 100.0 * counts.get(k, 0) / total for k in ["Attack", "Support", "Neutral", "None"]}
    return {
        "total": total,
        "kept": kept,
        "counts": counts,
        "pct": pct,
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = PROGRESS_DIR / f"{stem}.png"
    pdf_path = PROGRESS_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def make_ablation_accuracy_figure() -> None:
    ddo_rows = load_ablation_rows(DDO_ABLATION)
    logic_rows = load_ablation_rows(LOGIC_ABLATION)
    order = [r["config_name"] for r in ddo_rows]
    label_map = {r["config_name"]: r["label"] for r in ddo_rows}
    short_labels = {
        "single_llm": "Single",
        "cot": "CoT",
        "direct_judge": "Direct",
        "two_agents": "2 Agents",
        "six_agents": "6 Agents",
        "targeted_attacks": "Targeted",
        "dung_no_agents": "Graph Only",
        "full": "Full",
    }
    ddo_map = {r["config_name"]: r for r in ddo_rows}
    logic_map = {r["config_name"]: r for r in logic_rows}

    x = np.arange(len(order))
    width = 0.38
    persuasion = [ddo_map[k]["acc_mean_pct"] for k in order]
    correctness = [logic_map[k]["acc_mean_pct"] for k in order]
    persuasion_ci = [ci95_half_width(ddo_map[k]["acc_mean_pct"], ddo_map[k]["acc_n"]) for k in order]
    correctness_ci = [ci95_half_width(logic_map[k]["acc_mean_pct"], logic_map[k]["acc_n"]) for k in order]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(
        x - width / 2,
        persuasion,
        width,
        yerr=persuasion_ci,
        capsize=4,
        label="Persuasion (DDO)",
        color=COLORS["persuasion"],
        alpha=0.92,
    )
    ax.bar(
        x + width / 2,
        correctness,
        width,
        yerr=correctness_ci,
        capsize=4,
        label="Correctness (Logic test)",
        color=COLORS["correctness"],
        alpha=0.92,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([short_labels.get(k, label_map[k]) for k in order], rotation=20, ha="right")
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Agreement Across Eight Configurations")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    best_idx = int(np.argmax(persuasion))
    ax.annotate(
        "Best persuasion",
        xy=(x[best_idx] - width / 2, persuasion[best_idx]),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        fontsize=9,
    )

    save_figure(fig, "fig_ablation_acc")


def make_stage3_verdict_figure() -> None:
    ddo = stage3_metrics(DDO_STAGE3)
    logic = stage3_metrics(LOGIC_STAGE3)
    datasets = [("DDO persuasion", ddo), ("Logic correctness", logic)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.7, 1.0]})

    y = np.arange(len(datasets))
    left = np.zeros(len(datasets))
    for verdict in ["PRO", "CON", "TIE"]:
        vals = [d["pred_pct"][verdict] for _, d in datasets]
        ax1.barh(y, vals, left=left, color=COLORS[verdict], label=verdict)
        left += np.array(vals)

    gt_pro = [d["gold_pct"]["PRO"] for _, d in datasets]
    ax1.scatter(gt_pro, y, color=COLORS["gt"], marker="D", s=48, label="Ground-truth PRO share", zorder=5)
    ax1.set_yticks(y)
    ax1.set_yticklabels([name for name, _ in datasets])
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Verdict distribution (%)")
    ax1.set_title("Stage 3 verdict mix")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)
    ax1.legend(frameon=False, loc="lower right")
    ax1.invert_yaxis()

    pred_pro = [d["pred_pct"]["PRO"] for _, d in datasets]
    empty = [d["empty_grounded_pct"] for _, d in datasets]
    x = np.arange(len(datasets))
    width = 0.36
    ax2.bar(x - width / 2, pred_pro, width, color=COLORS["PRO"], label="Predicted PRO share")
    ax2.bar(x + width / 2, empty, width, color=COLORS["empty"], label="Empty grounded %")
    for i, g in enumerate(gt_pro):
        ax2.axhline(g, color=COLORS["gt"], linestyle="--", linewidth=1.1, alpha=0.8 if i == 0 else 0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["DDO", "Logic"])
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Percentage")
    ax2.set_title("Bias and graph sparsity")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.legend(frameon=False, fontsize=9)

    fig.suptitle("Stage 3 Verdict Distributions and Empty-Grounded Frequency", y=1.02)
    save_figure(fig, "fig_stage3_verdicts")


def make_stage2_labels_figure() -> None:
    zero = stage2_label_distribution(DDO_STAGE2_ZERO)
    targeted = stage2_label_distribution(DDO_STAGE2_TARGETED)
    configs = [("Zero-shot", zero), ("Targeted", targeted)]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    x = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    for label in ["Attack", "Support", "Neutral", "None"]:
        vals = [cfg["pct"][label] for _, cfg in configs]
        ax.bar(x, vals, bottom=bottom, width=0.62, color=COLORS[label], label=label)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{name}\nkept={cfg['kept']:,}\nattack={cfg['counts'].get('Attack', 0):,}"
            for name, cfg in configs
        ]
    )
    ax.set_ylabel("Share of all candidate relations (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Stage 2 Relation-Label Distribution on DDO")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    for idx, (_, cfg) in enumerate(configs):
        ax.text(
            idx,
            102,
            f"total={cfg['total']:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(fig, "fig_stage2_labels")


def make_persuasion_correctness_figure() -> None:
    ddo_rows = load_ablation_rows(DDO_ABLATION)
    logic_rows = load_ablation_rows(LOGIC_ABLATION)
    logic_map = {r["config_name"]: r for r in logic_rows}
    short_labels = {
        "single_llm": "Single",
        "cot": "CoT",
        "direct_judge": "Direct",
        "two_agents": "2 Agents",
        "six_agents": "6 Agents",
        "targeted_attacks": "Targeted",
        "dung_no_agents": "Graph Only",
        "full": "Full",
    }

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    x_positions = [0, 1]
    cmap = plt.get_cmap("tab10")

    for idx, row in enumerate(ddo_rows):
        logic_row = logic_map[row["config_name"]]
        y0 = row["acc_mean_pct"]
        y1 = logic_row["acc_mean_pct"]
        ax.plot(x_positions, [y0, y1], marker="o", linewidth=2.2, color=cmap(idx), alpha=0.95)
        ax.text(-0.03, y0, short_labels.get(row["config_name"], row["label"]), ha="right", va="center", fontsize=9)
        ax.text(1.03, y1, f"{y1:.1f}", ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Persuasion (DDO)", "Correctness (Logic)"])
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(30, 85)
    ax.set_title("Every Configuration Scores Higher on Correctness Than Persuasion")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    save_figure(fig, "fig_persuasion_correctness")


def main() -> None:
    ensure_output_dir()
    make_ablation_accuracy_figure()
    make_stage3_verdict_figure()
    make_stage2_labels_figure()
    make_persuasion_correctness_figure()
    print("Wrote:")
    for stem in [
        "fig_ablation_acc",
        "fig_stage3_verdicts",
        "fig_stage2_labels",
        "fig_persuasion_correctness",
    ]:
        print(f"  {PROGRESS_DIR / (stem + '.png')}")
        print(f"  {PROGRESS_DIR / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
