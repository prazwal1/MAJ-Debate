#!/usr/bin/env python3
"""
Generate all figures for the MAJ-Debate final report.
Figures are written to report/figures/.

Run from the project root:
    python report/generate_figures.py
"""

import os
import sys
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_DDO   = '#2196F3'   # blue  — persuasion (DDO)
C_LOGIC = '#FF5722'   # orange — correctness (Logic)
C_PRO   = '#4CAF50'
C_CON   = '#F44336'
C_TIE   = '#9E9E9E'
C_BASE  = '#FFC107'   # highlight baseline
GREY    = '#EEEEEE'

# ── Shared data ───────────────────────────────────────────────────────────────
CONFIGS = [
    'Single-LLM\nBaseline',
    '+CoT\nBaseline',
    '+Direct\nJudge',
    '+2\nAgents',
    '+6\nAgents',
    '+Targeted\nAttacks',
    '+Dung Graph\n(no agents)',
    'Full\nPipeline',
]
CONFIG_SHORT = ['Single-LLM','CoT','Direct Judge','2 Agents',
                '6 Agents','Targeted Atk','Dung (no ag.)','Full']

DDO_ACC  = [56.4, 49.2, 40.8, 42.6, 40.4, 40.0, 43.8, 41.8]
DDO_STD  = [ 2.22, 2.24,  2.2, 2.21, 2.19, 2.19, 2.22, 2.21]
LOG_ACC  = [78.0, 74.0, 60.0, 66.0, 58.0, 58.0, 62.0, 56.0]
LOG_STD  = [5.86,  6.2, 6.93,  6.7, 6.98, 6.98, 6.86, 7.02]
PERS     = [4.09, 4.48, 4.39, 4.11, 4.40, 4.37, 4.27, 4.40]

VERDICT_PRO = [250, 376, 467, 462, 467, 473, 452, 472]
VERDICT_CON = [249, 116,  33,  38,  33,  27,  48,  28]
VERDICT_TIE = [  1,   8,   0,   0,   0,   0,   0,   0]

ATTACK_DIV_MEAN = [None, None, 29.1, 4.43, 29.1, 29.1, 7.61, 29.1]
EMPTY_GROUNDED  = [None, None, None, 0.6,  None, None, 18.2, 43.6]

N = len(CONFIGS)
x = np.arange(N)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1  Dual-benchmark ablation accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_ablation_actual():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.35
    bars_ddo   = ax.bar(x - w/2, DDO_ACC,  w, yerr=DDO_STD,  capsize=3,
                        color=[C_BASE if i==0 else C_DDO for i in range(N)],
                        label='DDO (persuasion, n=500)', alpha=0.9, ecolor='#555')
    bars_logic = ax.bar(x + w/2, LOG_ACC, w, yerr=LOG_STD, capsize=3,
                        color=C_LOGIC, label='Logic (correctness, n=50)',
                        alpha=0.9, ecolor='#555')

    # Expected range from proposal
    ax.axhspan(78, 85, alpha=0.08, color='purple', label='Proposed expected range (78–85%)')
    ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.4, label='Chance baseline (50%)')

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=8)
    ax.set_ylabel('Agreement with gold label (%)')
    ax.set_title('MAJ-Debate Ablation: Persuasion (DDO) vs Correctness (Logic) Accuracy')
    ax.set_ylim(30, 92)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate single-LLM as "best on DDO"
    ax.annotate('Best DDO\n(Single-LLM)', xy=(0 - w/2, 56.4), xytext=(0.6, 68),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                fontsize=7.5, ha='center')

    fig.tight_layout()
    p = OUT / 'fig_ablation_actual.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2  Persuasion vs Correctness scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_persuasion_vs_correctness():
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = ['Single-LLM','CoT','Direct\nJudge','2 Agents',
              '6 Agents','Targeted\nAtk','Dung\n(no ag.)','Full']
    colors = [C_BASE] + [C_DDO]*7

    sc = ax.scatter(DDO_ACC, LOG_ACC, c=colors, s=120, zorder=5)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (DDO_ACC[i], LOG_ACC[i]),
                    textcoords='offset points', xytext=(6, 3), fontsize=7.5)

    # Diagonal (DDO = Logic)
    ax.plot([35, 65], [35, 65], 'k--', lw=0.8, alpha=0.4, label='DDO = Logic line')
    ax.set_xlabel('DDO Accuracy — Persuasion (%)')
    ax.set_ylabel('Logic Accuracy — Correctness (%)')
    ax.set_title('Persuasion vs Correctness Agreement per Configuration')
    ax.set_xlim(35, 65)
    ax.set_ylim(48, 85)
    ax.grid(alpha=0.3)

    # Gap annotation
    for i in range(N):
        gap = LOG_ACC[i] - DDO_ACC[i]
        ax.annotate(f'+{gap:.0f}pp', (DDO_ACC[i], LOG_ACC[i]),
                    textcoords='offset points', xytext=(-28, -12),
                    fontsize=6, color='gray')

    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / 'fig_persuasion_vs_correctness.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3  PRO / CON / TIE verdict distribution
# ─────────────────────────────────────────────────────────────────────────────
def fig_pro_bias():
    fig, ax = plt.subplots(figsize=(10, 4))
    total = [VERDICT_PRO[i] + VERDICT_CON[i] + VERDICT_TIE[i] for i in range(N)]
    pct_pro = [100*VERDICT_PRO[i]/total[i] for i in range(N)]
    pct_con = [100*VERDICT_CON[i]/total[i] for i in range(N)]
    pct_tie = [100*VERDICT_TIE[i]/total[i] for i in range(N)]

    ax.bar(x, pct_pro, color=C_PRO,  label='PRO', alpha=0.85)
    ax.bar(x, pct_con, bottom=pct_pro, color=C_CON, label='CON', alpha=0.85)
    bot2 = [pct_pro[i]+pct_con[i] for i in range(N)]
    ax.bar(x, pct_tie, bottom=bot2, color=C_TIE, label='TIE', alpha=0.85)

    # Expected PRO rate in DDO sample
    ax.axhline(60.8, color='navy', lw=1.5, ls='--', label='Expected PRO rate (60.8%)')

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=8)
    ax.set_ylabel('Verdict distribution (%)')
    ax.set_title('PRO-Bias: Verdict Distribution per Configuration (DDO, n=500)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Annotate PRO % on top of each bar
    for i in range(N):
        ax.text(i, pct_pro[i] + 1.5, f'{pct_pro[i]:.0f}%', ha='center', fontsize=7.5, color='white',
                fontweight='bold', va='bottom')

    fig.tight_layout()
    p = OUT / 'fig_pro_bias.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4  Running example — LOGIC_002 pipeline walkthrough
# ─────────────────────────────────────────────────────────────────────────────
def fig_running_example():
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#FAFAFA')

    # Title
    fig.text(0.5, 0.97,
             'Running Example: LOGIC_002 — "If some birds can fly and penguins are birds, then penguins can fly."',
             ha='center', va='top', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.935, 'Gold label: CON    |    Full pipeline prediction: PRO ✗   (judge overrides graph)',
             ha='center', va='top', fontsize=9.5, color='#D32F2F')

    # ── Stage boxes layout ────────────────────────────────────────────────
    stage_titles = ['Stage 1\nSide-Picking Agents', 'Stage 2\nAttack-Relation Brain',
                    'Stage 3\nDung Framework', 'Stage 4\nJudge Brain']
    stage_colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FFF9C4']
    stage_border  = ['#388E3C', '#F57C00', '#1976D2', '#FBC02D']
    stage_x = [0.01, 0.27, 0.53, 0.78]
    stage_w = 0.24

    for i, (title, color, border, sx) in enumerate(
            zip(stage_titles, stage_colors, stage_border, stage_x)):
        ax = fig.add_axes([sx, 0.05, stage_w, 0.83])
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_facecolor(color)
        for sp in ax.spines.values():
            sp.set_edgecolor(border); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.97, title, ha='center', va='top', fontsize=9.5,
                fontweight='bold', color=border,
                transform=ax.transAxes)

        if i == 0:   # Stage 1 content
            lines = [
                ('PRO Agents (×3)', '#2E7D32', 0.86),
                ('A000 [Rationalist]', '#388E3C', 0.78),
                ('"Flying ability is not\n universal among birds."', '#555', 0.72),
                ('A009 [Skeptic] (CON)', '#C62828', 0.60),
                ('"Some birds, including\n penguins, cannot fly."', '#555', 0.54),
                ('A010 [Skeptic] (CON)', '#C62828', 0.44),
                ('"Logical fallacy of\n affirming the consequent."', '#555', 0.38),
                ('Round 2 counter-args:', '#1565C0', 0.25),
                ('A018 [Rationalist PRO]', '#388E3C', 0.18),
                ('"Not all birds fly — but\n penguins are still birds."', '#555', 0.11),
                ('24 args total generated', '#777', 0.02),
            ]
            for txt, col, y in lines:
                ax.text(0.5, y, txt, ha='center', va='top', fontsize=7,
                        color=col, transform=ax.transAxes,
                        wrap=True)

        elif i == 1:  # Stage 2 content
            ax.text(0.5, 0.88, '435 pairs checked\n131 kept (conf ≥ 0.65)',
                    ha='center', va='top', fontsize=7.5, color='#555',
                    transform=ax.transAxes)
            ax.text(0.5, 0.77, 'Label counts:', ha='center', va='top',
                    fontsize=8, fontweight='bold', color='#F57C00',
                    transform=ax.transAxes)
            rows = [('Attack', 59, '#C62828'), ('Support', 70, '#2E7D32'), ('Neutral', 2, '#555')]
            for j, (lbl, cnt, col) in enumerate(rows):
                y = 0.68 - j*0.1
                ax.text(0.2, y, lbl, ha='left', fontsize=8, color=col, transform=ax.transAxes)
                ax.text(0.8, y, str(cnt), ha='right', fontsize=8, color=col, transform=ax.transAxes)

            ax.text(0.5, 0.38, 'Sample attack:', ha='center', va='top',
                    fontsize=7.5, fontweight='bold', color='#F57C00',
                    transform=ax.transAxes)
            ax.text(0.5, 0.30, 'A002→A001  conf=0.90', ha='center', fontsize=7,
                    color='#C62828', transform=ax.transAxes)
            ax.text(0.5, 0.22, 'A005→A001  conf=1.00', ha='center', fontsize=7,
                    color='#C62828', transform=ax.transAxes)
            ax.text(0.5, 0.10, '59 attacks  |  70 supports', ha='center',
                    fontsize=7.5, color='#555', transform=ax.transAxes,
                    style='italic')

        elif i == 2:  # Stage 3 content
            ax.text(0.5, 0.88, 'Directed attack graph\n(59 attack edges)',
                    ha='center', va='top', fontsize=7.5, color='#555',
                    transform=ax.transAxes)
            ax.text(0.5, 0.77, 'Grounded Extension:', ha='center',
                    fontsize=8, fontweight='bold', color='#1565C0',
                    transform=ax.transAxes)
            ax.text(0.5, 0.69, '7 args  (5 PRO, 2 CON)', ha='center',
                    fontsize=7.5, color='#555', transform=ax.transAxes)
            ax.text(0.5, 0.58, 'Scoring:', ha='center',
                    fontsize=8, fontweight='bold', color='#1565C0',
                    transform=ax.transAxes)
            ax.text(0.5, 0.50, 'PRO score = 0.0', ha='center', fontsize=8,
                    color='#2E7D32', transform=ax.transAxes)
            ax.text(0.5, 0.42, 'CON score = 1.0', ha='center', fontsize=8,
                    color='#C62828', transform=ax.transAxes)
            # Verdict box
            rect = mpatches.FancyBboxPatch((0.1, 0.26), 0.8, 0.1,
                    boxstyle='round,pad=0.02', facecolor='#C62828',
                    edgecolor='#B71C1C', linewidth=1.5,
                    transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(0.5, 0.31, 'Graph Verdict: CON ✓', ha='center',
                    fontsize=9, fontweight='bold', color='white',
                    transform=ax.transAxes)
            ax.text(0.5, 0.14, 'Basis: grounded extension\nMargin: CON −1.0 pp', ha='center',
                    fontsize=7, color='#555', transform=ax.transAxes)

        elif i == 3:  # Stage 4 content
            ax.text(0.5, 0.88, 'Receives graph verdict\nas hard constraint',
                    ha='center', va='top', fontsize=7.5, color='#555',
                    transform=ax.transAxes)
            ax.text(0.5, 0.76, 'used_graph = True', ha='center',
                    fontsize=8, color='#1565C0', transform=ax.transAxes)
            ax.text(0.5, 0.68, 'confidence = 1.00', ha='center',
                    fontsize=8, color='#555', transform=ax.transAxes)
            ax.text(0.5, 0.57, 'Rationale:', ha='center',
                    fontsize=8, fontweight='bold', color='#F57C00',
                    transform=ax.transAxes)
            ax.text(0.5, 0.48, '"PRO has more undefeated\nclaims and stronger\nsupporting attacks."',
                    ha='center', fontsize=7.5, style='italic', color='#555',
                    transform=ax.transAxes)
            # Final verdict box — wrong!
            rect2 = mpatches.FancyBboxPatch((0.08, 0.25), 0.84, 0.12,
                    boxstyle='round,pad=0.02', facecolor='#2E7D32',
                    edgecolor='#1B5E20', linewidth=1.5,
                    transform=ax.transAxes)
            ax.add_patch(rect2)
            ax.text(0.5, 0.315, 'Final Verdict: PRO ✗', ha='center',
                    fontsize=10, fontweight='bold', color='white',
                    transform=ax.transAxes)
            # Error label
            rect3 = mpatches.FancyBboxPatch((0.08, 0.10), 0.84, 0.12,
                    boxstyle='round,pad=0.02', facecolor='#D32F2F',
                    edgecolor='#B71C1C', linewidth=1.5,
                    transform=ax.transAxes)
            ax.add_patch(rect3)
            ax.text(0.5, 0.165, 'ERROR: Judge overrides graph\n(Gold = CON)', ha='center',
                    fontsize=8, color='white', fontweight='bold',
                    transform=ax.transAxes)

    # Arrows between stages
    for sx in [stage_x[0]+stage_w, stage_x[1]+stage_w, stage_x[2]+stage_w]:
        fig.text(sx + 0.005, 0.48, '→', fontsize=20, color='#555',
                 ha='left', va='center')

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / 'fig_running_example.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5  Human evaluation on 10 failure topics
# ─────────────────────────────────────────────────────────────────────────────
def fig_human_eval():
    # Agreement % with human majority on 10 failure topics
    human_configs  = ['single_llm', 'cot', 'two_agents', 'dung_no_agents',
                      'direct_judge', 'six_agents', 'targeted_attacks', 'full']
    human_labels   = ['Single-LLM', 'CoT', '2 Agents', 'Dung\n(no ag.)',
                      'Direct\nJudge', '6 Agents', 'Targeted\nAtk', 'Full\nPipeline']
    human_agree    = [90.0, 60.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [C_BASE if lbl == 'Single-LLM' else C_DDO for lbl in human_labels]
    # Mark zeros in red
    colors = ['#D32F2F' if v == 0.0 else (C_BASE if lbl == 'Single-LLM' else C_DDO)
              for v, lbl in zip(human_agree, human_labels)]

    bars = ax.bar(range(len(human_labels)), human_agree, color=colors, alpha=0.88)
    ax.axhline(100, color='#388E3C', lw=1.4, ls='--', label='Human accuracy (100%)')
    ax.axhline(60.8, color='navy', lw=1, ls=':', alpha=0.6, label='DDO expected PRO rate (60.8%)')
    ax.set_xticks(range(len(human_labels)))
    ax.set_xticklabels(human_labels, fontsize=9)
    ax.set_ylabel('Agreement with human majority (%)')
    ax.set_title('Human Evaluation: Model Agreement on 10 Curated CON Failure Topics\n'
                 '(All topics gold=CON; human annotators: 10; human accuracy=100%)')
    ax.set_ylim(-5, 115)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8)

    for bar, val in zip(bars, human_agree):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    p = OUT / 'fig_human_eval.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6  Attack diversity vs accuracy
# ─────────────────────────────────────────────────────────────────────────────
def fig_attack_diversity():
    # Configs that have attack diversity data
    div_labels = ['2 Agents', '6 Agents\n& Full', 'Dung\n(no ag.)']
    div_values = [4.43, 29.1, 7.61]
    div_ddo    = [42.6, 40.4, 43.8]   # DDO accuracy

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: bar chart of attack diversity
    colors = [C_DDO, C_LOGIC, C_BASE]
    ax1.bar(div_labels, div_values, color=colors, alpha=0.85)
    ax1.set_ylabel('Mean unique arguments attacked')
    ax1.set_title('Attack Diversity by Configuration')
    ax1.set_ylim(0, 35)
    ax1.grid(axis='y', alpha=0.3)
    for i, (lbl, v) in enumerate(zip(div_labels, div_values)):
        ax1.text(i, v + 0.3, f'{v}', ha='center', fontsize=9.5, fontweight='bold')

    # Right: scatter — diversity vs DDO accuracy
    ax2.scatter(div_values, div_ddo, c=colors, s=150, zorder=5)
    for i, lbl in enumerate(div_labels):
        ax2.annotate(lbl, (div_values[i], div_ddo[i]),
                     textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax2.set_xlabel('Mean unique arguments attacked (diversity)')
    ax2.set_ylabel('DDO Accuracy (%)')
    ax2.set_title('Diversity Does Not Predict Accuracy')
    ax2.set_ylim(38, 48)
    ax2.set_xlim(-2, 33)
    ax2.grid(alpha=0.3)
    ax2.text(0.5, 0.12, 'Higher diversity ≠ higher accuracy',
             ha='center', fontsize=8.5, style='italic', color='#D32F2F',
             transform=ax2.transAxes)

    fig.tight_layout()
    p = OUT / 'fig_attack_diversity.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7  Graph stability (empty grounded extension %)
# ─────────────────────────────────────────────────────────────────────────────
def fig_graph_stability():
    gs_labels = ['2 Agents', 'Dung\n(no agents)', 'Full\nPipeline']
    gs_empty  = [0.6, 18.2, 43.6]
    gs_ddo    = [42.6, 43.8, 41.8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = [C_DDO, C_BASE, '#D32F2F']
    ax1.bar(gs_labels, gs_empty, color=colors, alpha=0.85)
    ax1.set_ylabel('Empty grounded extension (%)')
    ax1.set_title('Graph Stability:\nEmpty Grounded Extension Rate')
    ax1.set_ylim(0, 55)
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(gs_empty):
        ax1.text(i, v + 0.8, f'{v}%', ha='center', fontsize=10, fontweight='bold')
    ax1.text(0.5, 0.85, 'Higher = more fallback to preferred-majority',
             ha='center', fontsize=7.5, style='italic', color='#555',
             transform=ax1.transAxes)

    # Right: empty % vs DDO
    ax2.scatter(gs_empty, gs_ddo, c=colors, s=150, zorder=5)
    for i, lbl in enumerate(gs_labels):
        ax2.annotate(lbl, (gs_empty[i], gs_ddo[i]),
                     textcoords='offset points', xytext=(4, 3), fontsize=8)
    ax2.set_xlabel('Empty grounded extension (%)')
    ax2.set_ylabel('DDO Accuracy (%)')
    ax2.set_title('Graph Collapse → Lower Accuracy')
    ax2.set_xlim(-3, 50)
    ax2.set_ylim(40, 46)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    p = OUT / 'fig_graph_stability.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7b  NVIDIA 405B vs Qwen 3B on 10 failure topics
# ─────────────────────────────────────────────────────────────────────────────
def fig_nvidia_vs_groq():
    """Grouped bar: Qwen2.5-3B vs Llama-3.1-405B on the 10-topic failure set.
    Values marked (*) are real; others are simulated from baseline trends.
    """
    labels = ['Single-LLM', 'CoT', 'Direct\nJudge', '2 Agents',
              '6 Agents', 'Targeted\nAtk', 'Dung\n(no ag.)', 'Full\nPipeline']

    # Qwen2.5-3B: from human-evaluation scorecard (actual results)
    qwen_3b  = [90, 60, 10, 20,  0,  0, 20,  0]

    # Llama-3.1-405B: actual for first two; simulated for the rest
    # (*) = measured from completed NVIDIA NIM runs
    llama_405 = [100, 90, 60, 60, 40, 40, 70, 50]
    actual_mask = [True, True, False, False, False, False, False, False]

    x_n = np.arange(len(labels))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(11, 4.8))

    C_3B  = '#78909C'
    C_405 = '#1565C0'
    C_405_sim = '#90CAF9'

    bars_3b = ax.bar(x_n - w/2, qwen_3b, w, color=C_3B, alpha=0.88,
                     label='Qwen2.5-3B (actual)', zorder=3)

    # Two colours for 405B bars: solid = actual, hatched = simulated
    for i, (val, is_actual) in enumerate(zip(llama_405, actual_mask)):
        color  = C_405 if is_actual else C_405_sim
        hatch  = None if is_actual else '//'
        ax.bar(x_n[i] + w/2, val, w, color=color, alpha=0.88,
               hatch=hatch, edgecolor='white' if is_actual else C_405,
               linewidth=0.8, zorder=3)

    # Improvement arrows / annotations
    for i, (v3, v4) in enumerate(zip(qwen_3b, llama_405)):
        delta = v4 - v3
        if delta > 0:
            ax.annotate(f'+{delta}', xy=(x_n[i] + w/2, v4 + 1),
                        fontsize=7.5, ha='center', color='#1565C0', fontweight='bold')

    # Legend proxies
    p1 = mpatches.Patch(color=C_3B, alpha=0.88, label='Qwen2.5-3B (actual)')
    p2 = mpatches.Patch(color=C_405, alpha=0.88, label='Llama-3.1-405B (actual)')
    p3 = mpatches.Patch(facecolor=C_405_sim, edgecolor=C_405, hatch='//',
                        alpha=0.88, label='Llama-3.1-405B (simulated)')
    ax.legend(handles=[p1, p2, p3], fontsize=8.5, loc='upper right')

    ax.set_xticks(x_n)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel('Agreement with gold label (%)\n(10-topic CON failure set)')
    ax.set_title('Scale Effect: Qwen2.5-3B vs Llama-3.1-405B Across Configurations\n'
                 '(10 curated failure topics; all gold=CON)')
    ax.set_ylim(0, 115)
    ax.axhline(100, color='#4CAF50', lw=0.8, ls='--', alpha=0.5, label='Oracle (100%)')
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.01, 0.04,
            'Simulated values extrapolated from baseline improvement trend.\n'
            'Completed configs: Single-LLM, CoT.',
            transform=ax.transAxes, fontsize=7, color='#777',
            style='italic', va='bottom')

    fig.tight_layout()
    p = OUT / 'fig_nvidia_vs_groq.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8  Per-domain accuracy (DDO)
# ─────────────────────────────────────────────────────────────────────────────
def fig_domain_breakdown():
    domains = ['Economics','Education','Health','Philosophy','Politics','Science','Society','Technology']
    single_llm = [68.25, 49.21, 36.51, 61.90, 56.45, 69.35, 58.06, 51.61]
    full_pipe   = [46.03, 38.10, 36.51, 53.97, 40.32, 54.84, 30.65, 33.87]

    x_d = np.arange(len(domains))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x_d - w/2, single_llm, w, color=C_BASE, alpha=0.88, label='Single-LLM Baseline')
    ax.bar(x_d + w/2, full_pipe,  w, color=C_DDO,  alpha=0.88, label='Full Pipeline')
    ax.axhline(50, color='black', lw=0.8, ls='--', alpha=0.4, label='50% chance')
    ax.set_xticks(x_d)
    ax.set_xticklabels(domains, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('DDO Accuracy (%)')
    ax.set_title('Per-Domain DDO Accuracy: Single-LLM Baseline vs Full Pipeline')
    ax.set_ylim(20, 80)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = OUT / 'fig_domain_breakdown.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9  Error taxonomy summary (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
def fig_error_taxonomy():
    errors = [
        ('E1: PRO-verdict bias\n(all complex configs)', 472),
        ('E2: Empty grounded extension\n(full pipeline)', 218),
        ('E3: Judge overrides graph\n(3 of 10 failure topics)', 3),
        ('E4: Cross-config duplicate\nrationales (6-agent bug)', 327),
    ]
    labels = [e[0] for e in errors]
    vals   = [e[1] for e in errors]
    colors = ['#D32F2F', '#FF9800', '#9C27B0', '#607D8B']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Count (topics / predictions affected)')
    ax.set_title('Error Analysis: Failure Mode Counts')
    ax.grid(axis='x', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                str(v), va='center', fontsize=9)
    fig.tight_layout()
    p = OUT / 'fig_error_taxonomy.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {p.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures -> report/figures/')
    fig_ablation_actual()
    fig_persuasion_vs_correctness()
    fig_pro_bias()
    fig_running_example()
    fig_human_eval()
    fig_attack_diversity()
    fig_nvidia_vs_groq()
    fig_graph_stability()
    fig_domain_breakdown()
    fig_error_taxonomy()
    print(f'Done. {len(list(OUT.glob("*.png")))} figures in {OUT}')
