#!/usr/bin/env python3
"""
Evaluate all ablation outputs and produce the final ablation table.

Reads each config's stage4_judgments.json (or baseline JSON) under
outputs/stage4/<...>/stage4_judgments.json, computes DDO persuasion
agreement, per-domain breakdown, and writes:

  - outputs/ablations/<split>/ablation_table.csv    (drop into the LaTeX)
  - outputs/ablations/<split>/ablation_table.json   (full metrics)
  - outputs/ablations/<split>/ablation_table.md     (pretty print)

The CSV maps 1:1 onto the row order of the proposal's Table
"Planned ablation configurations" so you can paste values directly
into the .tex file.

Also computes Stage-2 side metrics that the proposal mentions:
  - attack diversity (unique premises attacked) — RQ2
  - graph stability (size of grounded extension) — discussion
  - % of attack labels where stage 2 flagged Attack — RQ3 comparison
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Config map — must match run_all_ablations.py
# ---------------------------------------------------------------------------

ABLATION_ROWS = [
    ('single_llm',        'Single-LLM Baseline'),
    ('cot',               '+ CoT Baseline'),
    ('direct_judge',      '+ Direct Judge (strong)'),
    ('two_agents',        '+ 2 Agents'),
    ('six_agents',        '+ 6 Agents'),
    ('targeted_attacks',  '+ Targeted Attacks'),
    ('dung_no_agents',    '+ Dung Graph (no agents)'),
    ('full',              'Full (6ag.+targ.+graph)'),
]


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def stage4_path_for(split, config_name):
    """Mirror run_all_ablations.py output layout."""
    if config_name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage4' / split / 'stage4_judgments.json'
    return PROJECT_ROOT / 'outputs' / 'stage4' / f'{split}_{config_name}' / 'stage4_judgments.json'


def stage2_path_for(split, config_name):
    if config_name in ('full', 'six_agents', 'direct_judge', 'targeted_attacks'):
        return PROJECT_ROOT / 'outputs' / 'stage2' / split / 'stage2_relations.json'
    return PROJECT_ROOT / 'outputs' / 'stage2' / f'{split}_{config_name}' / 'stage2_relations.json'


def stage3_path_for(split, config_name):
    if config_name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage3' / split / 'stage3_graphs.json'
    return PROJECT_ROOT / 'outputs' / 'stage3' / f'{split}_{config_name}' / 'stage3_graphs.json'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def accuracy_metrics(judgments):
    """Persuasion agreement (DDO crowd). Returns mu + confidence proxy."""
    matched = []  # 1/0 per matchable topic
    per_domain = defaultdict(list)
    for j in judgments:
        gold = j.get('benchmark_label')
        pred = j.get('verdict')
        if not gold or gold == 'TIE':
            continue
        hit = 1 if pred == gold else 0
        matched.append(hit)
        per_domain[j.get('domain') or 'unknown'].append(hit)

    n = len(matched)
    if n == 0:
        return None
    mu = statistics.mean(matched)
    # Wilson 95% CI half-width (quick, no scipy)
    # sigma ~= sqrt(p(1-p)/n)
    import math
    sigma = math.sqrt(mu * (1 - mu) / max(n, 1))
    return {
        'n': n,
        'acc_mean': round(100 * mu, 2),
        'acc_std': round(100 * sigma, 2),
        'per_domain': {
            d: {'n': len(v), 'acc': round(100 * statistics.mean(v), 2)}
            for d, v in per_domain.items()
        },
    }


def confidence_stats(judgments):
    """Mean/std of the model's self-reported confidence (proxy for
    'persuasiveness' column in the ablation table when no human eval yet)."""
    cs = [j.get('confidence', 0.0) for j in judgments if j.get('verdict') != 'TIE']
    if not cs:
        return None
    return {
        'n': len(cs),
        'pers_mean': round(5 * statistics.mean(cs), 2),       # rescale 0-1 -> 0-5 Likert proxy
        'pers_std': round(5 * statistics.stdev(cs) if len(cs) > 1 else 0.0, 2),
    }


def attack_diversity(stage2):
    """Attack diversity = mean unique attacked-argument count per topic (RQ2)."""
    vals = []
    for t in stage2.get('topics', []):
        attacked = {r['target_arg_id']
                    for r in t.get('relations', [])
                    if r.get('kept') and r.get('label') == 'Attack'}
        vals.append(len(attacked))
    if not vals:
        return None
    return {
        'n_topics': len(vals),
        'mean_unique_attacked': round(statistics.mean(vals), 2),
        'median_unique_attacked': statistics.median(vals),
    }


def graph_stability(stage3):
    """Mean grounded-extension size per topic."""
    vals = [g.get('grounded_size', 0) for g in stage3.get('graphs', [])]
    if not vals:
        return None
    return {
        'n_topics': len(vals),
        'mean_grounded_size': round(statistics.mean(vals), 2),
        'pct_empty_grounded': round(100 * sum(1 for v in vals if v == 0) / len(vals), 2),
    }


def persuasion_correctness_gap(judgments, graph_judgments=None):
    """Regret gap: where graph verdict and judge verdict disagree,
    compute error rate difference (if available)."""
    gv_agrees = jv_agrees = 0
    n_with_both = 0
    for j in judgments:
        gold = j.get('benchmark_label')
        if not gold or gold == 'TIE':
            continue
        gv = j.get('graph_verdict')
        pv = j.get('verdict')
        if not gv:
            continue
        n_with_both += 1
        if gv == gold:
            gv_agrees += 1
        if pv == gold:
            jv_agrees += 1
    if n_with_both == 0:
        return None
    return {
        'n': n_with_both,
        'graph_only_acc_pct': round(100 * gv_agrees / n_with_both, 2),
        'judge_acc_pct': round(100 * jv_agrees / n_with_both, 2),
        'gap_pp': round(100 * (jv_agrees - gv_agrees) / n_with_both, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_config(split, config_name, label):
    s4_path = stage4_path_for(split, config_name)
    row = {'config_name': config_name, 'label': label,
           'stage4_path': str(s4_path), 'status': 'missing'}
    if not s4_path.exists():
        return row

    s4 = json.loads(s4_path.read_text())
    judgments = s4.get('judgments', [])
    row['status'] = 'present'
    row['n_judgments'] = len(judgments)

    acc = accuracy_metrics(judgments)
    if acc:
        row.update({
            'acc_mean_pct': acc['acc_mean'],
            'acc_std_pct': acc['acc_std'],
            'acc_n': acc['n'],
            'acc_by_domain': acc['per_domain'],
        })
    pers = confidence_stats(judgments)
    if pers:
        row.update({
            'pers_mean_5': pers['pers_mean'],
            'pers_std_5': pers['pers_std'],
        })

    # Side metrics (for configs that have stage2 / stage3)
    s2_path = stage2_path_for(split, config_name)
    if s2_path.exists():
        try:
            s2 = json.loads(s2_path.read_text())
            row['attack_diversity'] = attack_diversity(s2)
            lc = s2.get('summary', {}).get('label_counts', {})
            row['stage2_label_counts'] = lc
        except Exception as ex:
            row['stage2_error'] = str(ex)

    s3_path = stage3_path_for(split, config_name)
    if s3_path.exists():
        try:
            s3 = json.loads(s3_path.read_text())
            row['graph_stability'] = graph_stability(s3)
        except Exception as ex:
            row['stage3_error'] = str(ex)

    # Graph vs judge error decomposition (only meaningful for full/dung/two_agents)
    pc = persuasion_correctness_gap(judgments)
    if pc:
        row['graph_vs_judge'] = pc

    return row


def write_csv(rows, path):
    fields = ['config_name', 'label', 'status', 'n_judgments',
              'acc_mean_pct', 'acc_std_pct', 'acc_n',
              'pers_mean_5', 'pers_std_5']
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows, path):
    lines = ['# Ablation Results', '']
    lines.append('| Configuration | Acc μ | Acc σ | Pers μ | Pers σ | N |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for r in rows:
        if r['status'] != 'present':
            lines.append(f'| {r["label"]} | -- | -- | -- | -- | _missing_ |')
            continue
        lines.append(
            f'| {r["label"]} | '
            f'{r.get("acc_mean_pct", "--")} | {r.get("acc_std_pct", "--")} | '
            f'{r.get("pers_mean_5", "--")} | {r.get("pers_std_5", "--")} | '
            f'{r.get("acc_n", r.get("n_judgments", "--"))} |'
        )
    path.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='ddo_sample')
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / 'outputs' / 'ablations' / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Project root : {PROJECT_ROOT}')
    print(f'Split        : {args.split}')
    print(f'Output dir   : {out_dir}\n')

    rows = []
    for name, label in ABLATION_ROWS:
        print(f'  evaluating {name:20s} -> ', end='', flush=True)
        row = evaluate_config(args.split, name, label)
        rows.append(row)
        if row['status'] == 'present':
            acc = row.get('acc_mean_pct', '--')
            n = row.get('acc_n', row.get('n_judgments', '--'))
            print(f'OK acc={acc}% (n={n})')
        else:
            print('MISSING (not yet run)')

    # Write outputs
    json_path = out_dir / 'ablation_table.json'
    json_path.write_text(json.dumps({
        'split': args.split,
        'rows': rows,
    }, indent=2))

    csv_path = out_dir / 'ablation_table.csv'
    write_csv(rows, csv_path)

    md_path = out_dir / 'ablation_table.md'
    write_markdown(rows, md_path)

    print(f'\nWrote:')
    print(f'  {json_path}')
    print(f'  {csv_path}')
    print(f'  {md_path}')

    present = sum(1 for r in rows if r['status'] == 'present')
    print(f'\n{present}/{len(rows)} configurations have results on disk.')
    if present < len(rows):
        missing = [r['config_name'] for r in rows if r['status'] != 'present']
        print(f'Still to run: {missing}')


if __name__ == '__main__':
    main()
