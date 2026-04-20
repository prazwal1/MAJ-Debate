#!/usr/bin/env python3
"""
Side-by-side comparison of persuasion (DDO) vs correctness (logic_test) results.

This is the key analysis that makes MAJ-Debate's story coherent:
  - DDO crowd votes measure audience persuasion
  - logic_test topics measure logical/factual correctness
  - If the full pipeline wins on logic_test but loses on DDO, the proposal's
    central thesis (persuasion != correctness) is supported.

Reads:
  - outputs/ablations/ddo_sample/ablation_table.json
  - outputs/ablations/logic_test/ablation_table.json

Writes:
  - outputs/ablations/persuasion_vs_correctness.md
  - outputs/ablations/persuasion_vs_correctness.csv

Usage:
    python scripts/compare_benchmarks.py
"""

import argparse
import csv
import json
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def load_ablation_table(split):
    p = PROJECT_ROOT / 'outputs' / 'ablations' / split / 'ablation_table.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--persuasion-split', default='ddo_sample',
                    help='split measuring audience persuasion (default ddo_sample)')
    ap.add_argument('--correctness-split', default='logic_test',
                    help='split measuring logical correctness (default logic_test)')
    args = ap.parse_args()

    persuasion = load_ablation_table(args.persuasion_split)
    correctness = load_ablation_table(args.correctness_split)

    if persuasion is None:
        print(f'ERROR: no ablation table for {args.persuasion_split}')
        print(f'       run ablations on the DDO sample first')
        raise SystemExit(1)
    if correctness is None:
        print(f'ERROR: no ablation table for {args.correctness_split}')
        print(f'       run:')
        print(f'         bash scripts/run_ablations_on_split.sh logic_test')
        raise SystemExit(1)

    # Build a config_name -> row map for each
    pers_by_cfg = {r['config_name']: r for r in persuasion.get('rows', [])}
    corr_by_cfg = {r['config_name']: r for r in correctness.get('rows', [])}

    configs = [
        ('single_llm',        'Single-LLM Baseline'),
        ('cot',               '+ CoT Baseline'),
        ('direct_judge',      '+ Direct Judge (strong)'),
        ('two_agents',        '+ 2 Agents'),
        ('six_agents',        '+ 6 Agents'),
        ('targeted_attacks',  '+ Targeted Attacks'),
        ('dung_no_agents',    '+ Dung Graph (no agents)'),
        ('full',              'Full (6ag.+targ.+graph)'),
    ]

    # Build combined rows
    rows = []
    for name, label in configs:
        p = pers_by_cfg.get(name, {})
        c = corr_by_cfg.get(name, {})
        p_acc = p.get('acc_mean_pct') if p.get('status') == 'present' else None
        c_acc = c.get('acc_mean_pct') if c.get('status') == 'present' else None
        p_n = p.get('acc_n', p.get('n_judgments'))
        c_n = c.get('acc_n', c.get('n_judgments'))

        gap = None
        if p_acc is not None and c_acc is not None:
            gap = round(c_acc - p_acc, 2)

        rows.append({
            'config_name': name,
            'label': label,
            'persuasion_acc_pct': p_acc,
            'persuasion_n': p_n,
            'correctness_acc_pct': c_acc,
            'correctness_n': c_n,
            'correctness_minus_persuasion_pp': gap,
        })

    # Per-domain breakdown for correctness (shows which topic types the
    # graph helps on)
    per_domain_corr = {}
    for name, label in configs:
        c = corr_by_cfg.get(name, {})
        if c.get('status') != 'present':
            continue
        domains = c.get('acc_by_domain', {})
        per_domain_corr[name] = domains

    # ---- Write markdown ----
    out_dir = PROJECT_ROOT / 'outputs' / 'ablations'
    md_path = out_dir / 'persuasion_vs_correctness.md'
    lines = [
        '# Persuasion (DDO) vs Correctness (Logic) — Ablation Comparison',
        '',
        'Left column: DDO crowd-vote agreement (audience persuasion).',
        'Right column: logic_test ground-truth agreement (logical correctness).',
        'Gap: correctness − persuasion. Positive = pipeline helps correctness '
        'more than persuasion.',
        '',
        '| Configuration | Pers. (DDO) | n | Corr. (Logic) | n | Gap |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        p = r['persuasion_acc_pct']
        c = r['correctness_acc_pct']
        gap = r['correctness_minus_persuasion_pp']
        p_str = f'{p:.1f}%' if p is not None else '--'
        c_str = f'{c:.1f}%' if c is not None else '--'
        gap_str = f'{gap:+.1f}' if gap is not None else '--'
        lines.append(f'| {r["label"]} | {p_str} | {r["persuasion_n"] or "--"} | '
                     f'{c_str} | {r["correctness_n"] or "--"} | {gap_str} |')

    lines.append('')
    lines.append('## Per-domain correctness (logic_test only)')
    lines.append('')
    all_domains = set()
    for domains in per_domain_corr.values():
        all_domains.update(domains.keys())
    all_domains = sorted(all_domains)
    header = '| Configuration | ' + ' | '.join(all_domains) + ' |'
    sep = '|---|' + '|'.join(['---:'] * len(all_domains)) + '|'
    lines.append(header)
    lines.append(sep)
    for name, label in configs:
        domains = per_domain_corr.get(name, {})
        if not domains:
            continue
        cells = []
        for d in all_domains:
            info = domains.get(d, {})
            if 'acc' in info:
                cells.append(f'{info["acc"]:.1f}% (n={info["n"]})')
            else:
                cells.append('--')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    lines.append('')
    lines.append('## Interpretation guide')
    lines.append('')
    lines.append('**RQ1 (graph contribution)** compares `full` vs `direct_judge`:')
    lines.append('  - If Gap(full) > Gap(direct_judge), the graph helps correctness '
                 'more than persuasion.')
    lines.append('')
    lines.append('**Per-domain signal** matters most for the logic benchmark:')
    lines.append('  - `syllogism` and `fallacy` topics have pure logical structure.')
    lines.append('  - `paradox` tests self-contradiction detection.')
    lines.append('  - `math_fact` and `empirical_fact` test grounded reasoning about '
                 'objective claims.')
    lines.append('  - Large gains on `syllogism`/`fallacy` but not others would be '
                 'strong evidence that graph semantics help where argument structure '
                 'matters.')
    md_path.write_text('\n'.join(lines) + '\n')

    # ---- Write CSV ----
    csv_path = out_dir / 'persuasion_vs_correctness.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Console output ----
    print(f'Persuasion split  : {args.persuasion_split}')
    print(f'Correctness split : {args.correctness_split}')
    print()
    print(f'{"Config":<28} {"Pers%":>8} {"Corr%":>8} {"Gap":>8}')
    print('-' * 60)
    for r in rows:
        p = r['persuasion_acc_pct']
        c = r['correctness_acc_pct']
        gap = r['correctness_minus_persuasion_pp']
        p_str = f'{p:.1f}' if p is not None else '--'
        c_str = f'{c:.1f}' if c is not None else '--'
        gap_str = f'{gap:+.1f}' if gap is not None else '--'
        print(f'{r["label"]:<28} {p_str:>8} {c_str:>8} {gap_str:>8}')

    print()
    print(f'Wrote:')
    print(f'  {md_path}')
    print(f'  {csv_path}')


if __name__ == '__main__':
    main()
