#!/usr/bin/env python3
"""
Score human annotations and produce the correctness-agreement ablation table.

Reads:
  - annotations/<split>_annotator_{1,2,3}.csv  (filled by human judges)
  - outputs/stage4/<split>[_<config>]/stage4_judgments.json  (model verdicts)

Produces:
  - outputs/ablations/<split>/human_eval_table.csv   (for LaTeX)
  - outputs/ablations/<split>/human_eval_table.md
  - outputs/ablations/<split>/human_eval_table.json
  - outputs/ablations/<split>/human_annotation_summary.json
    (Krippendorff alpha, majority labels, per-annotator stats)

Metrics:
  - Correctness agreement % — model verdict vs majority-vote annotator label
  - Per-domain breakdown
  - Persuasion-vs-correctness regret gap (compares human-majority label
    against DDO crowd label when both exist — here always just human)
  - Inter-annotator agreement (Krippendorff alpha, computed from scratch
    so we don't need scipy/numpy)

Usage:
    python scripts/score_human_eval.py --split human_eval

Partial mode — score what you have even if some annotators haven't submitted:
    python scripts/score_human_eval.py --split human_eval --require-n 2
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (mirror run_all_ablations.py)
# ---------------------------------------------------------------------------

def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


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


def stage4_path(split, config):
    if config == 'full':
        return PROJECT_ROOT / 'outputs/stage4' / split / 'stage4_judgments.json'
    return PROJECT_ROOT / 'outputs/stage4' / f'{split}_{config}' / 'stage4_judgments.json'


# ---------------------------------------------------------------------------
# Krippendorff's alpha for nominal data (PRO/CON/TIE)
# ---------------------------------------------------------------------------

def krippendorff_alpha_nominal(judgments_per_annotator, units):
    """Compute Krippendorff's alpha for nominal (categorical) data.

    judgments_per_annotator: list of dicts {unit_id: label}
    units: iterable of unit IDs

    Returns alpha in [-1, 1]. 1 = perfect agreement, 0 = chance, <0 = worse
    than chance. Handles missing data (annotator didn't rate a unit).
    """
    # Build the reliability data matrix: rows = annotators, cols = units
    matrix = []
    for ann in judgments_per_annotator:
        matrix.append([ann.get(u) for u in units])

    n_annotators = len(matrix)
    n_units = len(units)

    # For each unit, count how many annotators rated it
    # Observed disagreement: pairwise disagreements within each unit,
    # summed and normalised.
    # Expected disagreement: derived from overall label marginals.

    all_labels = set()
    for row in matrix:
        for v in row:
            if v is not None:
                all_labels.add(v)
    all_labels = sorted(all_labels)

    # Coincidence matrix o[c,c'] = how often labels c,c' appear together
    # in the same unit across all annotator pairs.
    o = {c: {c2: 0.0 for c2 in all_labels} for c in all_labels}
    n_c = {c: 0.0 for c in all_labels}
    total_pairs = 0
    for j in range(n_units):
        column = [matrix[i][j] for i in range(n_annotators) if matrix[i][j] is not None]
        m_u = len(column)
        if m_u < 2:
            continue
        # All ordered pairs within this unit, weight = 1/(m_u - 1)
        for a in range(m_u):
            for b in range(m_u):
                if a == b:
                    continue
                o[column[a]][column[b]] += 1.0 / (m_u - 1)
        for c in column:
            n_c[c] += 1.0
        total_pairs += m_u

    if total_pairs == 0:
        return None

    # Observed disagreement Do
    n_total = sum(n_c.values())
    Do = 0.0
    for c in all_labels:
        for c2 in all_labels:
            if c != c2:
                Do += o[c][c2]
    # For nominal: Do_nominal = sum_{c!=c'} o_{c c'} / n  (n = total coded values)
    if n_total == 0:
        return None
    Do = Do / n_total

    # Expected disagreement De
    De = 0.0
    for c in all_labels:
        for c2 in all_labels:
            if c != c2:
                De += n_c[c] * n_c[c2]
    # Normalise by n(n-1)
    if n_total <= 1:
        return None
    De = De / (n_total * (n_total - 1))

    if De == 0:
        return 1.0 if Do == 0 else None
    alpha = 1 - (Do / De)
    return alpha


# ---------------------------------------------------------------------------
# Load annotations
# ---------------------------------------------------------------------------

def load_annotator_csv(path):
    """Return dict topic_id -> {winner, persuasiveness, confidence, notes}."""
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get('topic_id', '').strip()
            if not tid:
                continue
            winner = (row.get('winner_pick') or '').strip().upper()
            if winner not in ('PRO', 'CON', 'TIE', ''):
                winner = ''
            try:
                pers = int(row['winner_persuasiveness']) if row.get('winner_persuasiveness') else None
            except (ValueError, TypeError):
                pers = None
            try:
                conf = int(row['correctness_confidence']) if row.get('correctness_confidence') else None
            except (ValueError, TypeError):
                conf = None
            rows[tid] = {
                'winner': winner if winner else None,
                'persuasiveness': pers,
                'confidence': conf,
                'notes': (row.get('notes') or '').strip(),
                'domain': (row.get('domain') or '').strip(),
            }
    return rows


def majority_label(labels):
    """Return majority of valid (non-None) labels, with 'TIE' tie-break.

    If 2 of 3 agree -> that label.
    If all 3 disagree -> 'TIE' (we report these separately).
    """
    valid = [l for l in labels if l]
    if not valid:
        return None
    c = Counter(valid)
    top = c.most_common()
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return 'TIE'  # no clear majority


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(diffs, n_permutations=10000, seed=42):
    """One-sample permutation test for H0: mean(diffs) = 0.

    diffs: list of (model_A_correct - model_B_correct) per topic in {-1,0,1}
    Returns two-sided p-value.
    """
    import random
    rng = random.Random(seed)
    diffs = [d for d in diffs if d is not None]
    if not diffs:
        return None
    observed = sum(diffs) / len(diffs)
    n = len(diffs)
    count = 0
    abs_observed = abs(observed)
    for _ in range(n_permutations):
        # Random sign flip on each diff
        shuffled = sum(rng.choice([-1, 1]) * d for d in diffs)
        if abs(shuffled / n) >= abs_observed:
            count += 1
    return count / n_permutations


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='human_eval')
    ap.add_argument('--annotations-dir', default='annotations')
    ap.add_argument('--require-n', type=int, default=3,
                    help='minimum annotators required to score (default 3)')
    ap.add_argument('--n-permutations', type=int, default=10000)
    args = ap.parse_args()

    ann_dir = PROJECT_ROOT / args.annotations_dir
    if not ann_dir.exists():
        raise SystemExit(f'Annotations dir not found: {ann_dir}')

    # Load all annotator CSVs
    annotator_files = sorted(ann_dir.glob(f'{args.split}_annotator_*.csv'))
    print(f'Found {len(annotator_files)} annotator file(s):')
    for p in annotator_files:
        print(f'  {p}')

    if len(annotator_files) < args.require_n:
        print(f'\nERROR: need at least {args.require_n} annotators, got {len(annotator_files)}.')
        print('Either wait for more annotations, or lower --require-n.')
        raise SystemExit(1)

    annotations = [load_annotator_csv(p) for p in annotator_files]

    # Build the set of topic_ids that ALL annotators rated with a valid winner
    all_topic_ids = set()
    for a in annotations:
        all_topic_ids.update(a.keys())

    # For each topic, get the list of annotator winners
    per_topic = {}
    for tid in sorted(all_topic_ids):
        winners = [a.get(tid, {}).get('winner') for a in annotations]
        per_topic[tid] = {
            'annotator_winners': winners,
            'majority': majority_label(winners),
            'persuasiveness_scores': [a.get(tid, {}).get('persuasiveness') for a in annotations],
            'confidence_scores': [a.get(tid, {}).get('confidence') for a in annotations],
            'domain': next((a.get(tid, {}).get('domain') for a in annotations
                            if a.get(tid, {}).get('domain')), ''),
        }

    n_with_majority = sum(1 for v in per_topic.values() if v['majority'])
    print(f'\n{n_with_majority}/{len(per_topic)} topics have a valid majority label')

    # Krippendorff alpha for inter-annotator agreement on winner
    topic_ids_sorted = sorted(per_topic.keys())
    winner_per_ann = [dict((tid, a.get(tid, {}).get('winner')) for tid in topic_ids_sorted)
                      for a in annotations]
    alpha = krippendorff_alpha_nominal(winner_per_ann, topic_ids_sorted)
    print(f'Krippendorff alpha (winner): {alpha:.3f}' if alpha is not None else
          'Krippendorff alpha: could not compute')

    # For each config, compute correctness agreement
    rows = []
    raw_correct = {}  # config -> {tid: 1|0|None}  for permutation tests
    for config_name, label in ABLATION_ROWS:
        p = stage4_path(args.split, config_name)
        row = {'config_name': config_name, 'label': label, 'status': 'missing'}
        if not p.exists():
            rows.append(row)
            continue
        doc = json.loads(p.read_text())
        judgments = {j['topic_id']: j for j in doc.get('judgments', [])}

        # Score against human majority
        matched = []
        per_domain = defaultdict(list)
        per_topic_correct = {}
        for tid, info in per_topic.items():
            gold = info['majority']
            if not gold or gold == 'TIE':
                per_topic_correct[tid] = None
                continue
            if tid not in judgments:
                per_topic_correct[tid] = None
                continue
            pred = judgments[tid].get('verdict')
            hit = 1 if pred == gold else 0
            matched.append(hit)
            per_domain[info.get('domain') or 'unknown'].append(hit)
            per_topic_correct[tid] = hit

        raw_correct[config_name] = per_topic_correct

        n = len(matched)
        if n == 0:
            row['status'] = 'no_matchable'
            rows.append(row)
            continue

        mu = statistics.mean(matched)
        sigma = math.sqrt(mu * (1 - mu) / max(n, 1))

        # Average persuasiveness rating of the winner, when the model agreed
        # with the majority — approximation for proposal's "Pers." column
        pers_when_correct = []
        for tid, info in per_topic.items():
            if per_topic_correct.get(tid) == 1:
                for s in info['persuasiveness_scores']:
                    if s is not None:
                        pers_when_correct.append(s)

        row.update({
            'status': 'present',
            'n_matched': n,
            'correctness_acc_pct': round(100 * mu, 2),
            'correctness_se_pct': round(100 * sigma, 2),
            'per_domain_acc': {
                d: {'n': len(v),
                    'acc': round(100 * statistics.mean(v), 2) if v else None}
                for d, v in per_domain.items()
            },
            'persuasiveness_when_correct_mean':
                round(statistics.mean(pers_when_correct), 2) if pers_when_correct else None,
            'persuasiveness_when_correct_std':
                round(statistics.stdev(pers_when_correct), 2) if len(pers_when_correct) > 1 else None,
            'n_persuasiveness_obs': len(pers_when_correct),
        })
        rows.append(row)

    # Paired permutation tests: every config vs single_llm (and vs full)
    print('\nPaired permutation tests (vs single_llm):')
    for config_name, _ in ABLATION_ROWS:
        if config_name == 'single_llm':
            continue
        base = raw_correct.get('single_llm', {})
        other = raw_correct.get(config_name, {})
        diffs = []
        for tid in per_topic:
            b = base.get(tid)
            o = other.get(tid)
            if b is not None and o is not None:
                diffs.append(o - b)
        if diffs:
            p_val = paired_permutation_test(diffs, n_permutations=args.n_permutations)
            mean_diff = sum(diffs) / len(diffs) * 100
            sig = ''
            if p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            print(f'  {config_name:<20} vs single_llm: '
                  f'delta={mean_diff:+.1f}pp  p={p_val:.4f} {sig}')
            # Stash into the row
            for r in rows:
                if r['config_name'] == config_name:
                    r['vs_single_llm_delta_pp'] = round(mean_diff, 2)
                    r['vs_single_llm_pvalue'] = round(p_val, 4)

    # ---- Write outputs ----
    out_dir = PROJECT_ROOT / 'outputs/ablations' / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (full detail)
    json_path = out_dir / 'human_eval_table.json'
    json_path.write_text(json.dumps({
        'split': args.split,
        'n_annotators': len(annotations),
        'n_topics': len(per_topic),
        'n_topics_with_majority': n_with_majority,
        'krippendorff_alpha_winner': alpha,
        'rows': rows,
        'per_topic': per_topic,
    }, indent=2))

    # CSV
    csv_path = out_dir / 'human_eval_table.csv'
    fields = ['config_name', 'label', 'status', 'n_matched',
              'correctness_acc_pct', 'correctness_se_pct',
              'persuasiveness_when_correct_mean', 'persuasiveness_when_correct_std',
              'vs_single_llm_delta_pp', 'vs_single_llm_pvalue']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown
    md_path = out_dir / 'human_eval_table.md'
    lines = ['# Human Evaluation Results (Correctness Agreement)', '']
    alpha_s = f'{alpha:.3f}' if alpha is not None else 'n/a'
    lines.append(f'- **Annotators**: {len(annotations)}')
    lines.append(f'- **Topics**: {len(per_topic)} ({n_with_majority} with clear majority)')
    lines.append(f'- **Krippendorff alpha (winner)**: {alpha_s}')
    lines.append('')
    lines.append('| Configuration | Corr.% | SE | Pers. μ | Pers. σ | Δ vs single | p |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        if r['status'] != 'present':
            lines.append(f'| {r["label"]} | -- | -- | -- | -- | -- | _{r["status"]}_ |')
            continue
        delta = r.get('vs_single_llm_delta_pp')
        p_val = r.get('vs_single_llm_pvalue')
        sig_marker = ''
        if p_val is not None:
            if p_val < 0.01:
                sig_marker = '**'
            elif p_val < 0.05:
                sig_marker = '*'
        pmu = r.get('persuasiveness_when_correct_mean') or '--'
        psigma = r.get('persuasiveness_when_correct_std') or '--'
        lines.append(
            f'| {r["label"]} | '
            f'{r["correctness_acc_pct"]} | {r["correctness_se_pct"]} | '
            f'{pmu} | {psigma} | '
            f'{delta if delta is not None else "--"} | '
            f'{p_val if p_val is not None else "--"} {sig_marker} |'
        )
    lines.append('')
    lines.append('*p<0.05, **p<0.01 (paired permutation test vs single_llm).')
    md_path.write_text('\n'.join(lines) + '\n')

    # Human annotation summary
    summ_path = out_dir / 'human_annotation_summary.json'
    summ = {
        'n_annotators': len(annotations),
        'annotator_files': [str(p.name) for p in annotator_files],
        'krippendorff_alpha_winner': alpha,
        'topic_majority_distribution': dict(Counter(
            v['majority'] for v in per_topic.values())),
        'per_topic': per_topic,
    }
    summ_path.write_text(json.dumps(summ, indent=2))

    print(f'\nWrote:')
    print(f'  {json_path}')
    print(f'  {csv_path}')
    print(f'  {md_path}')
    print(f'  {summ_path}')
    print(f'\nHuman evaluation results summary:')
    for r in rows:
        if r['status'] != 'present':
            continue
        delta_s = ''
        if r.get('vs_single_llm_delta_pp') is not None:
            delta_s = f' (Δ={r["vs_single_llm_delta_pp"]:+.1f}pp vs single_llm, p={r["vs_single_llm_pvalue"]})'
        print(f'  {r["label"]:<28} {r["correctness_acc_pct"]:>5.1f}% ± {r["correctness_se_pct"]:.1f}{delta_s}')


if __name__ == '__main__':
    main()
