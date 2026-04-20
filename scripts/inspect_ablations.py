#!/usr/bin/env python3
"""
Inspect ablation outputs to spot parse failures and obvious problems
BEFORE launching the big 500-topic run.

Prints a per-config diagnostic:
  - total judgments, scorable (with benchmark_label), parse failures
  - verdict distribution (how many PRO/CON/TIE)
  - avg confidence (suspiciously low = many fallback verdicts)
  - agreement with benchmark
  - first 2 rationale samples (so you can eyeball quality)

Usage:
    python scripts/inspect_ablations.py --split ddo_sample
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()

ABLATION_NAMES = [
    'single_llm', 'cot', 'direct_judge', 'two_agents',
    'six_agents', 'targeted_attacks', 'dung_no_agents', 'full',
]


def stage4_path(split, name):
    if name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage4' / split / 'stage4_judgments.json'
    return PROJECT_ROOT / 'outputs' / 'stage4' / f'{split}_{name}' / 'stage4_judgments.json'


def inspect(name, split):
    p = stage4_path(split, name)
    if not p.exists():
        return {'name': name, 'status': 'MISSING', 'path': str(p)}

    doc = json.loads(p.read_text())
    judgments = doc.get('judgments', [])
    verdicts = Counter(j.get('verdict') for j in judgments)
    parse_fails = sum(1 for j in judgments
                      if 'parse_failed' in (j.get('rationale') or '').lower())
    # Partial recoveries are DIFFERENT from parse failures: the verdict is
    # usable (PRO/CON), just with shorter rationale.
    recoveries = sum(1 for j in judgments
                     if '[recovered' in (j.get('rationale') or '').lower())
    confs = [j.get('confidence', 0.0) for j in judgments]
    avg_conf = sum(confs) / max(len(confs), 1)

    # Did this config actually USE graph context? Check both the summary
    # flag and whether the rationale mentions graph-semantic terms.
    used_graph_flag = doc.get('summary', {}).get('used_graph', False)
    mentions_graph = sum(
        1 for j in judgments
        if any(w in (j.get('rationale') or '').lower()
               for w in ('grounded extension', 'preferred extension',
                         'dung', 'grounded', 'undefeated', 'attack graph'))
    )

    # Agreement with benchmark
    scorable = [j for j in judgments
                if j.get('benchmark_label') and j['benchmark_label'] != 'TIE']
    correct = sum(1 for j in scorable
                  if j.get('verdict') == j['benchmark_label'])
    agreement = 100.0 * correct / len(scorable) if scorable else None

    # Sample rationales
    samples = []
    for j in judgments[:2]:
        r = j.get('rationale', '')[:120]
        samples.append(f"    {j.get('topic_id', '?')}: verdict={j.get('verdict')}  \"{r}\"")

    return {
        'name': name,
        'status': 'OK',
        'n_judgments': len(judgments),
        'n_scorable': len(scorable),
        'parse_failures': parse_fails,
        'parse_fail_pct': round(100 * parse_fails / max(len(judgments), 1), 1),
        'recoveries': recoveries,
        'recovery_pct': round(100 * recoveries / max(len(judgments), 1), 1),
        'verdicts': dict(verdicts),
        'avg_confidence': round(avg_conf, 3),
        'agreement_pct': round(agreement, 1) if agreement is not None else None,
        'used_graph_flag': used_graph_flag,
        'mentions_graph_terms': mentions_graph,
        'mentions_graph_pct': round(100 * mentions_graph / max(len(judgments), 1), 1),
        'judgments': judgments,  # for cross-config duplicate detection
        'samples': samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='ddo_sample')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='show sample rationales for each config')
    args = ap.parse_args()

    print(f'Inspecting ablations for split: {args.split}\n')
    print(f'{"Config":<20} {"Status":<8} {"Judg":<6} {"Score":<6} '
          f'{"ParseF":<7} {"Recov":<7} {"Conf":<6} {"Agree%":<7}  Verdicts')
    print('-' * 110)

    suspicious = []
    results = {}   # name -> inspection dict (used by duplicate check below)
    for name in ABLATION_NAMES:
        r = inspect(name, args.split)
        if r['status'] == 'OK':
            results[name] = r
        if r['status'] == 'MISSING':
            print(f'{name:<20} \033[1;31mMISSING\033[0m  -> {r["path"]}')
            suspicious.append(('missing', name, 'file not found'))
            continue

        agree_str = f'{r["agreement_pct"]}%' if r['agreement_pct'] is not None else '--'
        conf_str = f'{r["avg_confidence"]:.2f}'
        pf_str = f'{r["parse_fail_pct"]}%'
        rec_str = f'{r["recovery_pct"]}%'

        # Color-code bad rows
        row_color = ''
        reset = ''
        if r['n_scorable'] == 0:
            row_color = '\033[1;31m'; reset = '\033[0m'
            suspicious.append(('no_scorable', name,
                               'no judgments have benchmark_label'))
        elif r['parse_fail_pct'] > 20:
            row_color = '\033[1;33m'; reset = '\033[0m'
            suspicious.append(('high_parse_fail', name,
                               f'{r["parse_fail_pct"]}% parse failures'))
        elif r['avg_confidence'] < 0.3:
            row_color = '\033[1;33m'; reset = '\033[0m'
            suspicious.append(('low_confidence', name,
                               f'avg confidence {conf_str}'))

        print(f'{row_color}{name:<20} OK       {r["n_judgments"]:<6} '
              f'{r["n_scorable"]:<6} {pf_str:<7} {rec_str:<7} {conf_str:<6} '
              f'{agree_str:<7}  {r["verdicts"]}{reset}')

        if args.verbose and r.get('samples'):
            for s in r['samples']:
                print(s)
            print()

    print()
    # -----------------------------------------------------------------
    # Cross-config duplicate detection.
    # If two configs produce identical rationales on the same topic, the
    # judge is reading identical inputs for both — which is a bug.
    # -----------------------------------------------------------------
    print('=' * 110)
    print('CROSS-CONFIG DUPLICATE CHECK')
    print('=' * 110)
    print('Comparing rationales across configs on the same topic_id.')
    print('Identical rationales across configs = they read identical inputs (BUG).\n')

    # Build topic -> {config_name: rationale} map
    per_topic = {}
    for cfg_name, r in results.items():
        for j in r['judgments']:
            tid = j.get('topic_id')
            if not tid:
                continue
            per_topic.setdefault(tid, {})[cfg_name] = j.get('rationale', '')

    # For each topic, find configs that share rationales
    dup_pairs = Counter()
    topics_with_dups = 0
    for tid, cfg_rats in per_topic.items():
        items = list(cfg_rats.items())
        matched_this_topic = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a_name, a_rat = items[i]
                b_name, b_rat = items[j]
                if a_rat and a_rat == b_rat:
                    dup_pairs[(a_name, b_name)] += 1
                    matched_this_topic = True
        if matched_this_topic:
            topics_with_dups += 1

    if not dup_pairs:
        print('\033[1;32mNo duplicate rationales detected across configs. '
              'Each config appears to read distinct inputs.\033[0m')
    else:
        print(f'\033[1;31m{topics_with_dups} topic(s) have identical rationales '
              f'across multiple configs.\033[0m\n')
        print('Duplicate pairs (config_a, config_b -> # shared topics):')
        for (a, b), n in dup_pairs.most_common():
            print(f'  {a:<20} == {b:<20}  ({n} topic(s) share identical rationale)')
        print()
        print('\033[1;33mDIAGNOSIS:\033[0m configs giving identical rationales on '
              'the same topics are reading identical inputs. Likely causes:')
        print('  - "full" config is NOT passing --stage3 to stage4_judge.py '
              '(graph context is missing from the prompt)')
        print('  - Two configs accidentally point at the same stage4_judgments.json')
        print('  - The orchestrator is re-using the "full" stage4 output for '
              'multiple config names')

    print()
    # -----------------------------------------------------------------
    # Graph usage signal.
    # "full" and "dung_no_agents" and "two_agents" should show used_graph=True
    # and should cite graph-semantic terms in rationales.
    # "direct_judge" and "targeted_attacks" should NOT use graph.
    # -----------------------------------------------------------------
    print('=' * 110)
    print('GRAPH USAGE CHECK')
    print('=' * 110)
    print(f'{"Config":<20} {"Expected":<10} {"used_graph flag":<16} '
          f'{"cites graph %":<14}  Verdict')
    print('-' * 90)
    expected_graph = {
        'single_llm': False, 'cot': False, 'direct_judge': False,
        'two_agents': True, 'six_agents': True,
        'targeted_attacks': False, 'dung_no_agents': True, 'full': True,
    }
    for name in ABLATION_NAMES:
        r = results.get(name)
        if not r:
            continue
        exp = expected_graph[name]
        exp_s = 'GRAPH' if exp else 'no graph'
        flag_s = str(r['used_graph_flag'])
        mentions_s = f'{r["mentions_graph_pct"]}%'
        # Red-flag if expected GRAPH but flag is False, or if expected no graph
        # but >20% of rationales cite grounded/preferred extensions
        if exp and not r['used_graph_flag']:
            verdict = '\033[1;31mBUG: expected graph but flag=False\033[0m'
        elif not exp and r['used_graph_flag']:
            verdict = '\033[1;31mBUG: expected no graph but flag=True\033[0m'
        elif exp and r['mentions_graph_pct'] < 20 and r['n_judgments'] >= 5:
            verdict = ('\033[1;33mWARNING: graph config rarely cites graph terms '
                       '(graph prompt may not be effective)\033[0m')
        else:
            verdict = '\033[1;32mOK\033[0m'
        print(f'{name:<20} {exp_s:<10} {flag_s:<16} {mentions_s:<14}  {verdict}')

    print()
    if suspicious:
        print(f'\033[1;33m{len(suspicious)} suspicious config(s):\033[0m')
        for kind, name, reason in suspicious:
            print(f'  [{kind}] {name}: {reason}')
        print()
        print('Recommended fixes before launching 500-topic run:')
        for kind, name, _ in suspicious:
            if kind == 'no_scorable':
                p = stage4_path(args.split, name)
                print(f'  rm {p}   # then re-run with --configs {name} --force-stage4')
            elif kind in ('high_parse_fail', 'low_confidence'):
                print(f'  # {name}: ensure stage4_judge.py has --max-tokens 500+, '
                      f'then --force-stage4 --configs {name}')
    else:
        print('\033[1;32mAll configs look healthy. Safe to kick off the full run.\033[0m')


if __name__ == '__main__':
    main()