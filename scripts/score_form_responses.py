#!/usr/bin/env python3
"""
Score Google Forms CSV responses for the MAJ-Debate human evaluation.

Usage:
    # After collecting CSV from Google Forms > Responses > Download CSV
    python scripts/score_form_responses.py \
        --responses-csv ~/Downloads/form_responses.csv \
        --topics data/eval/google_form/form_topics.jsonl

Produces:
    outputs/ablations/human_form/scorecard.md    main results
    outputs/ablations/human_form/raw_votes.json  per-topic vote detail
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


ROOT = find_project_root()


def parse_topic_id(col_name):
    """Extract topic_id marker from column name like 'Who wins? [topic_id: DDO_00123]'."""
    m = re.search(r"topic_id:\s*([A-Z_]+_?\d+)", col_name)
    return m.group(1) if m else None


def load_topic_gold():
    """Load gold verdicts for both splits."""
    gold = {}
    for split in ['ddo_sample', 'logic_test']:
        p = ROOT / f'outputs/stage1/{split}/stage1_arguments.json'
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        for t in doc.get('topics', []):
            gold[t['topic_id']] = {
                'gold': t.get('benchmark_label'),
                'domain': t.get('domain'),
                'split': split,
            }
    return gold


def load_model_verdicts():
    """Load stage4 verdicts for all 8 configs from both splits."""
    configs = ['single_llm', 'cot', 'direct_judge', 'two_agents',
               'six_agents', 'targeted_attacks', 'dung_no_agents', 'full']
    verdicts = defaultdict(dict)  # verdicts[config][topic_id] = 'PRO'|'CON'|'TIE'
    for split in ['ddo_sample', 'logic_test']:
        for c in configs:
            if c == 'full':
                p = ROOT / f'outputs/stage4/{split}/stage4_judgments.json'
            else:
                p = ROOT / f'outputs/stage4/{split}_{c}/stage4_judgments.json'
            if not p.exists():
                continue
            doc = json.loads(p.read_text())
            for j in doc.get('judgments', []):
                verdicts[c][j['topic_id']] = j.get('verdict')
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--responses-csv', required=True)
    ap.add_argument('--topics', default='data/eval/google_form/form_topics.jsonl')
    args = ap.parse_args()

    # Load topic metadata from form_topics file
    form_topic_ids = set()
    for line in Path(args.topics).read_text().strip().split('\n'):
        if line:
            form_topic_ids.add(json.loads(line)['topic_id'])

    gold = load_topic_gold()
    model_verdicts = load_model_verdicts()

    # Parse CSV
    with open(args.responses_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    print(f'Loaded {len(rows)} form responses (annotators)')

    # Bucket by topic
    verdict_by_topic = defaultdict(list)
    conf_by_topic = defaultdict(list)

    for row in rows:
        for col, val in row.items():
            if not val or not val.strip():
                continue
            tid = parse_topic_id(col)
            if not tid:
                continue
            v = val.strip().upper()
            low = col.lower()
            if 'wins' in low or 'who wins' in low:
                if v.startswith('PRO'):
                    verdict_by_topic[tid].append('PRO')
                elif v.startswith('CON'):
                    verdict_by_topic[tid].append('CON')
                elif v.startswith('TIE'):
                    verdict_by_topic[tid].append('TIE')
            elif 'confident' in low:
                try:
                    conf_by_topic[tid].append(int(val.strip()[0]))
                except Exception:
                    pass

    # Build per-topic summary
    topic_results = {}
    for tid in sorted(form_topic_ids):
        votes = verdict_by_topic.get(tid, [])
        confs = conf_by_topic.get(tid, [])
        if not votes:
            topic_results[tid] = {'status': 'no_votes'}
            continue
        majority = Counter(votes).most_common(1)[0][0]
        g = gold.get(tid, {})
        topic_results[tid] = {
            'split': g.get('split'),
            'domain': g.get('domain'),
            'gold': g.get('gold'),
            'n_votes': len(votes),
            'votes': votes,
            'majority': majority,
            'human_correct_vs_gold': (g.get('gold') == majority) if g.get('gold') else None,
            'avg_confidence': round(statistics.mean(confs), 2) if confs else None,
            'model_verdicts': {c: model_verdicts[c].get(tid) for c in model_verdicts
                               if model_verdicts[c].get(tid)},
        }

    # Per-config agreement with human majority
    config_stats = defaultdict(lambda: {'n': 0, 'agree': 0})
    for tid, info in topic_results.items():
        if info.get('status') == 'no_votes':
            continue
        hm = info['majority']
        for c, mv in info.get('model_verdicts', {}).items():
            config_stats[c]['n'] += 1
            if mv == hm:
                config_stats[c]['agree'] += 1

    # Score vs gold (human majority accuracy)
    n_with_gold = sum(1 for r in topic_results.values()
                      if r.get('gold') and r.get('status') != 'no_votes')
    n_human_correct = sum(1 for r in topic_results.values()
                          if r.get('human_correct_vs_gold'))

    # Build scorecard
    lines = ['# Human Form Evaluation Results', '']
    lines.append(f'## Summary')
    lines.append(f'- **Annotators**: {len(rows)}')
    lines.append(f'- **Topics**: {len(form_topic_ids)}')
    lines.append(f'- **Human majority vs gold**: {n_human_correct}/{n_with_gold} '
                 f'({100*n_human_correct/max(n_with_gold,1):.1f}%)')
    lines.append('')

    lines.append('## Model configuration agreement with human majority')
    lines.append('')
    lines.append('| Config | Agree | Total | Agreement % |')
    lines.append('|---|---:|---:|---:|')
    for c in sorted(config_stats):
        s = config_stats[c]
        pct = 100 * s['agree'] / max(s['n'], 1)
        lines.append(f'| {c} | {s["agree"]} | {s["n"]} | {pct:.1f}% |')
    lines.append('')

    lines.append('## Per-topic detail')
    lines.append('')
    lines.append('| Topic | Domain | Gold | Human maj. | Conf. | Full verdict | SingleLLM |')
    lines.append('|---|---|---|---|---:|---|---|')
    for tid, info in sorted(topic_results.items()):
        if info.get('status') == 'no_votes':
            lines.append(f'| {tid} | — | — | NO VOTES | — | — | — |')
            continue
        full_v = info['model_verdicts'].get('full', '—')
        single_v = info['model_verdicts'].get('single_llm', '—')
        lines.append(f'| {tid} | {info.get("domain", "")} | '
                     f'{info.get("gold", "—")} | {info["majority"]} | '
                     f'{info.get("avg_confidence", "—")} | {full_v} | {single_v} |')
    lines.append('')

    out_dir = ROOT / 'outputs/ablations/human_form'
    out_dir.mkdir(parents=True, exist_ok=True)
    scorecard = out_dir / 'scorecard.md'
    scorecard.write_text('\n'.join(lines))
    (out_dir / 'raw_votes.json').write_text(json.dumps({
        'summary': {
            'n_annotators': len(rows),
            'n_topics': len(form_topic_ids),
            'n_human_correct_vs_gold': n_human_correct,
            'n_with_gold': n_with_gold,
            'human_accuracy_pct': round(100*n_human_correct/max(n_with_gold,1), 1),
        },
        'config_agreement_with_human': {
            c: {**s, 'pct': round(100*s['agree']/max(s['n'],1), 1)}
            for c, s in config_stats.items()
        },
        'per_topic': topic_results,
    }, indent=2))

    print()
    print('\n'.join(lines))
    print()
    print(f'Wrote: {scorecard}')


if __name__ == '__main__':
    main()
