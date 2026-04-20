#!/usr/bin/env python3
"""
Build Google Forms content for human verdict evaluation (MAJ-Debate).

Specialized for:
  - Verdict-only judgment (no pairwise comparison)
  - Mixed sampling: 5 DDO topics + 5 logic_test topics
  - Informative topic selection:
      DDO: prefer topics where pipeline and single_llm DISAGREE
      Logic: prefer topics where pipeline was WRONG
      Stratified across categories where possible

Output total: 10 topics, ~15-20 min per annotator.

Usage:
    python scripts/build_human_form_v2.py
        [--n-ddo 5] [--n-logic 5] [--seed 42]

Produces data/eval/google_form/:
    - google_form_content.md       copy-paste into Google Forms
    - form_topics.jsonl            sampled topics with gold labels for scoring
    - form_instructions.md         description for the form
    - score_form_responses.py      scorer (after CSV comes back)
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


ROOT = find_project_root()


def stage1_path(split, config='full'):
    if config in ('full', 'six_agents', 'direct_judge', 'targeted_attacks'):
        return ROOT / 'outputs/stage1' / split / 'stage1_arguments.json'
    return ROOT / 'outputs/stage1' / f'{split}_{config}' / 'stage1_arguments.json'


def stage4_path(split, config):
    if config == 'full':
        return ROOT / 'outputs/stage4' / split / 'stage4_judgments.json'
    return ROOT / 'outputs/stage4' / f'{split}_{config}' / 'stage4_judgments.json'


def load_stage4_verdicts(split, config):
    """Return {topic_id: verdict_str} for a config's stage 4 output."""
    p = stage4_path(split, config)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {j['topic_id']: j.get('verdict') for j in doc.get('judgments', [])}


def top_args(arguments, stance, n=3):
    """Pick n arguments for a stance, prefer round 1."""
    r1 = [a for a in arguments if a.get('stance') == stance and a.get('round') == 1]
    r2 = [a for a in arguments if a.get('stance') == stance and a.get('round') == 2]
    picked = r1[:n]
    if len(picked) < n:
        picked += r2[: n - len(picked)]
    if len(picked) < n:
        rest = [a for a in arguments if a.get('stance') == stance and a not in picked]
        picked += rest[: n - len(picked)]
    return picked


# ----------------------------------------------------------------------------
# DDO sampling: prefer topics where pipeline and baseline disagree
# ----------------------------------------------------------------------------

def sample_ddo_topics(n, seed, min_disagreements_ok=True):
    """Sample n DDO topics, preferring ones where 'full' and 'single_llm' give
    different verdicts (higher-signal for human evaluation). If not enough
    disagreements exist, fall back to stratified random."""
    split = 'ddo_sample'
    s1 = stage1_path(split)
    if not s1.exists():
        raise SystemExit(f'DDO stage 1 missing: {s1}\n'
                         f'Re-run the DDO pipeline first.')
    stage1 = json.loads(s1.read_text())
    all_topics = stage1.get('topics', [])
    topics_by_id = {t['topic_id']: t for t in all_topics}
    print(f'DDO: {len(all_topics)} topics available')

    full_verdicts = load_stage4_verdicts(split, 'full')
    base_verdicts = load_stage4_verdicts(split, 'single_llm')

    if not full_verdicts or not base_verdicts:
        print('  WARNING: stage 4 outputs missing for full/single_llm — '
              'falling back to uniform random sampling')
        disagreement_ids = []
        agreement_ids = list(topics_by_id.keys())
    else:
        disagreement_ids = [tid for tid in topics_by_id
                            if full_verdicts.get(tid) and base_verdicts.get(tid)
                            and full_verdicts[tid] != base_verdicts[tid]]
        agreement_ids = [tid for tid in topics_by_id
                         if tid not in set(disagreement_ids)]
        print(f'  Full vs single_llm disagree on {len(disagreement_ids)}/'
              f'{len(topics_by_id)} topics')

    rng = random.Random(seed)

    # Stratify by domain to avoid all-policy or all-society sample
    def by_domain(ids):
        out = defaultdict(list)
        for tid in ids:
            t = topics_by_id[tid]
            out[t.get('domain', 'other')].append(tid)
        return out

    disagree_by_dom = by_domain(disagreement_ids)
    agree_by_dom = by_domain(agreement_ids)

    # Also stratify by gold label to get PRO/CON balance
    picked = []
    pool_priority = [disagree_by_dom, agree_by_dom]  # pick disagreements first
    domains_available = sorted(set(list(disagree_by_dom.keys()) +
                                   list(agree_by_dom.keys())))

    attempts = 0
    pro_count = 0
    con_count = 0
    while len(picked) < n and attempts < 100:
        attempts += 1
        for domain in domains_available:
            if len(picked) >= n:
                break
            for pool in pool_priority:
                if domain not in pool or not pool[domain]:
                    continue
                # Prefer the under-represented gold label
                candidates = pool[domain][:]
                rng.shuffle(candidates)
                want = 'CON' if pro_count > con_count else 'PRO'
                chosen = None
                for cand_id in candidates:
                    lbl = topics_by_id[cand_id].get('benchmark_label')
                    if lbl == want:
                        chosen = cand_id
                        break
                if chosen is None and candidates:
                    chosen = candidates[0]
                if chosen is not None:
                    picked.append(chosen)
                    pool[domain].remove(chosen)
                    lbl = topics_by_id[chosen].get('benchmark_label')
                    if lbl == 'PRO':
                        pro_count += 1
                    elif lbl == 'CON':
                        con_count += 1
                    break

    print(f'  Picked {len(picked)} DDO topics (PRO={pro_count}, CON={con_count})')
    return [topics_by_id[tid] for tid in picked]


# ----------------------------------------------------------------------------
# Logic_test sampling: prefer topics where full config was wrong
# ----------------------------------------------------------------------------

def sample_logic_topics(n, seed):
    """Sample n logic_test topics, stratified across the 5 categories, preferring
    topics where the full pipeline got it WRONG (those are the informative cases
    for understanding where the pipeline fails)."""
    split = 'logic_test'
    s1 = stage1_path(split)
    if not s1.exists():
        raise SystemExit(f'logic_test stage 1 missing: {s1}\n'
                         f'Re-run the logic_test pipeline first.')
    stage1 = json.loads(s1.read_text())
    all_topics = stage1.get('topics', [])
    topics_by_id = {t['topic_id']: t for t in all_topics}
    print(f'Logic_test: {len(all_topics)} topics available')

    full_verdicts = load_stage4_verdicts(split, 'full')

    # Bucket by domain (syllogism/paradox/fallacy/math_fact/empirical_fact)
    by_domain = defaultdict(list)
    for tid, t in topics_by_id.items():
        by_domain[t.get('domain', 'other')].append(tid)

    domains = sorted(by_domain.keys())
    per_domain = max(1, n // len(domains))  # how many per category
    leftover = n - per_domain * len(domains)

    rng = random.Random(seed + 1)
    picked = []

    for domain in domains:
        candidates = by_domain[domain][:]
        rng.shuffle(candidates)

        # Split into pipeline-wrong and pipeline-right
        wrong = []
        right = []
        for cand_id in candidates:
            gold = topics_by_id[cand_id].get('benchmark_label')
            pred = full_verdicts.get(cand_id)
            if gold is None or pred is None:
                right.append(cand_id)
                continue
            if pred == gold:
                right.append(cand_id)
            else:
                wrong.append(cand_id)

        # Prefer wrong ones
        ordered = wrong + right
        picked.extend(ordered[:per_domain])

    # Allocate leftover to domains that still have candidates
    if leftover > 0:
        remaining = []
        for domain in domains:
            remaining_in_dom = [tid for tid in by_domain[domain] if tid not in picked]
            remaining.extend(remaining_in_dom)
        rng.shuffle(remaining)
        picked.extend(remaining[:leftover])

    picked = picked[:n]

    # Report what was picked
    wrong_count = 0
    for tid in picked:
        gold = topics_by_id[tid].get('benchmark_label')
        pred = full_verdicts.get(tid)
        if gold and pred and pred != gold:
            wrong_count += 1
    print(f'  Picked {len(picked)} logic_test topics '
          f'({wrong_count} where full pipeline was wrong)')
    return [topics_by_id[tid] for tid in picked]


# ----------------------------------------------------------------------------
# Build form content
# ----------------------------------------------------------------------------

def build_form_content(sampled, max_arg_chars=350):
    lines = []
    lines.append('# MAJ-Debate Human Evaluation — Google Form Content')
    lines.append('')
    lines.append(f'{len(sampled)} topics total. Verdict-only judgment. '
                 f'Estimated time: {len(sampled) * 90 // 60} minutes per annotator.')
    lines.append('')
    lines.append('## How to build the Google Form')
    lines.append('')
    lines.append('1. Create a new Google Form. Title: "MAJ-Debate Human Evaluation"')
    lines.append('2. Paste the content of `form_instructions.md` into the form description.')
    lines.append('3. In Settings > Responses, **turn ON "Collect email addresses"** so you can '
                 'track per-annotator responses.')
    lines.append('4. For EACH of the 10 topics below:')
    lines.append('   a. Click "Add section" (bottom right of current section) to create a new page.')
    lines.append('   b. Name the section `Topic N of 10` (just a visual cue for annotators).')
    lines.append('   c. Add a "Title and description" block and paste the topic text block.')
    lines.append('   d. Add a Multiple-choice question with the verdict question text.')
    lines.append('   e. Add a Linear-scale question (1-5) for confidence.')
    lines.append('   f. **IMPORTANT:** keep the `[topic_id: LOGIC_001]` marker in the question '
                 'title — the scorer extracts it with regex.')
    lines.append('5. When done, copy the form link and send to annotators.')
    lines.append('')
    lines.append('---')
    lines.append('')

    for idx, topic in enumerate(sampled, 1):
        tid = topic['topic_id']
        topic_text = topic['topic_text']
        domain = topic.get('domain', '')
        source = 'DDO' if tid.startswith('DDO') else 'Logic'

        pros = top_args(topic['arguments'], 'PRO', 3)
        cons = top_args(topic['arguments'], 'CON', 3)

        lines.append(f'## Topic {idx}/{len(sampled)} — `{tid}`  '
                     f'_({source}, domain: {domain})_')
        lines.append('')
        lines.append('### Step 1: Paste this as a "Title and description" block')
        lines.append('')
        lines.append('**Title field:**')
        lines.append('```')
        lines.append(f'Topic {idx} of {len(sampled)}')
        lines.append('```')
        lines.append('')
        lines.append('**Description field:**')
        lines.append('```')
        lines.append(f'Resolution: {topic_text}')
        lines.append('')
        lines.append('ARGUMENTS FOR (PRO):')
        for i, a in enumerate(pros, 1):
            txt = a.get('text', '')[:max_arg_chars].replace('\n', ' ').strip()
            lines.append(f'  {i}. {txt}')
        lines.append('')
        lines.append('ARGUMENTS AGAINST (CON):')
        for i, a in enumerate(cons, 1):
            txt = a.get('text', '')[:max_arg_chars].replace('\n', ' ').strip()
            lines.append(f'  {i}. {txt}')
        lines.append('```')
        lines.append('')
        lines.append('### Step 2: Add Multiple-choice question (Required)')
        lines.append('')
        lines.append('**Question title:**')
        lines.append('```')
        lines.append(f'Who wins this debate? [topic_id: {tid}]')
        lines.append('```')
        lines.append('')
        lines.append('**Description (optional but helpful):**')
        lines.append('```')
        lines.append('Pick the side whose arguments are logically stronger. Judge on')
        lines.append('reasoning quality and evidence, not personal agreement.')
        lines.append('```')
        lines.append('')
        lines.append('**Options:**')
        lines.append('```')
        lines.append('PRO (the resolution is correct)')
        lines.append('CON (the resolution is wrong)')
        lines.append('TIE (arguments genuinely balanced)')
        lines.append('```')
        lines.append('')
        lines.append('### Step 3: Add Linear-scale question (Required)')
        lines.append('')
        lines.append('**Question title:**')
        lines.append('```')
        lines.append(f'How confident are you? [topic_id: {tid}]')
        lines.append('```')
        lines.append('')
        lines.append('**Scale: 1 to 5**')
        lines.append('- Left label: Not confident')
        lines.append('- Right label: Very confident')
        lines.append('')
        lines.append('---')
        lines.append('')

    return '\n'.join(lines)


def build_instructions():
    return '''Thank you for helping evaluate a research project on AI-driven debate judging.

You'll read 20 short debate topics. For each topic, we show:
- A resolution (the statement being debated)
- 3 arguments FOR the resolution (PRO side)
- 3 arguments AGAINST the resolution (CON side)

Your task for each topic:
1. Decide which side made the stronger case — PRO, CON, or TIE
2. Rate your confidence on a 1-5 scale

Important: Judge the arguments as presented, NOT whether you personally agree
with the resolution. We want to know which side argued better, not which side
you support. If both sides argued badly or the topic is genuinely balanced,
choose TIE.

Estimated time: 25-30 minutes. If you need a break, Google Forms will save
your progress as long as you stay signed into Google — you can close the tab
and return later to finish.

Your responses are anonymous (your email is collected only so we can track
multiple submissions from the same person, not to identify you publicly).

There are no right or wrong answers — your judgment is the data we need.
'''


def build_scorer():
    return '''#!/usr/bin/env python3
"""
Score Google Forms CSV responses for the MAJ-Debate human evaluation.

Usage:
    # After collecting CSV from Google Forms > Responses > Download CSV
    python scripts/score_form_responses.py \\
        --responses-csv ~/Downloads/form_responses.csv \\
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
    m = re.search(r"topic_id:\\s*([A-Z_]+_?\\d+)", col_name)
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
    for line in Path(args.topics).read_text().strip().split('\\n'):
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
    scorecard.write_text('\\n'.join(lines))
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
    print('\\n'.join(lines))
    print()
    print(f'Wrote: {scorecard}')


if __name__ == '__main__':
    main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-ddo', type=int, default=10,
                    help='number of DDO topics to sample (default 10)')
    ap.add_argument('--n-logic', type=int, default=10,
                    help='number of logic_test topics (default 10)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='data/eval/google_form')
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample
    print('Sampling DDO topics (preferring disagreements between full and single_llm)...')
    ddo_topics = sample_ddo_topics(args.n_ddo, args.seed)
    print()
    print('Sampling logic_test topics (preferring pipeline failures, stratified by category)...')
    logic_topics = sample_logic_topics(args.n_logic, args.seed)
    print()

    # Interleave DDO and logic topics so annotator doesn't hit a block of one type
    all_topics = []
    mixed = []
    max_len = max(len(ddo_topics), len(logic_topics))
    for i in range(max_len):
        if i < len(ddo_topics):
            mixed.append(ddo_topics[i])
        if i < len(logic_topics):
            mixed.append(logic_topics[i])
    all_topics = mixed

    # Write form content
    content = build_form_content(all_topics)
    content_path = out_dir / 'google_form_content.md'
    content_path.write_text(content)

    # Write topic manifest (with gold labels, for scoring)
    topics_path = out_dir / 'form_topics.jsonl'
    with open(topics_path, 'w') as f:
        for t in all_topics:
            f.write(json.dumps({
                'topic_id': t['topic_id'],
                'topic_text': t['topic_text'],
                'domain': t.get('domain'),
                'benchmark_label': t.get('benchmark_label'),
            }) + '\n')

    # Write instructions
    (out_dir / 'form_instructions.md').write_text(build_instructions())

    # Write scorer
    scorer_path = ROOT / 'scripts' / 'score_form_responses.py'
    scorer_path.write_text(build_scorer())

    print('Done.')
    print()
    print('Wrote:')
    print(f'  {content_path}')
    print(f'  {topics_path}')
    print(f'  {out_dir / "form_instructions.md"}')
    print(f'  {scorer_path}')
    print()
    print('Topics sampled:')
    for t in all_topics:
        src = 'DDO' if t['topic_id'].startswith('DDO') else 'Logic'
        print(f'  [{src:<5}] {t["topic_id"]:<14} ({t.get("domain", "")}) '
              f'gold={t.get("benchmark_label", "?")}')
    print()
    print('Next:')
    print('  1. Open google_form_content.md')
    print('  2. Build the form manually (takes ~25 min; instructions in the file)')
    print('  3. Collect responses, download CSV from Google Forms')
    print('  4. Run: python scripts/score_form_responses.py --responses-csv <csv>')


if __name__ == '__main__':
    main()
