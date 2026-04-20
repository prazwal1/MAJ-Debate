#!/usr/bin/env python3
"""
Build annotation CSV sheets for the 50-topic human evaluation.

Produces one CSV per annotator with:
  - Topic ID, topic text, domain
  - The PRO side's top 3 arguments (from stage 1 output)
  - The CON side's top 3 arguments
  - Blank columns for the annotator to fill:
      * winner_pick              (PRO / CON / TIE)
      * winner_persuasiveness    (1-5 Likert)
      * correctness_confidence   (1-5 how sure they are)
      * notes                    (optional free text)

Why this design:
  - Annotators see the SAME arguments Claude saw, so they're judging the
    debate on the same evidence — not forming independent opinions from scratch.
  - Arguments are ordered neutrally (3 PRO then 3 CON, not by agent persona).
  - We don't show which model/config produced the arguments. All 50 topics
    pull their arguments from the FULL configuration's stage 1 output, so
    every annotator sees an identical view.
  - They don't see Claude's verdict. Correctness eval must be independent.

Usage:
    python scripts/build_annotation_sheet.py \\
        --split human_eval \\
        --source-config full \\
        --n-annotators 3

Outputs go to annotations/human_eval_annotator_{1,2,3}.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def stage1_path(split, config):
    """Mirror run_all_ablations.py path layout."""
    if config in ('full', 'six_agents', 'direct_judge', 'targeted_attacks'):
        return PROJECT_ROOT / 'outputs/stage1' / split / 'stage1_arguments.json'
    return PROJECT_ROOT / 'outputs/stage1' / f'{split}_{config}' / 'stage1_arguments.json'


def pick_top_args(arguments, stance, n=3):
    """Pick the n most representative arguments for a stance.
    Strategy: take round-1 arguments first (initial positions, no counter
    framing), fall back to round-2 if r1 has fewer than n."""
    r1 = [a for a in arguments if a.get('stance') == stance and a.get('round') == 1]
    r2 = [a for a in arguments if a.get('stance') == stance and a.get('round') == 2]
    picked = r1[:n]
    if len(picked) < n:
        picked += r2[: n - len(picked)]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='human_eval',
                    help='evaluation split (must match run_human_eval_ablations.sh)')
    ap.add_argument('--source-config', default='full',
                    help='which config to pull arguments from (default: full, '
                         'since all configs using 6 agents share stage 1 output)')
    ap.add_argument('--n-annotators', type=int, default=3)
    ap.add_argument('--n-args-per-side', type=int, default=3)
    ap.add_argument('--out-dir', default='annotations')
    ap.add_argument('--shuffle-order', action='store_true',
                    help='shuffle topic order per annotator to reduce order bias')
    args = ap.parse_args()

    s1_path = stage1_path(args.split, args.source_config)
    if not s1_path.exists():
        print(f'ERROR: stage 1 output not found: {s1_path}')
        print('Run this first:')
        print(f'    bash scripts/run_human_eval_ablations.sh')
        raise SystemExit(1)

    stage1 = json.loads(s1_path.read_text())
    topics = stage1.get('topics', [])
    print(f'Loaded {len(topics)} topics from {s1_path}')

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the canonical row for each topic (same content for all annotators)
    rows = []
    for t in topics:
        pro = pick_top_args(t['arguments'], 'PRO', args.n_args_per_side)
        con = pick_top_args(t['arguments'], 'CON', args.n_args_per_side)

        row = {
            'topic_id': t['topic_id'],
            'topic_text': t['topic_text'],
            'domain': t.get('domain', ''),
        }
        # PRO arguments
        for i in range(args.n_args_per_side):
            row[f'pro_arg_{i+1}'] = pro[i]['text'] if i < len(pro) else ''
        # CON arguments
        for i in range(args.n_args_per_side):
            row[f'con_arg_{i+1}'] = con[i]['text'] if i < len(con) else ''
        # Blank columns for annotator
        row['winner_pick'] = ''                 # PRO | CON | TIE
        row['winner_persuasiveness'] = ''       # 1-5
        row['correctness_confidence'] = ''      # 1-5
        row['notes'] = ''
        rows.append(row)

    # Write one CSV per annotator
    fieldnames = (['topic_id', 'topic_text', 'domain']
                  + [f'pro_arg_{i+1}' for i in range(args.n_args_per_side)]
                  + [f'con_arg_{i+1}' for i in range(args.n_args_per_side)]
                  + ['winner_pick', 'winner_persuasiveness',
                     'correctness_confidence', 'notes'])

    for a in range(1, args.n_annotators + 1):
        out_path = out_dir / f'{args.split}_annotator_{a}.csv'
        annotator_rows = list(rows)
        if args.shuffle_order:
            rng = random.Random(1000 + a)  # seeded for reproducibility
            rng.shuffle(annotator_rows)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            w.writeheader()
            for r in annotator_rows:
                w.writerow(r)
        print(f'  wrote {out_path} ({len(annotator_rows)} topics)')

    # Also dump a README for annotators alongside the CSVs
    readme_path = out_dir / 'ANNOTATOR_INSTRUCTIONS.md'
    readme_path.write_text(f'''# Annotator Instructions

You have been assigned a set of {len(rows)} debate topics to judge. Each
row in your CSV contains:

- **topic_id** — identifier (do not edit)
- **topic_text** — the debate resolution
- **domain** — policy / ethics / science / society
- **pro_arg_1..3** — three arguments for the PRO side
- **con_arg_1..3** — three arguments for the CON side

Please fill in these columns (one row at a time):

## winner_pick
Which side made the stronger *logical* case? Choose exactly one:
- `PRO` — the PRO arguments were logically stronger
- `CON` — the CON arguments were logically stronger
- `TIE` — genuinely balanced or neither side made a clear case

Judge based on:
  - Are the arguments well-reasoned?
  - Do they address the actual resolution?
  - Do they avoid logical fallacies?
  - Which side has more undefeated claims?

Try NOT to judge based on your personal opinion about the topic. You're judging
the *arguments as presented*, not whether you agree with the resolution.

## winner_persuasiveness (1-5)
How persuasive was the winning side, in absolute terms?
- 1 = Weak, barely argued
- 2 = Basic, unconvincing
- 3 = Reasonable, average-quality debate
- 4 = Strong, well-argued
- 5 = Exceptional, highly persuasive

## correctness_confidence (1-5)
How confident are you that your `winner_pick` is the logically correct call?
- 1 = Very unsure, could easily go either way
- 2 = Leaning but not strongly
- 3 = Moderately confident
- 4 = Confident
- 5 = Very confident

## notes (optional)
Anything you want to flag — "both sides weak", "this topic is ambiguous",
"CON arguments are off-topic", etc.

## Timing
Aim for ~3 minutes per topic. Full set ({len(rows)} topics) should take ~2.5 hours.
Take breaks — annotator fatigue hurts quality.

## Submit
Save your filled CSV with the same filename, and send it back to the team lead.
''')
    print(f'  wrote {readme_path}')

    print()
    print(f'Next steps:')
    print(f'  1. Send each annotator their CSV + ANNOTATOR_INSTRUCTIONS.md')
    print(f'  2. Collect filled CSVs back at {out_dir}/')
    print(f'  3. Run: python scripts/score_human_eval.py --split {args.split}')


if __name__ == '__main__':
    main()
