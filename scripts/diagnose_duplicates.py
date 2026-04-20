#!/usr/bin/env python3
"""
Diagnose why direct_judge, targeted_attacks, and full produce identical rationales.

Prints, for the same topic across all 3 configs:
  - The full rationale
  - The raw_output_preview (first 200 chars of the model's actual output)
  - The used_graph flag
  - The source stage4 path (to prove we're not reading the same file)
  - The file's mtime and size (are they actually different files?)
  - The first 300 characters of the prompt that was sent (inferred from topic + stage3_path presence)

Usage:
    python scripts/diagnose_duplicates.py --split ddo_sample
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def stage4_path(split, name):
    if name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage4' / split / 'stage4_judgments.json'
    return PROJECT_ROOT / 'outputs' / 'stage4' / f'{split}_{name}' / 'stage4_judgments.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='ddo_sample')
    ap.add_argument('--configs', default='direct_judge,targeted_attacks,full',
                    help='comma-separated configs to compare')
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(',') if c.strip()]

    # ---- Step 1: prove each file actually exists and is distinct ----
    print('=' * 80)
    print('STEP 1: Are the output files actually distinct on disk?')
    print('=' * 80)
    print(f'{"Config":<20} {"Size":<10} {"mtime":<22} {"MD5 of content":<35}')
    print('-' * 90)
    file_hashes = {}
    for name in configs:
        p = stage4_path(args.split, name)
        if not p.exists():
            print(f'{name:<20} MISSING -> {p}')
            continue
        st = p.stat()
        content = p.read_bytes()
        h = hashlib.md5(content).hexdigest()
        file_hashes[name] = h
        size = f'{st.st_size} B'
        import datetime
        mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f'{name:<20} {size:<10} {mtime:<22} {h}')
    if len(set(file_hashes.values())) == 1 and len(file_hashes) > 1:
        print()
        print('\033[1;31mDIAGNOSIS: all 3 files have the same MD5 hash.\033[0m')
        print('They are byte-identical — possibly symlinked or the same file was copied.')
    elif len(set(file_hashes.values())) == len(file_hashes):
        print()
        print('\033[1;32mFiles are distinct on disk (different MD5).\033[0m')
    else:
        print()
        print('\033[1;33mSome files share content; others differ.\033[0m')

    # ---- Step 2: dump per-topic side-by-side judgment content ----
    print()
    print('=' * 80)
    print('STEP 2: Per-topic judgment comparison')
    print('=' * 80)

    data = {}
    for name in configs:
        p = stage4_path(args.split, name)
        if p.exists():
            data[name] = json.loads(p.read_text())

    # Build topic index
    if not data:
        return
    topic_ids = set()
    for name, doc in data.items():
        for j in doc.get('judgments', []):
            topic_ids.add(j.get('topic_id'))
    topic_ids = sorted(t for t in topic_ids if t)

    for tid in topic_ids:
        print(f'\n--- Topic {tid} ---')
        print(f'{"Config":<20} {"used_graph":<12} {"verdict":<8} {"conf":<6} Rationale (first 150 chars)')
        print('-' * 120)
        for name in configs:
            doc = data.get(name)
            if not doc:
                continue
            j = next((x for x in doc.get('judgments', [])
                      if x.get('topic_id') == tid), None)
            if not j:
                print(f'{name:<20} (no judgment for this topic)')
                continue
            used_graph = str(j.get('used_graph', 'UNSET'))
            verdict = j.get('verdict', '?')
            conf = j.get('confidence', 0.0)
            rat = (j.get('rationale') or '')[:150].replace('\n', ' ')
            print(f'{name:<20} {used_graph:<12} {verdict:<8} {conf:<6.2f} {rat}')

        # Show the raw_output_preview too if available
        print()
        print(f'  raw_output_preview (first 200 chars of the model\'s actual output):')
        for name in configs:
            doc = data.get(name)
            if not doc:
                continue
            j = next((x for x in doc.get('judgments', [])
                      if x.get('topic_id') == tid), None)
            if j:
                raw = (j.get('raw_output_preview') or '')[:200].replace('\n', ' ')
                print(f'    {name:<20}: {raw}')

    # ---- Step 3: check the stage 3 dependency ----
    print()
    print('=' * 80)
    print('STEP 3: Does `full` config have a stage3 graph output to read from?')
    print('=' * 80)
    stage3_full = PROJECT_ROOT / 'outputs' / 'stage3' / args.split / 'stage3_graphs.json'
    if stage3_full.exists():
        g = json.loads(stage3_full.read_text())
        n_graphs = len(g.get('graphs', []))
        st = stage3_full.stat()
        import datetime
        mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f'  EXISTS: {stage3_full}')
        print(f'    mtime: {mtime}')
        print(f'    graphs: {n_graphs}')
        print(f'    first graph verdict: {g["graphs"][0].get("graph_verdict") if n_graphs else "(none)"}')
    else:
        print(f'\033[1;31m  MISSING: {stage3_full}\033[0m')
        print('  This means `full` stage4 could not have read graph context '
              'because the graph file does not exist.')

    # ---- Step 4: explicit check — is the per-judgment used_graph flag consistent? ----
    print()
    print('=' * 80)
    print('STEP 4: Per-judgment used_graph flag consistency')
    print('=' * 80)
    print('If `full` prompts actually included graph blocks, every judgment in')
    print('outputs/stage4/.../stage4_judgments.json should have used_graph=True.\n')
    for name in configs:
        doc = data.get(name)
        if not doc:
            continue
        flags = [j.get('used_graph') for j in doc.get('judgments', [])]
        true_count = sum(1 for f in flags if f is True)
        false_count = sum(1 for f in flags if f is False)
        unset_count = sum(1 for f in flags if f is None)
        print(f'  {name:<20}  used_graph: True={true_count}, False={false_count}, '
              f'None={unset_count}')

    print()
    print('=' * 80)
    print('INTERPRETATION')
    print('=' * 80)
    print('If `full` has `used_graph=True` for all 5 judgments AND the rationales')
    print('are still identical to `direct_judge` (used_graph=False), then the bug')
    print('is that the graph block is being inserted but the model ignores it.')
    print()
    print('If `full` has `used_graph=False` for any judgment, then the --stage3 flag')
    print('is not propagating through the stage4_judge subprocess call.')
    print()
    print('If all 3 stage4_judgments.json files have the same MD5 hash, the')
    print('orchestrator copied a single output across all 3 paths.')


if __name__ == '__main__':
    main()
