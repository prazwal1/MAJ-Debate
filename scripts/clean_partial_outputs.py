#!/usr/bin/env python3
"""
Pre-run cleanup for ablation runs.

Deletes stage 1/2/3/4 outputs that do NOT cover the expected topic count.
Useful when switching from a smoke test (--topic-limit 5) to a full run
(no --topic-limit) — the orchestrator can't tell these apart on its own,
so it would skip 5-topic files as "done" and never regenerate them.

Rules:
  - Always deletes all stage 4 outputs (they're cheap to regenerate and
    may have been produced with outdated prompts / stage 3)
  - Deletes stage 1 / 2 / 3 ONLY if the file covers fewer topics than
    --expected
  - Keeps stage 1 / 2 / 3 that already have the expected topic count

Usage:
    # See what would be deleted (safe, dry run):
    python scripts/clean_partial_outputs.py --split ddo_sample --expected 500

    # Actually delete:
    python scripts/clean_partial_outputs.py --split ddo_sample --expected 500 --yes

    # Nuclear option (delete all stage 1-4 for this split):
    python scripts/clean_partial_outputs.py --split ddo_sample --all --yes
"""

import argparse
import json
import sys
from pathlib import Path


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


# Mirror the path layout from run_all_ablations.py
CONFIGS = [
    ('single_llm',        False, False, False, True),   # baseline: stage4 only
    ('cot',               False, False, False, True),
    ('direct_judge',      True,  True,  False, True),   # uses shared full stage1+2
    ('two_agents',        True,  True,  True,  True),   # own stage 1+2+3
    ('six_agents',        True,  True,  True,  True),   # own stage 3, shares 1+2
    ('targeted_attacks',  True,  True,  False, True),   # shares full stage1+2
    ('dung_no_agents',    True,  True,  True,  True),   # own stage 1+2+3
    ('full',              True,  True,  True,  True),
]


def path_for_stage(stage, split, cfg_name):
    """Mirror the path rules in run_all_ablations.py."""
    shared_full = cfg_name in ('full', 'six_agents', 'direct_judge', 'targeted_attacks')

    if stage == 1:
        if shared_full:
            return PROJECT_ROOT / 'outputs/stage1' / split / 'stage1_arguments.json'
        return PROJECT_ROOT / 'outputs/stage1' / f'{split}_{cfg_name}' / 'stage1_arguments.json'
    if stage == 2:
        if shared_full:
            return PROJECT_ROOT / 'outputs/stage2' / split / 'stage2_relations.json'
        return PROJECT_ROOT / 'outputs/stage2' / f'{split}_{cfg_name}' / 'stage2_relations.json'
    if stage == 3:
        if cfg_name == 'full':
            return PROJECT_ROOT / 'outputs/stage3' / split / 'stage3_graphs.json'
        return PROJECT_ROOT / 'outputs/stage3' / f'{split}_{cfg_name}' / 'stage3_graphs.json'
    if stage == 4:
        if cfg_name == 'full':
            return PROJECT_ROOT / 'outputs/stage4' / split / 'stage4_judgments.json'
        return PROJECT_ROOT / 'outputs/stage4' / f'{split}_{cfg_name}' / 'stage4_judgments.json'


def count_records(path, stage):
    """Return (count, key) for the record collection in this stage's file."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception:
        return None, None
    if stage == 1 or stage == 2:
        return len(doc.get('topics', [])), 'topics'
    if stage == 3:
        return len(doc.get('graphs', [])), 'graphs'
    if stage == 4:
        return len(doc.get('judgments', [])), 'judgments'
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='ddo_sample')
    ap.add_argument('--expected', type=int, default=500,
                    help='expected topic count for the upcoming run')
    ap.add_argument('--stage4-always', action='store_true', default=True,
                    help='always delete stage 4 outputs regardless of size '
                         '(default True — stage 4 is cheap to regenerate)')
    ap.add_argument('--keep-stage4', action='store_true',
                    help='do NOT delete stage 4 outputs (overrides default)')
    ap.add_argument('--all', action='store_true',
                    help='nuclear option: delete all stage 1-4 for this split')
    ap.add_argument('--yes', '-y', action='store_true',
                    help='actually delete (default is dry-run)')
    args = ap.parse_args()

    to_delete = []
    to_keep = []

    for cfg_name, has_s1, has_s2, has_s3, has_s4 in CONFIGS:
        for stage in [1, 2, 3, 4]:
            if stage == 1 and not has_s1: continue
            if stage == 2 and not has_s2: continue
            if stage == 3 and not has_s3: continue
            if stage == 4 and not has_s4: continue

            p = path_for_stage(stage, args.split, cfg_name)
            if not p.exists():
                continue

            n, key = count_records(p, stage)

            reason = None
            if args.all:
                reason = 'nuclear (--all)'
            elif stage == 4 and not args.keep_stage4:
                reason = 'stage 4 always regenerated'
            elif n is None:
                reason = 'unreadable'
            elif n < args.expected:
                reason = f'{n} {key} < expected {args.expected}'

            if reason:
                to_delete.append((p, stage, cfg_name, n, key, reason))
            else:
                to_keep.append((p, stage, cfg_name, n, key))

    # Deduplicate (shared paths referenced by multiple configs)
    seen = set()
    uniq_delete = []
    for item in to_delete:
        if item[0] in seen:
            continue
        seen.add(item[0])
        uniq_delete.append(item)
    seen = set()
    uniq_keep = []
    for item in to_keep:
        if item[0] in seen:
            continue
        seen.add(item[0])
        uniq_keep.append(item)

    print(f'Project root : {PROJECT_ROOT}')
    print(f'Split        : {args.split}')
    print(f'Expected     : {args.expected} topics per file')
    print(f'Mode         : {"DELETE" if args.yes else "DRY RUN (use --yes to actually delete)"}')
    print()

    if uniq_keep:
        print(f'Keeping {len(uniq_keep)} file(s) that already cover {args.expected} topics:')
        for p, stage, cfg, n, key in uniq_keep:
            rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
            print(f'  KEEP  stage{stage}  [{cfg:<18}]  {n} {key}  {rel}')
        print()

    if not uniq_delete:
        print('Nothing to delete.')
        return

    print(f'Will delete {len(uniq_delete)} file(s):')
    for p, stage, cfg, n, key, reason in uniq_delete:
        rel = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p
        n_str = f'{n} {key}' if n is not None else 'unreadable'
        print(f'  DEL   stage{stage}  [{cfg:<18}]  {n_str:<25} {reason}')
        print(f'        {rel}')

    if not args.yes:
        print()
        print('DRY RUN. Re-run with --yes to actually delete.')
        return

    print()
    deleted = 0
    for p, *_ in uniq_delete:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
        except Exception as ex:
            print(f'  could not delete {p}: {ex}')

    # Also clean up any stray shard log files in the deleted dirs
    for p, stage, *_ in uniq_delete:
        if stage == 2:
            # stage 2 dir may have _shard_*.json siblings
            for shard in p.parent.glob('_shard_*.json'):
                try:
                    shard.unlink()
                except Exception:
                    pass
            for shard in p.parent.glob('_shard_*.log'):
                try:
                    shard.unlink()
                except Exception:
                    pass

    print(f'Deleted {deleted} file(s).')
    print()
    print('Next steps:')
    print(f'  ~/env-vllm/bin/python scripts/cleanup_gpus.py --kill -v')
    print(f'  ~/env-vllm/bin/python scripts/run_all_ablations.py --split {args.split}')


if __name__ == '__main__':
    main()
