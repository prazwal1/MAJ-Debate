#!/usr/bin/env python3
"""
MAJ-Debate — Run All 8 Ablations End-to-End with vLLM (no external APIs).

This is the single entrypoint that completes the proposal's ablation table.
It drives stage1_vllm -> stage2_vllm_shard (your working code) -> stage3_graph
-> stage4_judge across 8 configurations, then aggregates metrics into a final
CSV / JSON that maps 1:1 onto Table "Ablation configurations" in your proposal.

The 8 configurations (see proposal Table "Planned ablation configurations"):
    1. single_llm          - one model, direct verdict from topic only
    2. cot                 - single model, step-by-step then verdict
    3. gpt4_direct_judge   - "strong judge" on argument pairs, no graph.
                             Re-implemented locally with Qwen as strong judge
                             (honest rename printed in outputs: "direct_judge")
    4. two_agents          - 2 agents (1 PRO, 1 CON), targeted attacks, graph
    5. six_agents          - 6 agents (3 PRO, 3 CON), targeted attacks, graph
    6. targeted_attacks    - 6 agents, targeted premise attacks, no graph
    7. dung_no_agents      - 2 agents only, zero-shot attacks, graph
    8. full                - 6 agents + targeted attacks + graph (your run)

Run on Puffer:
    cd ~/Project
    ~/env-vllm/bin/python scripts/run_all_ablations.py \\
        --split ddo_sample \\
        --model ~/models/Qwen2.5-3B-Instruct \\
        --gpus 0,1,2

For a fast smoke test on 5 topics first:
    ~/env-vllm/bin/python scripts/run_all_ablations.py \\
        --split ddo_sample --topic-limit 5 --configs single_llm,cot,full

Configs can be selected/skipped via --configs and --skip-configs. The
orchestrator re-uses your existing `full` run if it is already on disk,
and it re-uses Stage 1 outputs across configurations whenever the agent
count matches.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ----------------------------------------------------------------------------
# Ablation catalogue. Each entry is self-describing so the orchestrator
# does not hard-code assumptions about how to run it.
# ----------------------------------------------------------------------------

ABLATIONS = [
    {
        'name': 'single_llm',
        'label': 'Single-LLM Baseline',
        'runner': 'baseline',           # -> scripts/run_baseline_judge.py
        'baseline_mode': 'single',
        'uses_stage1': False,
        'uses_stage2': False,
        'uses_stage3': False,
    },
    {
        'name': 'cot',
        'label': '+ CoT Baseline',
        'runner': 'baseline',
        'baseline_mode': 'cot',
        'uses_stage1': False,
        'uses_stage2': False,
        'uses_stage3': False,
    },
    {
        'name': 'direct_judge',
        'label': '+ Direct Judge (strong)',  # honest rename of "GPT-4 direct judge"
        'runner': 'direct_judge',       # same relations as stage 2, no graph
        'uses_stage1': True,
        'uses_stage2': True,
        'uses_stage3': False,
        'n_pro': 3, 'n_con': 3,         # inherits full-pipeline stage1/2
        'targeted_attacks': False,      # zero-shot pairwise (contrasts with targeted_attacks config)
    },
    {
        'name': 'two_agents',
        'label': '+ 2 Agents',
        'runner': 'full_pipeline',
        'uses_stage1': True, 'uses_stage2': True, 'uses_stage3': True,
        'n_pro': 1, 'n_con': 1,         # 2 agents total
        'targeted_attacks': True,
    },
    {
        'name': 'six_agents',
        'label': '+ 6 Agents (no graph)',
        # This config measures the effect of agent count alone — it uses 6
        # agents + targeted stage 2 but DOES NOT pass the Dung graph to
        # the judge. 'full' adds graph on top, so full-six_agents now
        # cleanly isolates the graph contribution (RQ1).
        'runner': 'six_agents_no_graph',
        'uses_stage1': True, 'uses_stage2': True, 'uses_stage3': False,
        'n_pro': 3, 'n_con': 3,
        'targeted_attacks': True,
    },
    {
        'name': 'targeted_attacks',
        'label': '+ Targeted Attacks',
        'runner': 'targeted_no_graph',
        'uses_stage1': True, 'uses_stage2': True, 'uses_stage3': False,
        'n_pro': 3, 'n_con': 3,
        'targeted_attacks': True,
    },
    {
        'name': 'dung_no_agents',
        'label': '+ Dung Graph (no agents)',
        'runner': 'full_pipeline',
        'uses_stage1': True, 'uses_stage2': True, 'uses_stage3': True,
        'n_pro': 1, 'n_con': 1,
        'targeted_attacks': False,      # zero-shot pairwise instead
    },
    {
        'name': 'full',
        'label': 'Full (6ag.+targ.+graph)',
        'runner': 'full_pipeline',
        'uses_stage1': True, 'uses_stage2': True, 'uses_stage3': True,
        'n_pro': 3, 'n_con': 3,
        'targeted_attacks': True,
    },
]


# ----------------------------------------------------------------------------
# Pathing / project layout
# ----------------------------------------------------------------------------

def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / 'outputs').exists() or (p / 'scripts').exists() or (p / 'notebooks').exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def stage1_out(split, config_name):
    # The "full" run and "six_agents" share the same generation config
    # (6 agents, targeted stage 2). 'direct_judge' and 'targeted_attacks'
    # also reuse the same stage 1 (arguments are the same, only the
    # relation-labelling prompt style differs in stage 2).
    if config_name in ('full', 'six_agents', 'direct_judge', 'targeted_attacks'):
        return PROJECT_ROOT / 'outputs' / 'stage1' / split / 'stage1_arguments.json'
    return PROJECT_ROOT / 'outputs' / 'stage1' / f'{split}_{config_name}' / 'stage1_arguments.json'


def stage2_out(split, config_name):
    # CRITICAL: direct_judge and targeted_attacks DIFFER in the stage 2
    # prompt style (zero-shot pairwise vs Ozaki targeted). Each must write
    # to its OWN stage 2 output file. Otherwise they collapse into identical
    # runs because whoever ran stage 2 first wins and the other reads the
    # shared file.
    # full and six_agents DO share stage 2 (both use 6 agents + targeted).
    if config_name in ('full', 'six_agents'):
        return PROJECT_ROOT / 'outputs' / 'stage2' / split / 'stage2_relations.json'
    return PROJECT_ROOT / 'outputs' / 'stage2' / f'{split}_{config_name}' / 'stage2_relations.json'


def stage3_out(split, config_name):
    if config_name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage3' / split / 'stage3_graphs.json'
    return PROJECT_ROOT / 'outputs' / 'stage3' / f'{split}_{config_name}' / 'stage3_graphs.json'


def stage4_out(split, config_name):
    if config_name == 'full':
        return PROJECT_ROOT / 'outputs' / 'stage4' / split / 'stage4_judgments.json'
    return PROJECT_ROOT / 'outputs' / 'stage4' / f'{split}_{config_name}' / 'stage4_judgments.json'


def ablation_dir(split):
    d = PROJECT_ROOT / 'outputs' / 'ablations' / split
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------
# Subprocess helpers
# ----------------------------------------------------------------------------

def run(cmd, env=None, log_path=None):
    """Run a subprocess, streaming output, failing loudly on non-zero exit."""
    print(f'\n\033[1;36m[run]\033[0m {" ".join(str(c) for c in cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    with open(log_path, 'a') if log_path else _NullCtx() as lf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if lf:
                lf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'command failed with exit code {rc}: {cmd[0]}')


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ----------------------------------------------------------------------------
# Per-config runners
# ----------------------------------------------------------------------------

def _pick_single_gpu(args, kill_zombies=True, min_free_override=None):
    """Pick the emptiest GPU from args.gpus (strictly within that pool).

    If kill_zombies=True, first scans the user's own python processes on all
    GPUs and SIGTERM/SIGKILLs any that are not this process or its parent.
    This reclaims CUDA memory from previous vLLM crashes.

    No GPU outside args.gpus is ever used (the extra GPU on this box crashes
    the server).
    """
    primary = [int(x) for x in args.gpus.split(',') if x.strip()]
    min_free = min_free_override if min_free_override is not None else args.min_free_mib
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
        from pick_gpu import pick_free_gpu_with_fallback
        picked = pick_free_gpu_with_fallback(
            primary=primary, fallback=None,
            min_free_mib=min_free, kill_zombies=kill_zombies,
            verbose=True,
        )
        if picked is not None:
            print(f'  [pick_gpu] selected GPU {picked} '
                  f'(min_free_mib={min_free}, pool={primary})')
            return str(picked)
        print(f'  [pick_gpu] NO GPU in {primary} meets min_free={min_free} MiB; '
              f'falling back to first listed')
    except Exception as ex:
        print(f'  [pick_gpu] fallback ({ex})')
    return str(primary[0])


def _wait_for_gpu_memory(gpu_id, min_free_mib, max_wait_s=180, check_interval_s=10):
    """Block until GPU `gpu_id` has at least `min_free_mib` free memory,
    or until max_wait_s elapses. Returns the free memory at return time."""
    sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
    try:
        from pick_gpu import query_gpus
    except ImportError:
        return None
    elapsed = 0
    last_free = None
    while elapsed <= max_wait_s:
        gpus = query_gpus()
        last_free = next((f for i, f, _ in gpus if i == int(gpu_id)), None)
        if last_free is None:
            return None
        if last_free >= min_free_mib:
            if elapsed > 0:
                print(f'  [wait_gpu] GPU {gpu_id} now has {last_free} MiB free '
                      f'(waited {elapsed}s)')
            return last_free
        print(f'  [wait_gpu] GPU {gpu_id} has only {last_free} MiB free '
              f'(need {min_free_mib}); waiting {check_interval_s}s... '
              f'[{elapsed}s / {max_wait_s}s]')
        time.sleep(check_interval_s)
        elapsed += check_interval_s
    print(f'  [wait_gpu] timeout; GPU {gpu_id} has {last_free} MiB free '
          f'(wanted {min_free_mib}). Launching anyway.')
    return last_free


def _run_with_gpu_retry(cmd, args, log_path, stage_name='stage'):
    """Run `cmd` with CUDA_VISIBLE_DEVICES set by the picker. On a known
    OOM failure, automatically retry ONCE with:
      - zombie cleanup run again (processes from the failed attempt)
      - a fresh GPU pick
      - waits for memory to actually be available before relaunch
    """
    gpu = _pick_single_gpu(args, kill_zombies=True)
    # Verify memory is actually there before launching
    _wait_for_gpu_memory(gpu, args.min_free_mib)
    try:
        run(cmd, env={'CUDA_VISIBLE_DEVICES': gpu}, log_path=log_path)
        return
    except RuntimeError as ex:
        if args.no_retry_oom:
            raise
        print(f'\n\033[1;33m[retry]\033[0m {stage_name} failed on GPU {gpu}: {ex}')
        print(f'\033[1;33m[retry]\033[0m killing zombies, waiting 10s, re-picking...')
        time.sleep(10)
    # Second attempt with same bar — do NOT lower min_free, because the
    # problem is not picker accuracy but absolute available memory.
    gpu2 = _pick_single_gpu(args, kill_zombies=True)
    _wait_for_gpu_memory(gpu2, args.min_free_mib)
    print(f'\033[1;33m[retry]\033[0m re-launching {stage_name} on GPU {gpu2}')
    run(cmd, env={'CUDA_VISIBLE_DEVICES': gpu2},
        log_path=log_path)


def run_stage1_vllm(args, cfg, log_dir):
    """Generate arguments with vLLM (replaces OpenRouter notebook)."""
    out = stage1_out(args.split, cfg['name'])
    if out.exists() and not args.force_stage1:
        print(f'  [skip stage1] {out} already exists')
        return out
    cmd = [
        args.vllm_python, '-u',
        str(PROJECT_ROOT / 'scripts' / 'stage1_vllm.py'),
        '--split', args.split,
        '--topic-file', str(args.topic_file),
        '--output', str(out),
        '--model', str(args.model),
        '--n-pro', str(cfg['n_pro']),
        '--n-con', str(cfg['n_con']),
        '--r1-args', str(args.r1_args),
        '--r2-args', str(args.r2_args),
        '--topic-limit', str(args.topic_limit),
        '--gpu-mem-util', str(args.gpu_mem_util),
        '--max-model-len', str(args.max_model_len),
    ]
    _run_with_gpu_retry(cmd, args, log_dir / f'{cfg["name"]}_stage1.log',
                         stage_name=f'{cfg["name"]} stage1')
    return out


def run_stage2_dp(args, cfg, log_dir, stage1_path, stage2_path,
                  targeted_attacks=True):
    """Run stage 2 data-parallel across GPUs (re-uses your working code).

    Before launching shards, kill any zombie python GPU processes owned by
    this user so shards don't race against them. On shard failure, retry
    failed shards once with shrunken memory footprint on the same GPU pool
    (no GPU outside args.gpus is ever used — the extra GPU on this box
    crashes the server).
    """
    if stage2_path.exists() and not args.force_stage2:
        print(f'  [skip stage2] {stage2_path} already exists')
        return stage2_path
    stage2_path.parent.mkdir(parents=True, exist_ok=True)

    # Zombie sweep first (this is the big fix)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
        from pick_gpu import kill_zombie_vllm_procs
        killed = kill_zombie_vllm_procs(verbose=True)
        if killed:
            print(f'  [stage2] killed {len(killed)} zombie PID(s) before shard launch')
    except Exception as ex:
        print(f'  [stage2] zombie sweep skipped: {ex}')

    gpu_ids = [int(g) for g in args.gpus.split(',') if g.strip()]
    n_shards = len(gpu_ids)

    procs = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        shard_out = stage2_path.parent / f'_shard_{shard_id:02d}.json'
        shard_log = stage2_path.parent / f'_shard_{shard_id:02d}.log'
        cmd = [
            args.vllm_python, '-u',
            str(PROJECT_ROOT / 'scripts' / 'stage2_vllm_shard.py'),
            '--shard-id', str(shard_id),
            '--total-shards', str(n_shards),
            '--input', str(stage1_path),
            '--output', str(shard_out),
            '--model', str(args.model),
            '--log-file', str(shard_log),
        ]
        env = {
            'CUDA_VISIBLE_DEVICES': str(gpu_id),
            'MAJ_STAGE2_TARGETED': '1' if targeted_attacks else '0',
            'MAJ_STAGE2_TOPIC_LIMIT': str(args.topic_limit),
            'MAJ_STAGE2_GPU_MEM_UTIL': str(args.gpu_mem_util),
            'MAJ_STAGE2_MAX_MODEL_LEN': str(args.max_model_len),
        }
        print(f'  [stage2 shard {shard_id}] GPU {gpu_id} -> {shard_out}')
        procs.append((shard_id, gpu_id, subprocess.Popen(
            [str(c) for c in cmd],
            env={**os.environ, **env},
            stdout=open(log_dir / f'{cfg["name"]}_stage2_shard{shard_id}.log', 'w'),
            stderr=subprocess.STDOUT,
        )))
        if shard_id < n_shards - 1:
            time.sleep(args.launch_stagger)

    failed = []
    for shard_id, gpu_id, p in procs:
        rc = p.wait()
        if rc != 0:
            failed.append((shard_id, gpu_id))
            print(f'  [stage2 shard {shard_id}] FAILED rc={rc} on GPU {gpu_id}')

    if failed and not args.no_retry_oom:
        # Retry failed shards once after zombie sweep, on the SAME GPU pool
        # (no fallback GPU — server crashes above 3), but with smaller
        # memory footprint.
        failed_shard_ids = [s for s, _ in failed]
        print(f'\n\033[1;33m[retry]\033[0m stage 2 shards {failed_shard_ids} failed; '
              f'sweeping zombies and retrying with reduced memory footprint...')
        try:
            from pick_gpu import kill_zombie_vllm_procs
            killed = kill_zombie_vllm_procs(verbose=True)
            if killed:
                print(f'  [retry] killed {len(killed)} zombie PID(s)')
        except Exception:
            pass
        time.sleep(10)

        retry_procs = []
        for shard_id, gpu_id in failed:
            shard_out = stage2_path.parent / f'_shard_{shard_id:02d}.json'
            shard_log = stage2_path.parent / f'_shard_{shard_id:02d}.log'
            cmd = [
                args.vllm_python, '-u',
                str(PROJECT_ROOT / 'scripts' / 'stage2_vllm_shard.py'),
                '--shard-id', str(shard_id),
                '--total-shards', str(n_shards),
                '--input', str(stage1_path),
                '--output', str(shard_out),
                '--model', str(args.model),
                '--log-file', str(shard_log),
            ]
            env = {
                'CUDA_VISIBLE_DEVICES': str(gpu_id),  # SAME GPU as original
                'MAJ_STAGE2_TARGETED': '1' if targeted_attacks else '0',
                'MAJ_STAGE2_TOPIC_LIMIT': str(args.topic_limit),
                # Retry uses a slightly smaller context but NOT a smaller
                # memory budget — dropping below 0.75 leaves insufficient
                # KV cache room for Qwen2.5-3B on an 11 GB GPU after weights.
                'MAJ_STAGE2_GPU_MEM_UTIL': str(max(args.gpu_mem_util, 0.75)),
                'MAJ_STAGE2_MAX_MODEL_LEN': str(min(args.max_model_len, 3072)),
            }
            print(f'  [stage2 retry shard {shard_id}] GPU {gpu_id} (shrink: '
                  f'mem={env["MAJ_STAGE2_GPU_MEM_UTIL"]}, '
                  f'len={env["MAJ_STAGE2_MAX_MODEL_LEN"]})')
            retry_procs.append((shard_id, subprocess.Popen(
                [str(c) for c in cmd],
                env={**os.environ, **env},
                stdout=open(log_dir / f'{cfg["name"]}_stage2_shard{shard_id}_retry.log', 'w'),
                stderr=subprocess.STDOUT,
            )))
            time.sleep(args.launch_stagger)

        failed_final = []
        for shard_id, p in retry_procs:
            rc = p.wait()
            if rc != 0:
                failed_final.append(shard_id)
                print(f'  [stage2 retry shard {shard_id}] FAILED rc={rc}')

        if failed_final:
            raise RuntimeError(f'stage2 shards failed even after retry: {failed_final}')
    elif failed:
        raise RuntimeError(f'stage2 shards failed: {[s for s, _ in failed]}')

    _merge_stage2_shards(stage2_path, n_shards)
    return stage2_path


def _merge_stage2_shards(final_path, n_shards):
    """Same merge logic as your stage2_vllm_dp.py, inlined here."""
    from collections import Counter
    all_topics, seen = [], set()
    for sid in range(n_shards):
        shard_path = final_path.parent / f'_shard_{sid:02d}.json'
        if not shard_path.exists():
            continue
        doc = json.loads(shard_path.read_text())
        for t in doc.get('topics', []):
            tid = t.get('topic_id')
            if tid in seen:
                continue
            seen.add(tid)
            all_topics.append(t)
    all_topics.sort(key=lambda t: t.get('topic_id', ''))

    counts = Counter()
    total_kept = total_rel = 0
    for t in all_topics:
        s = t.get('summary', {})
        total_kept += s.get('kept_relations', 0)
        total_rel += len(t.get('relations', []))
        for lbl, c in (s.get('label_counts') or {}).items():
            counts[lbl] += c

    final = {
        'topics': all_topics,
        'summary': {
            'merged_at': datetime.now().isoformat(timespec='seconds'),
            'n_topics': len(all_topics),
            'n_shards': n_shards,
            'total_relations': total_rel,
            'total_kept_relations': total_kept,
            'label_counts': dict(counts),
        },
    }
    tmp = final_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(final_path)
    print(f'  [merged] {len(all_topics)} topics -> {final_path}')


def run_stage3(args, cfg, log_dir, stage2_path, stage3_path):
    """Stage 3: Dung graph engine. No LLM - pure python.

    Validates any existing stage 3 output before skipping. An existing file
    with 0 graphs (bug from earlier runs) will be overwritten rather than
    silently reused — otherwise downstream stage 4 will run without graph
    context and produce results identical to the no-graph baseline.
    """
    if stage3_path.exists() and not args.force_stage3:
        try:
            with open(stage3_path) as f:
                existing = json.load(f)
            n_graphs = len(existing.get('graphs', []))
            if n_graphs == 0:
                print(f'  [stage3] existing file has 0 graphs; regenerating: '
                      f'{stage3_path}')
            else:
                print(f'  [skip stage3] {stage3_path} already exists '
                      f'({n_graphs} graphs)')
                return stage3_path
        except Exception as ex:
            print(f'  [stage3] existing file unreadable ({ex}); regenerating')
    stage3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.vllm_python, '-u',
        str(PROJECT_ROOT / 'scripts' / 'stage3_graph.py'),
        '--input', str(stage2_path),
        '--output', str(stage3_path),
    ]
    run(cmd, log_path=log_dir / f'{cfg["name"]}_stage3.log')

    # Verify it produced a non-empty output
    try:
        with open(stage3_path) as f:
            produced = json.load(f)
        n = len(produced.get('graphs', []))
        if n == 0:
            raise RuntimeError(
                f'stage 3 ran but produced 0 graphs — stage 2 may have '
                f'produced no attack edges above the confidence threshold. '
                f'Check {stage2_path}')
        print(f'  [stage3] generated {n} graphs')
    except json.JSONDecodeError as ex:
        raise RuntimeError(f'stage 3 output is not valid JSON: {ex}')
    return stage3_path


def run_stage4(args, cfg, log_dir, stage2_path, stage3_path, stage4_path,
               use_graph=True):
    """Stage 4: judge brain. LLM reads graph context + relations."""
    if stage4_path.exists() and not args.force_stage4:
        print(f'  [skip stage4] {stage4_path} already exists')
        return stage4_path
    stage4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.vllm_python, '-u',
        str(PROJECT_ROOT / 'scripts' / 'stage4_judge.py'),
        '--stage2', str(stage2_path),
        '--output', str(stage4_path),
        '--model', str(args.model),
        '--topic-limit', str(args.topic_limit),
        '--gpu-mem-util', str(args.gpu_mem_util),
        '--max-model-len', str(args.max_model_len),
    ]
    if use_graph and stage3_path and stage3_path.exists():
        cmd += ['--stage3', str(stage3_path)]
    _run_with_gpu_retry(cmd, args, log_dir / f'{cfg["name"]}_stage4.log',
                         stage_name=f'{cfg["name"]} stage4')
    return stage4_path


def run_baseline(args, cfg, log_dir):
    """Stages 1-4 bypassed: direct topic -> verdict with vLLM.

    Baselines only ever see a short topic string + a small JSON instruction,
    so they don't need the full 4096-token context or 80% GPU memory. We
    deliberately cut both to reduce OOM risk on Volta/Turing GPUs where
    xFormers peak memory is notoriously high."""
    out = stage4_out(args.split, cfg['name'])
    if out.exists() and not args.force_stage4:
        print(f'  [skip baseline] {out} already exists')
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Override args for the baseline-specific lighter footprint
    baseline_gpu_mem = min(args.gpu_mem_util, args.baseline_gpu_mem_util)
    baseline_max_len = min(args.max_model_len, args.baseline_max_model_len)

    cmd = [
        args.vllm_python, '-u',
        str(PROJECT_ROOT / 'scripts' / 'run_baseline_judge.py'),
        '--topic-file', str(args.topic_file),
        '--split', args.split,
        '--mode', cfg['baseline_mode'],
        '--output', str(out),
        '--model', str(args.model),
        '--topic-limit', str(args.topic_limit),
        '--gpu-mem-util', str(baseline_gpu_mem),
        '--max-model-len', str(baseline_max_len),
    ]
    _run_with_gpu_retry(cmd, args, log_dir / f'{cfg["name"]}.log',
                         stage_name=f'{cfg["name"]} baseline')
    return out


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

def run_config(args, cfg, log_dir):
    name = cfg['name']
    t0 = time.monotonic()
    print(f'\n\033[1;33m{"=" * 72}\033[0m')
    print(f'\033[1;33mRUN\033[0m {name}  ({cfg["label"]})')
    print(f'\033[1;33m{"=" * 72}\033[0m')

    if cfg['runner'] == 'baseline':
        stage4_path = run_baseline(args, cfg, log_dir)
        stage3_path = None

    else:
        s1 = run_stage1_vllm(args, cfg, log_dir)
        s2 = run_stage2_dp(args, cfg, log_dir,
                           s1, stage2_out(args.split, name),
                           targeted_attacks=cfg.get('targeted_attacks', True))

        if cfg['runner'] == 'full_pipeline':
            s3 = run_stage3(args, cfg, log_dir, s2, stage3_out(args.split, name))
            stage4_path = run_stage4(args, cfg, log_dir, s2, s3,
                                     stage4_out(args.split, name),
                                     use_graph=True)
            stage3_path = s3

        elif cfg['runner'] == 'six_agents_no_graph':
            # 6 agents + targeted stage 2, but judge does NOT see the graph.
            # This separates "agent-count effect" from "graph effect".
            # full − six_agents  ==  graph contribution
            stage4_path = run_stage4(args, cfg, log_dir, s2, None,
                                     stage4_out(args.split, name),
                                     use_graph=False)
            stage3_path = None

        elif cfg['runner'] == 'direct_judge':
            # No graph: judge reads relations only
            stage4_path = run_stage4(args, cfg, log_dir, s2, None,
                                     stage4_out(args.split, name),
                                     use_graph=False)
            stage3_path = None

        elif cfg['runner'] == 'targeted_no_graph':
            stage4_path = run_stage4(args, cfg, log_dir, s2, None,
                                     stage4_out(args.split, name),
                                     use_graph=False)
            stage3_path = None

        else:
            raise ValueError(f'unknown runner: {cfg["runner"]}')

    elapsed = time.monotonic() - t0
    return {
        'name': name,
        'label': cfg['label'],
        'stage4_path': str(stage4_path),
        'stage3_path': str(stage3_path) if stage3_path else None,
        'elapsed_seconds': round(elapsed, 1),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='ddo_sample')
    p.add_argument('--topic-file',
                   default=str(PROJECT_ROOT / 'data' / 'eval' / 'ddo_sample_topics.jsonl'))
    p.add_argument('--model',
                   default=str(Path.home() / 'models' / 'Qwen2.5-3B-Instruct'))
    p.add_argument('--gpus', default='0,1,2',
                   help='comma-separated GPU indices for stage 2 '
                        '(default 0,1,2 — using all 4 crashes this system)')
    p.add_argument('--vllm-python',
                   default=str(Path.home() / 'env-vllm' / 'bin' / 'python'))
    p.add_argument('--topic-limit', type=int, default=0,
                   help='0 = all topics')
    p.add_argument('--r1-args', type=int, default=3)
    p.add_argument('--r2-args', type=int, default=2)
    p.add_argument('--gpu-mem-util', type=float, default=0.80)
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--launch-stagger', type=int, default=45)

    # GPU-picker knobs. Baselines and stage 4 run on a single GPU; we pick
    # the emptiest one at runtime to avoid OOM when another process / stale
    # vLLM allocation is already on GPU 0.
    p.add_argument('--min-free-mib', type=int, default=7500,
                   help='minimum free GPU memory (MiB) required for single-GPU '
                        'stages. 7500 is the realistic minimum for Qwen2.5-3B: '
                        '~5800 MiB for weights + ~1500 MiB for KV cache + some '
                        'slack. If the picker returns None, the orchestrator '
                        'still launches on the first primary GPU, but vLLM '
                        'is likely to fail with "No available memory for the '
                        'cache blocks".')
    p.add_argument('--baseline-gpu-mem-util', type=float, default=0.80,
                   help='gpu_memory_utilization for baseline vLLM runs. '
                        'Must be high enough to leave ~2 GB for KV cache '
                        'after the 5.8 GB Qwen2.5-3B weights load. At '
                        '0.55 on an 11 GB GPU you get 0 KV blocks and '
                        'vLLM refuses to start.')
    p.add_argument('--baseline-max-model-len', type=int, default=2048,
                   help='max_model_len for baseline vLLM runs. Topics are '
                        'short; 2048 is plenty and halves KV cache size.')
    p.add_argument('--no-retry-oom', action='store_true',
                   help='by default, failed launches are retried once with '
                        'zombie cleanup + fallback GPU + shrunken memory '
                        'footprint; this flag disables that')

    p.add_argument('--configs', default='',
                   help='comma-separated config names to run (default: all)')
    p.add_argument('--skip-configs', default='',
                   help='comma-separated config names to skip')

    p.add_argument('--force-stage1', action='store_true')
    p.add_argument('--force-stage2', action='store_true')
    p.add_argument('--force-stage3', action='store_true')
    p.add_argument('--force-stage4', action='store_true')
    p.add_argument('--force-all', action='store_true',
                   help='re-run every stage for every config')

    p.add_argument('--dry-run', action='store_true',
                   help='print the plan and exit')

    args = p.parse_args()
    if args.force_all:
        args.force_stage1 = args.force_stage2 = True
        args.force_stage3 = args.force_stage4 = True

    # Pick which ablations to run
    want = set(args.configs.split(',')) if args.configs else {c['name'] for c in ABLATIONS}
    skip = set(args.skip_configs.split(',')) if args.skip_configs else set()
    want -= skip
    to_run = [c for c in ABLATIONS if c['name'] in want]

    adir = ablation_dir(args.split)
    log_dir = adir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f'Project root : {PROJECT_ROOT}')
    print(f'Split        : {args.split}')
    print(f'Topic file   : {args.topic_file}')
    print(f'Model        : {args.model}')
    print(f'GPUs         : {args.gpus}')
    print(f'Topic limit  : {args.topic_limit} (0=all)')
    print(f'Configs      : {[c["name"] for c in to_run]}')
    print(f'Output dir   : {adir}')

    if args.dry_run:
        print('\n[dry-run] not executing')
        return

    # Guard rail: topic file
    if not Path(args.topic_file).exists():
        print(f'\n\033[1;31mERROR\033[0m topic file not found: {args.topic_file}')
        print('       create it with scripts/build_ddo_sample.py or use a real path')
        sys.exit(1)

    results = []
    overall_t0 = time.monotonic()
    for cfg in to_run:
        try:
            r = run_config(args, cfg, log_dir)
            r['ok'] = True
            results.append(r)
        except Exception as ex:
            print(f'\n\033[1;31mCONFIG FAILED\033[0m {cfg["name"]}: {ex}')
            results.append({
                'name': cfg['name'], 'label': cfg['label'],
                'ok': False, 'error': str(ex),
            })

    manifest_path = adir / 'ablation_runs.json'
    manifest_path.write_text(json.dumps({
        'split': args.split,
        'model': str(args.model),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'total_elapsed_seconds': round(time.monotonic() - overall_t0, 1),
        'runs': results,
    }, indent=2))

    print(f'\n\033[1;32mALL DONE\033[0m -> {manifest_path}')
    print(f'Now run:\n    {args.vllm_python} scripts/evaluate_ablations.py '
          f'--split {args.split}')


if __name__ == '__main__':
    main()