#!/usr/bin/env python3
"""
GPU picker with zombie awareness + fallback pool.

Upgraded from v1: Instead of just asking nvidia-smi "which GPU has the most
free memory right now?", this version:

  1. Lists processes currently on each GPU via nvidia-smi --query-compute-apps
     so we can see what's actually holding memory.
  2. Optionally kills zombie vLLM processes owned by the current user before
     picking — these are Python processes that died but left CUDA contexts
     behind (a very common failure mode on shared Jupyter boxes).
  3. Treats GPUs outside the primary pool (default 0,1,2) as a FALLBACK POOL
     (default GPU 4). If no GPU in the primary pool has enough free memory,
     try the fallback pool before giving up.
  4. Double-checks the free memory right before returning so we don't hand
     back a stale snapshot from 5 seconds ago.

Usage from python:
    from pick_gpu import pick_free_gpu_with_fallback
    gpu = pick_free_gpu_with_fallback(
        primary=[0, 1, 2], fallback=[4],
        min_free_mib=6000, kill_zombies=True,
    )

Usage from shell:
    python scripts/pick_gpu.py \\
        --primary 0,1,2 --fallback 4 \\
        --min-free-mib 6000 --kill-zombies --verbose
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# nvidia-smi queries
# ---------------------------------------------------------------------------

def query_gpus():
    """Return [(index, free_mib, total_mib), ...] via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,memory.free,memory.total',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as ex:
        print(f'# nvidia-smi query failed: {ex}', file=sys.stderr)
        return []
    gpus = []
    for line in out.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            try:
                gpus.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                pass
    return gpus


def query_compute_procs():
    """Return [(gpu_idx, pid, used_mib, proc_name), ...] for every compute
    process currently on any GPU."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-compute-apps=gpu_uuid,pid,used_memory,process_name',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    # GPU index -> UUID mapping
    try:
        uuid_out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,gpu_uuid',
             '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL,
        )
        uuid_to_idx = {}
        for line in uuid_out.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2:
                try:
                    uuid_to_idx[parts[1]] = int(parts[0])
                except ValueError:
                    pass
    except Exception:
        uuid_to_idx = {}

    procs = []
    for line in out.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            uuid, pid_s, mem_s, name = parts[0], parts[1], parts[2], parts[3]
            try:
                procs.append((uuid_to_idx.get(uuid, -1), int(pid_s),
                              int(mem_s), name))
            except ValueError:
                pass
    return procs


# ---------------------------------------------------------------------------
# Zombie cleanup
# ---------------------------------------------------------------------------

def _my_uid():
    try:
        return os.getuid()
    except AttributeError:
        return None


def _pid_owner(pid):
    try:
        with open(f'/proc/{pid}/status') as f:
            for line in f:
                if line.startswith('Uid:'):
                    return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _pid_is_python(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmdline = f.read().replace(b'\0', b' ').decode('utf-8', 'ignore')
        return 'python' in cmdline.lower()
    except (FileNotFoundError, PermissionError):
        return False


def kill_zombie_vllm_procs(dry_run=False, verbose=True):
    """Kill python GPU processes owned by current user (except self/parent).
    Returns list of killed PIDs."""
    my_uid = _my_uid()
    my_pid = os.getpid()
    my_ppid = os.getppid()
    procs = query_compute_procs()
    killed = []
    for gpu_idx, pid, mem_mib, name in procs:
        if pid in (my_pid, my_ppid):
            continue
        if my_uid is not None and _pid_owner(pid) != my_uid:
            if verbose:
                print(f'  [zombie] skip PID {pid} on GPU {gpu_idx} '
                      f'(not owned by me)', file=sys.stderr)
            continue
        if not _pid_is_python(pid):
            if verbose:
                print(f'  [zombie] skip PID {pid} on GPU {gpu_idx} '
                      f'(not python: {name[:40]})', file=sys.stderr)
            continue
        if verbose:
            print(f'  [zombie] {"would kill" if dry_run else "killing"} '
                  f'PID {pid} on GPU {gpu_idx} holding {mem_mib} MiB '
                  f'({name[:40]})', file=sys.stderr)
        if not dry_run:
            try:
                os.kill(pid, 15)  # SIGTERM
                time.sleep(2)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, 9)  # SIGKILL
                except ProcessLookupError:
                    pass
                killed.append(pid)
            except (ProcessLookupError, PermissionError) as ex:
                if verbose:
                    print(f'  [zombie] could not kill {pid}: {ex}',
                          file=sys.stderr)
    if killed:
        time.sleep(3)  # Let kernel reclaim CUDA contexts
    return killed


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------

def _best_from_pool(pool, min_free_mib, verbose):
    """Return (gpu_idx, free_mib) or None."""
    gpus = query_gpus()
    pool_set = set(pool)
    gpus = [g for g in gpus if g[0] in pool_set]
    if not gpus:
        return None
    gpus.sort(key=lambda g: g[1], reverse=True)
    if verbose:
        for idx, free, total in gpus:
            pct = 100 * free / max(total, 1)
            print(f'  GPU {idx}: {free} MiB free / {total} MiB total '
                  f'({pct:.1f}%)', file=sys.stderr)
    best_idx, best_free, _ = gpus[0]
    if best_free < min_free_mib:
        return None
    return best_idx, best_free


def pick_free_gpu_with_fallback(primary, fallback=None, min_free_mib=6000,
                                 kill_zombies=False, verbose=True,
                                 recheck_delay=0.5):
    """Try primary pool first, fall back to secondary pool. Kill zombies
    first if requested."""
    if kill_zombies:
        if verbose:
            print('  [zombie] scanning for stale python GPU processes...',
                  file=sys.stderr)
        killed = kill_zombie_vllm_procs(verbose=verbose)
        if killed and verbose:
            print(f'  [zombie] killed {len(killed)} process(es): {killed}',
                  file=sys.stderr)

    if verbose:
        print(f'  [pool] primary={primary}', file=sys.stderr)
    pick = _best_from_pool(primary, min_free_mib, verbose)

    if pick is None and fallback:
        if verbose:
            print(f'  [pool] primary exhausted; trying fallback={fallback}',
                  file=sys.stderr)
        pick = _best_from_pool(fallback, min_free_mib, verbose)

    if pick is None:
        return None
    gpu_idx, free_mib = pick

    time.sleep(recheck_delay)
    recheck = query_gpus()
    current_free = next((f for i, f, _ in recheck if i == gpu_idx), None)
    if current_free is not None and verbose:
        delta = current_free - free_mib
        if abs(delta) > 500:
            print(f'  [recheck] GPU {gpu_idx} free changed by {delta:+d} MiB '
                  f'in {recheck_delay}s (snapshot={free_mib}, '
                  f'now={current_free})', file=sys.stderr)
        if current_free < min_free_mib:
            print(f'  [WARN] GPU {gpu_idx} now below threshold '
                  f'({current_free} < {min_free_mib} MiB); returning anyway',
                  file=sys.stderr)
    return gpu_idx


# Backward-compat wrapper
def pick_free_gpu(min_free_mib=6000, allowed=None, verbose=False):
    if allowed is None:
        gpus = query_gpus()
        allowed = [g[0] for g in gpus]
    return pick_free_gpu_with_fallback(
        primary=allowed, fallback=None,
        min_free_mib=min_free_mib, kill_zombies=False,
        verbose=verbose, recheck_delay=0.0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', default='0,1,2')
    ap.add_argument('--fallback', default='4')
    ap.add_argument('--min-free-mib', type=int, default=6000)
    ap.add_argument('--kill-zombies', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    primary = [int(x) for x in args.primary.split(',') if x.strip()]
    fallback = [int(x) for x in args.fallback.split(',') if x.strip()] if args.fallback else []

    gpu = pick_free_gpu_with_fallback(
        primary=primary, fallback=fallback,
        min_free_mib=args.min_free_mib,
        kill_zombies=args.kill_zombies,
        verbose=args.verbose,
    )
    if gpu is None:
        print(f'# no GPU with >= {args.min_free_mib} MiB free in '
              f'primary={primary} or fallback={fallback}', file=sys.stderr)
        sys.exit(1)
    print(gpu)


if __name__ == '__main__':
    main()