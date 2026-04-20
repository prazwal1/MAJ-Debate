#!/usr/bin/env python3
"""
Wait for GPU memory to become available, then launch a command.

Useful when another user has the GPUs and you want to queue your run to
start automatically as soon as they release memory.

Usage (non-blocking check):
    python scripts/wait_for_gpu.py --min-free-mib 7500 --check-only
    # exits 0 if a GPU meets the bar, 1 otherwise

Usage (blocking, then auto-launch):
    python scripts/wait_for_gpu.py --min-free-mib 7500 --max-wait-h 6 \\
        -- ~/env-vllm/bin/python scripts/run_all_ablations.py --split ddo_sample

Usage (just monitor):
    python scripts/wait_for_gpu.py --monitor
    # prints free memory + temp for all GPUs every 30s, Ctrl+C to stop

What it does:
    - Polls nvidia-smi every --interval seconds
    - When a GPU in --pool has >= --min-free-mib free AND temp < --max-temp-c
      AND util% < --max-util, the wait is satisfied
    - If --cmd is provided, launches it with CUDA_VISIBLE_DEVICES set to
      the winning GPU
    - Respects --max-wait-h time budget (default 6 hours)
"""

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path


def query_gpus():
    """Return [(idx, free_mib, total_mib, temp_c, util_pct), ...]."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,memory.free,memory.total,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            try:
                gpus.append(tuple(int(p) for p in parts[:5]))
            except ValueError:
                pass
    return gpus


def find_available_gpu(pool, min_free_mib, max_temp_c, max_util_pct):
    """Return (idx, free_mib) for the GPU meeting all criteria, or None."""
    gpus = [g for g in query_gpus() if g[0] in pool]
    gpus.sort(key=lambda g: g[1], reverse=True)  # most free first
    for idx, free, total, temp, util in gpus:
        if free >= min_free_mib and temp < max_temp_c and util < max_util_pct:
            return idx, free
    return None


def format_state(pool):
    """One-line readable summary of current GPU state."""
    gpus = [g for g in query_gpus() if g[0] in pool]
    parts = []
    for idx, free, total, temp, util in gpus:
        parts.append(f'GPU{idx}:{free}M/{temp}C/{util}%')
    return '  '.join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pool', default='0,1,2',
                    help='comma-separated GPU indices to watch (default: 0,1,2)')
    ap.add_argument('--min-free-mib', type=int, default=7500,
                    help='free memory threshold (default 7500)')
    ap.add_argument('--max-temp-c', type=int, default=85,
                    help='skip GPUs hotter than this (default 85C)')
    ap.add_argument('--max-util', type=int, default=50,
                    help='skip GPUs more than this %% utilized (default 50)')
    ap.add_argument('--interval', type=int, default=60,
                    help='poll interval in seconds (default 60)')
    ap.add_argument('--max-wait-h', type=float, default=6.0,
                    help='max wall-clock wait in hours (default 6)')
    ap.add_argument('--check-only', action='store_true',
                    help='check once and exit 0/1 without waiting')
    ap.add_argument('--monitor', action='store_true',
                    help='just print state every interval, never exit')
    ap.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='command to launch when GPU is ready (optional)')
    args = ap.parse_args()

    pool = [int(x) for x in args.pool.split(',') if x.strip()]

    if args.monitor:
        print(f'Monitoring GPUs {pool} every {args.interval}s. Ctrl+C to stop.\n')
        while True:
            now = datetime.datetime.now().strftime('%H:%M:%S')
            print(f'[{now}] {format_state(pool)}')
            time.sleep(args.interval)

    if args.check_only:
        found = find_available_gpu(pool, args.min_free_mib,
                                    args.max_temp_c, args.max_util)
        if found:
            print(f'GPU {found[0]} ready ({found[1]} MiB free)')
            sys.exit(0)
        else:
            print(f'No GPU meets bar ({args.min_free_mib} MiB, <{args.max_temp_c}C, '
                  f'<{args.max_util}% util)')
            print(f'  State: {format_state(pool)}')
            sys.exit(1)

    cmd = list(args.cmd)
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]

    # Wait loop
    max_wait_s = args.max_wait_h * 3600
    started = time.monotonic()
    attempt = 0
    print(f'Waiting up to {args.max_wait_h:.1f}h for a GPU in {pool} '
          f'with >= {args.min_free_mib} MiB free, <{args.max_temp_c}C, '
          f'<{args.max_util}% util...\n')
    while True:
        elapsed = time.monotonic() - started
        if elapsed > max_wait_s:
            print(f'\nTimeout after {elapsed / 3600:.1f}h. Giving up.')
            sys.exit(2)

        attempt += 1
        found = find_available_gpu(pool, args.min_free_mib,
                                    args.max_temp_c, args.max_util)
        now = datetime.datetime.now().strftime('%H:%M:%S')
        if found:
            gpu_id, free = found
            print(f'[{now}] GPU {gpu_id} is ready ({free} MiB free). Launching.\n')
            if not cmd:
                print(f'(No --cmd given; just exiting with GPU id on stdout.)')
                print(gpu_id)
                sys.exit(0)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
            print(f'$ CUDA_VISIBLE_DEVICES={gpu_id} {" ".join(cmd)}\n')
            rc = subprocess.call(cmd, env=env)
            sys.exit(rc)

        # Show state and wait
        state = format_state(pool)
        elapsed_m = int(elapsed / 60)
        print(f'[{now}] attempt {attempt}  elapsed {elapsed_m}m  | {state}')
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
