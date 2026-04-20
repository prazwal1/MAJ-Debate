#!/usr/bin/env python3
"""
Clean up zombie vLLM GPU processes before a fresh ablation run.

Run this if a previous ablation run crashed and left Python workers holding
GPU memory (you'll see this as "GPU 0 has only 322 MiB free" even though
nothing should be running).

Usage:
    python scripts/cleanup_gpus.py              # dry run, shows what would be killed
    python scripts/cleanup_gpus.py --kill       # actually kills them
    python scripts/cleanup_gpus.py --kill -v    # with extra output

What it does:
    - Lists all processes currently holding GPU memory (via nvidia-smi)
    - Filters to Python processes owned by the current user
    - Excludes the current process and its parent (never kill self)
    - SIGTERMs them, waits 2s, SIGKILLs any survivor

Safety:
    - Never touches processes owned by other users
    - Never touches non-python processes (xorg, cuda benchmarks, etc.)
    - Never touches this script's own PID or its shell parent

What this script cannot do:
    - Kill OS-level CUDA contexts from processes that are already dead but
      where the kernel hasn't fully cleaned up. Those usually resolve within
      ~30 seconds of the process exit. If nvidia-smi still shows memory
      held by a PID that no longer exists in /proc, you need to wait or
      reboot the GPU (nvidia-smi --gpu-reset, requires root).
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kill', action='store_true',
                    help='actually kill; omit for dry-run')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    # Import from the same scripts/ directory
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from pick_gpu import (query_gpus, query_compute_procs,
                           kill_zombie_vllm_procs)

    # Show current state
    print('=' * 70)
    print('GPU state BEFORE cleanup:')
    print('=' * 70)
    for idx, free, total in query_gpus():
        pct = 100 * free / max(total, 1)
        print(f'  GPU {idx}: {free} MiB free / {total} MiB total ({pct:.1f}%)')

    print()
    print('=' * 70)
    print('Compute processes currently on GPUs:')
    print('=' * 70)
    procs = query_compute_procs()
    if not procs:
        print('  (none reported by nvidia-smi)')
    else:
        for gpu_idx, pid, mem_mib, name in procs:
            print(f'  GPU {gpu_idx}: PID {pid} holding {mem_mib} MiB ({name[:50]})')

    print()
    print('=' * 70)
    if args.kill:
        print('KILLING zombie python processes owned by current user...')
    else:
        print('DRY RUN — would kill these (use --kill to actually do it):')
    print('=' * 70)

    killed = kill_zombie_vllm_procs(dry_run=not args.kill, verbose=True)

    if args.kill and killed:
        print()
        print(f'Killed {len(killed)} process(es). Waiting 5s for kernel cleanup...')
        time.sleep(5)

        print()
        print('=' * 70)
        print('GPU state AFTER cleanup:')
        print('=' * 70)
        for idx, free, total in query_gpus():
            pct = 100 * free / max(total, 1)
            print(f'  GPU {idx}: {free} MiB free / {total} MiB total ({pct:.1f}%)')
    elif not killed:
        print('  (no zombie python processes found — GPUs are clean)')


if __name__ == '__main__':
    main()
