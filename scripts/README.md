# MAJ-Debate — vLLM-Only Ablation Pipeline

Complete, external-API-free pipeline to produce every row of the proposal's
ablation table using local Qwen via vLLM.

## TL;DR — finish the whole project

```bash
cd ~/Project
bash scripts/run_everything.sh
```

That runs all 8 ablations and writes the final table to
`outputs/ablations/ddo_sample/ablation_table.{csv,md,json}`.

For a 5-topic smoke test first (recommended — ~5 minutes end-to-end):

```bash
cd ~/Project
~/env-vllm/bin/python scripts/run_all_ablations.py \
    --split ddo_sample --topic-limit 5 \
    --configs single_llm,cot,full
~/env-vllm/bin/python scripts/evaluate_ablations.py --split ddo_sample
```

---

## Directory layout

Drop these files into `~/Project/scripts/`:

```
scripts/
  run_all_ablations.py       # orchestrator (one entrypoint)
  stage1_vllm.py             # argument generation (replaces OpenRouter)
  stage2_vllm_shard.py       # pair labelling (patched: +targeted flag)
  stage3_graph.py            # Dung semantics (no LLM)
  stage4_judge.py            # judge brain (vLLM)
  run_baseline_judge.py      # single-LLM + CoT baselines
  evaluate_ablations.py      # fills the ablation table
  run_everything.sh          # one-liner wrapper
```

Drop these into `~/Project/configs/`:

```
configs/
  ablation_manifest.json     # maps 8 proposal rows -> disk paths
  external_baselines.json    # (unchanged — honestly empty for now)
```

Outputs land under:

```
outputs/
  stage1/<split>[_<config>]/stage1_arguments.json
  stage2/<split>[_<config>]/stage2_relations.json
  stage3/<split>[_<config>]/stage3_graphs.json
  stage4/<split>[_<config>]/stage4_judgments.json
  ablations/<split>/
    ablation_runs.json        # which configs ran, when, with what elapsed
    ablation_table.csv        # paste into LaTeX
    ablation_table.md
    ablation_table.json       # full per-config metrics
    logs/                     # per-stage per-config logs
```

The `full` config re-uses your existing on-disk paths
(`outputs/stage{2,3,4}/ddo_sample/...`), so no recomputation there.

---

## The 8 ablation configurations

Mirrors Table "Planned ablation configurations" in the proposal exactly:

| # | Name               | Label                     | Pipeline                                          |
|---|--------------------|---------------------------|--------------------------------------------------|
| 1 | `single_llm`       | Single-LLM Baseline       | Qwen sees topic only -> verdict                  |
| 2 | `cot`              | + CoT Baseline            | Qwen with step-by-step reasoning prefix          |
| 3 | `direct_judge`     | + Direct Judge (strong)   | Stage 1+2, judge without graph                   |
| 4 | `two_agents`       | + 2 Agents                | 2 agents, targeted attacks, graph                |
| 5 | `six_agents`       | + 6 Agents                | 6 agents, targeted attacks, graph                |
| 6 | `targeted_attacks` | + Targeted Attacks        | 6 agents, targeted attacks, no graph             |
| 7 | `dung_no_agents`   | + Dung Graph (no agents)  | 2 agents, zero-shot attacks, graph               |
| 8 | `full`             | Full (6ag.+targ.+graph)   | 6 agents + targeted + graph (your existing run)  |

**Honesty note on row 3:** the proposal calls this "GPT-4 Direct Judge." Since
you're running 100 % local, it's implemented as a strong-judge Qwen pass on the
same relations the full pipeline produces, without the graph. This isolates
the graph contribution (the point of RQ1) but is NOT a real GPT-4 comparison.
Update the table label in the `.tex` to match — or note this substitution
explicitly.

---

## Runtime estimates (Qwen2.5-3B, 3 GPUs, 500 topics)

| Stage         | Time           | Notes                                         |
|---------------|----------------|-----------------------------------------------|
| Stage 1       | ~20-30 min     | Batched across all topics at once             |
| Stage 2       | ~60-110 min    | Data-parallel across 3 GPUs                   |
| Stage 3       | <1 min         | Pure python, CPU                              |
| Stage 4       | ~5-10 min      | Single batched generate() call                |
| **Per config**| ~90-150 min    | Excluding reused Stage 1/2 across configs     |
| **All 8**     | ~6-10 hours    | Most Stage 1/2 outputs are reused             |

If you're time-pressed, do `--topic-limit 100` first to get a rough table
fast, then let the full run finish overnight.

(Using 4 GPUs has been observed to crash this system, so the default is 3.
Override with `--gpus 0,1,2,3` or `GPUS=0,1,2,3 bash scripts/run_everything.sh`
if you want to try it.)

---

## Key command-line flags

All orchestrator flags:

```
--split             evaluation split name           (default: ddo_sample)
--topic-file        path to topics .jsonl            (default: data/eval/ddo_sample_topics.jsonl)
--model             local HF model dir              (default: ~/models/Qwen2.5-3B-Instruct)
--gpus              comma-sep GPU indices           (default: 0,1,2 — 4 crashes this box)
--vllm-python       env-vllm python path            (default: ~/env-vllm/bin/python)
--topic-limit       cap topics for fast runs       (default: 0 = all)
--configs           comma-sep config names to run   (default: all)
--skip-configs      comma-sep config names to skip
--force-stage1/2/3/4    re-run a stage even if output exists
--force-all         re-run every stage for every config
--dry-run           print the plan and exit

# OOM mitigation (single-GPU stages: baselines + stage 4)
--min-free-mib              6000   min free GPU memory; helper picks emptiest
--baseline-gpu-mem-util     0.55   lower mem util for baselines
--baseline-max-model-len    2048   shorter context for baselines
```

---

## Incremental workflow (recommended)

Don't kick off all 8 configs at once the first time. Do it in stages so
failures are cheap:

```bash
# 1. Smoke test on 5 topics, 3 configs -- ~5 min
~/env-vllm/bin/python scripts/run_all_ablations.py \
    --split ddo_sample --topic-limit 5 \
    --configs single_llm,cot,full
~/env-vllm/bin/python scripts/evaluate_ablations.py --split ddo_sample

# 2. If that looks right, widen to all configs on 5 topics -- ~30 min
~/env-vllm/bin/python scripts/run_all_ablations.py \
    --split ddo_sample --topic-limit 5

# 3. Happy? Kick off the full 500-topic run -- ~5-8 hours
~/env-vllm/bin/python scripts/run_all_ablations.py --split ddo_sample
```

Because the scripts all resume from disk, Ctrl-C mid-run is safe. Re-run
the same command to pick up where you left off. The orchestrator also
skips stages whose output already exists unless you pass `--force-...`.

---

## Filling the proposal table from the outputs

After `evaluate_ablations.py` runs, `outputs/ablations/ddo_sample/ablation_table.csv`
has 8 rows, columns:

```
config_name, label, status, n_judgments,
acc_mean_pct, acc_std_pct, acc_n,
pers_mean_5, pers_std_5
```

Copy numbers straight into the LaTeX table (proposal lines ~595-602).

- `acc_mean_pct` / `acc_std_pct` -> `Acc. (%)` columns (μ / σ)
- `pers_mean_5` / `pers_std_5`   -> `Pers.` columns (μ / σ)
  - **Important**: this is the *model's self-reported confidence* rescaled
    to 0-5, used as a machine-persuasiveness proxy. It is NOT the human
    Likert-1-to-5 rating the proposal ultimately describes. The proposal's
    human eval is separate (proposal §5, "Human evaluation on the in-group
    test set is conducted on 50 topics by three annotators"); this number
    should be replaced by the real Likert values once you collect them.

`ablation_table.json` has richer side-metrics you can report in the analysis
section:

- `acc_by_domain`        — accuracy broken down by policy/ethics/science/society
- `attack_diversity`     — unique premises attacked (RQ2)
- `graph_stability`      — mean grounded-extension size, % empty grounded
- `graph_vs_judge`       — per-config comparison of the Dung verdict alone vs.
                           the judge's final verdict (accuracy gap)
- `stage2_label_counts`  — Attack/Support/Neutral/None distribution from stage 2

---

## Configuration knobs (env vars)

Stage 2 respects all existing env vars PLUS one new one:

- `MAJ_STAGE2_TARGETED`       `1`=targeted premise prompt (RQ3 default),
                              `0`=zero-shot pairwise (RQ3 control).
                              Set automatically by the orchestrator per config.

All other stage 2 env vars (`MAJ_STAGE2_TOPIC_LIMIT`, `MAJ_STAGE2_GPU_MEM_UTIL`,
`MAJ_STAGE2_CONFIDENCE`, etc.) work exactly as in your existing setup.

Stage 1 env vars are exposed via CLI flags on `stage1_vllm.py` — see
`stage1_vllm.py --help`.

Stage 3 has one knob: `--conf-threshold 0.65` (matches stage 2 default).

---

## Known honest limitations

1. **GPT-4 baseline is Qwen.** Re-labelled "Direct Judge (strong)" in the
   outputs. If you later get GPT-4 access back, re-run just that config.

2. **"Persuasiveness" column = model self-confidence.** This is a placeholder
   for the human Likert rating. It trends with real persuasion in practice,
   but you should replace it before publishing.

3. **Preferred-extension enumeration is exponential** (O(2^n)). The AF class
   caps at n=20 arguments; above that it falls back to the grounded
   extension plus one greedy-maximal admissible set. With n_pro=3, n_con=3,
   r1=3, r2=2 you get 30 total args per topic -- above the cap, so expect
   approximate preferred extensions on some topics. Grounded is always exact.

4. **Attack-diversity metric** uses unique *attacked-argument count*, not
   unique *premises* (we don't store premise text from the targeted prompt
   unless you enable it). Fine for a lower bound; upgrade by parsing the
   `premise` field in kept relations.

5. **Human test set (50 topics)** — not automated. These need real human
   annotation; the pipeline produces judgments you can hand to annotators.

---

## Troubleshooting

**Zombie vLLM processes hold GPU memory after a crash.** This is the #1 cause
of OOM in subsequent runs. Your earlier vLLM workers exited, but CUDA contexts
stayed resident. Fix with the cleanup utility:
```bash
~/env-vllm/bin/python scripts/cleanup_gpus.py          # dry run, shows what's there
~/env-vllm/bin/python scripts/cleanup_gpus.py --kill   # actually clean them up
```
`run_everything.sh` runs this automatically before each ablation batch, so you
usually don't need to run it by hand. It only kills Python processes owned by
the current user — never anyone else's jobs.

**Baseline / stage 4 crashes with CUDA OOM at model load.** This happens on
Volta/Turing GPUs because vLLM falls back to xFormers (FlashAttention-2 is
Ampere+), and xFormers peaks ~2x higher memory. The orchestrator now does five
things to prevent this:
- `cleanup_gpus.py` kills zombies before each run (single biggest win)
- `pick_gpu.py` selects the emptiest GPU at runtime, not blindly GPU 0
- **Fallback GPU 4** — if the primary pool (0,1,2) is saturated, the picker
  automatically tries GPU 4
- **Automatic retry-on-OOM** — if a single-GPU launch fails, the orchestrator
  kills zombies again, re-picks a GPU (preferring the fallback), and retries
  once with `MAJ_STAGE_SHRINK=1`
- `--baseline-gpu-mem-util 0.55` and `--baseline-max-model-len 2048` for
  baselines (lower KV cache footprint)

If OOM still occurs after all that, lower further: `--baseline-gpu-mem-util 0.40`,
or expand the fallback pool: `--fallback-gpus 3,4,5`.

**Stage 2 shard fails.** The orchestrator now automatically: (1) kills zombies
before launching shards, (2) swaps a shard onto the fallback GPU if its assigned
GPU is saturated at launch time, (3) retries failed shards once with smaller
memory footprint. All of that happens transparently.

**System hangs with all 4 GPUs.** Known issue on this machine; default is now
`--gpus 0,1,2` with `--fallback-gpus 4`. If you want to experiment, pass
`--gpus 0,1,2,3 --fallback-gpus "" ` explicitly.

**Stage 2 produces 0 kept relations.** The confidence threshold is too high
for your split. Try `MAJ_STAGE2_CONFIDENCE=0.55`.

**Stage 4 parse failures >10 %.** Increase `--max-tokens` on stage 4 from
300 to 500 — Qwen is probably truncating the JSON.

**`topic_file not found`.** The orchestrator defaults to
`data/eval/ddo_sample_topics.jsonl`; pass `--topic-file` to override.

**Resume from a crash.** Just re-run the same command. All 4 stages + the
baseline runner check their own outputs and skip done topics. The zombie
sweep at the start of each run makes resume actually work reliably.