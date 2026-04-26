#!/usr/bin/env python3
"""
Run the "no internal contradiction" graph experiment.

This reuses existing Stage 2 relation files, rebuilds Stage 3 after dropping
same-stance Attack edges (PRO->PRO and CON->CON), then reruns Stage 4 for the
graph-using configurations only.

Default target is the 50-topic logic_test split.

Example:
    ~/env-vllm/bin/python scripts/run_no_internal_graph_experiment.py \
        --split logic_test \
        --topic-limit 50 \
        --model ~/models/Qwen2.5-3B-Instruct \
        --gpu 0

Outputs:
    outputs/stage3/<split>_no_internal/stage3_graphs.json
    outputs/stage4/<split>_no_internal/stage4_judgments.json
    outputs/ablations/<split>_no_internal/summary.{json,md}
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


GRAPH_CONFIGS = [
    {
        "name": "two_agents",
        "label": "+ 2 Agents (no internal graph)",
    },
    {
        "name": "dung_no_agents",
        "label": "+ Dung Graph / no agents (no internal graph)",
    },
    {
        "name": "full",
        "label": "Full (no internal graph)",
    },
]


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "outputs").exists() or (p / "scripts").exists():
            return p
    return cwd


PROJECT_ROOT = find_project_root()


def stage2_path(split, config):
    if config == "full":
        return PROJECT_ROOT / "outputs" / "stage2" / split / "stage2_relations.json"
    return PROJECT_ROOT / "outputs" / "stage2" / f"{split}_{config}" / "stage2_relations.json"


def stage3_original_path(split, config):
    if config == "full":
        return PROJECT_ROOT / "outputs" / "stage3" / split / "stage3_graphs.json"
    return PROJECT_ROOT / "outputs" / "stage3" / f"{split}_{config}" / "stage3_graphs.json"


def stage4_original_path(split, config):
    if config == "full":
        return PROJECT_ROOT / "outputs" / "stage4" / split / "stage4_judgments.json"
    return PROJECT_ROOT / "outputs" / "stage4" / f"{split}_{config}" / "stage4_judgments.json"


def stage3_no_internal_path(split, config):
    if config == "full":
        return PROJECT_ROOT / "outputs" / "stage3" / f"{split}_no_internal" / "stage3_graphs.json"
    return PROJECT_ROOT / "outputs" / "stage3" / f"{split}_{config}_no_internal" / "stage3_graphs.json"


def stage4_no_internal_path(split, config):
    if config == "full":
        return PROJECT_ROOT / "outputs" / "stage4" / f"{split}_no_internal" / "stage4_judgments.json"
    return PROJECT_ROOT / "outputs" / "stage4" / f"{split}_{config}_no_internal" / "stage4_judgments.json"


def run(cmd, env=None, log_path=None):
    print(f"\n\033[1;36m[run]\033[0m {' '.join(str(c) for c in cmd)}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") if log_path else _NullCtx() as lf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=PROJECT_ROOT,
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
        raise RuntimeError(f"command failed with exit code {rc}: {cmd[0]}")


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def graph_accuracy(stage3_doc):
    graphs = stage3_doc.get("graphs", [])
    total = agree = 0
    for g in graphs:
        gold = g.get("benchmark_label")
        pred = (g.get("graph_verdict") or {}).get("winner")
        if not gold or gold == "TIE":
            continue
        total += 1
        agree += int(gold == pred)
    return round(100 * agree / total, 2) if total else None, total


def judgment_accuracy(stage4_doc):
    judgments = stage4_doc.get("judgments", [])
    total = agree = 0
    for j in judgments:
        gold = j.get("benchmark_label")
        pred = j.get("verdict")
        if not gold or gold == "TIE":
            continue
        total += 1
        agree += int(gold == pred)
    return round(100 * agree / total, 2) if total else None, total


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else None


def summarize_config(split, config):
    original_s3 = stage3_original_path(split, config)
    new_s3 = stage3_no_internal_path(split, config)
    original_s4 = stage4_original_path(split, config)
    new_s4 = stage4_no_internal_path(split, config)

    row = {
        "config": config,
        "original_stage3": str(original_s3.relative_to(PROJECT_ROOT)),
        "no_internal_stage3": str(new_s3.relative_to(PROJECT_ROOT)),
        "original_stage4": str(original_s4.relative_to(PROJECT_ROOT)),
        "no_internal_stage4": str(new_s4.relative_to(PROJECT_ROOT)),
    }

    if original_s3.exists():
        doc = load_json(original_s3)
        acc, n = graph_accuracy(doc)
        row["original_graph_acc_pct"] = acc
        row["original_graph_acc_n"] = n
        row["original_graph_verdict_counts"] = doc.get("summary", {}).get("verdict_counts")
        row["original_mean_grounded_size"] = mean(
            g.get("grounded_size", 0) for g in doc.get("graphs", [])
        )

    if new_s3.exists():
        doc = load_json(new_s3)
        acc, n = graph_accuracy(doc)
        row["no_internal_graph_acc_pct"] = acc
        row["no_internal_graph_acc_n"] = n
        row["no_internal_graph_verdict_counts"] = doc.get("summary", {}).get("verdict_counts")
        row["no_internal_mean_grounded_size"] = mean(
            g.get("grounded_size", 0) for g in doc.get("graphs", [])
        )
        row["dropped_same_stance_attack_edges"] = doc.get("summary", {}).get(
            "dropped_same_stance_attack_edges"
        )

    if original_s4.exists():
        doc = load_json(original_s4)
        acc, n = judgment_accuracy(doc)
        row["original_stage4_acc_pct"] = acc
        row["original_stage4_acc_n"] = n
        row["original_stage4_verdict_counts"] = dict(
            Counter(j.get("verdict") for j in doc.get("judgments", []))
        )

    if new_s4.exists():
        doc = load_json(new_s4)
        acc, n = judgment_accuracy(doc)
        row["no_internal_stage4_acc_pct"] = acc
        row["no_internal_stage4_acc_n"] = n
        row["no_internal_stage4_verdict_counts"] = dict(
            Counter(j.get("verdict") for j in doc.get("judgments", []))
        )

    return row


def write_summary(split, rows):
    out_dir = PROJECT_ROOT / "outputs" / "ablations" / f"{split}_no_internal"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "description": (
            "No-internal graph experiment: Stage 3 drops same-stance Attack "
            "edges before Dung semantics; Stage 4 is rerun with those graphs."
        ),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# No-Internal Graph Experiment",
        "",
        "| Config | Graph Acc Old | Graph Acc New | Stage4 Acc Old | Stage4 Acc New | Dropped Same-Stance Attacks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | "
            f"{r.get('original_graph_acc_pct', '--')} | "
            f"{r.get('no_internal_graph_acc_pct', '--')} | "
            f"{r.get('original_stage4_acc_pct', '--')} | "
            f"{r.get('no_internal_stage4_acc_pct', '--')} | "
            f"{r.get('dropped_same_stance_attack_edges', '--')} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="logic_test")
    ap.add_argument("--topic-limit", type=int, default=50)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for Stage 4")
    ap.add_argument("--vllm-python", default=str(Path.home() / "env-vllm" / "bin" / "python"))
    ap.add_argument("--configs", default="two_agents,dung_no_agents,full")
    ap.add_argument("--conf-threshold", type=float, default=0.65)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--force-stage3", action="store_true")
    ap.add_argument("--force-stage4", action="store_true")
    ap.add_argument("--skip-stage4", action="store_true",
                    help="only build/evaluate the no-internal Stage 3 graphs")
    args = ap.parse_args()

    wanted = {c.strip() for c in args.configs.split(",") if c.strip()}
    configs = [c for c in GRAPH_CONFIGS if c["name"] in wanted]
    if not configs:
        raise SystemExit(f"No graph configs selected from: {sorted(wanted)}")

    log_dir = PROJECT_ROOT / "outputs" / "ablations" / f"{args.split}_no_internal" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Split        : {args.split}")
    print(f"Topic limit  : {args.topic_limit}")
    print(f"Configs      : {[c['name'] for c in configs]}")
    print(f"Model        : {args.model}")
    print(f"GPU          : {args.gpu}")

    for cfg in configs:
        name = cfg["name"]
        s2 = stage2_path(args.split, name)
        s3 = stage3_no_internal_path(args.split, name)
        s4 = stage4_no_internal_path(args.split, name)

        if not s2.exists():
            raise FileNotFoundError(f"Missing Stage 2 for {name}: {s2}")

        if s3.exists() and not args.force_stage3:
            print(f"  [skip stage3] {s3} already exists")
        else:
            s3.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    args.vllm_python,
                    "-u",
                    PROJECT_ROOT / "scripts" / "stage3_graph.py",
                    "--input",
                    s2,
                    "--output",
                    s3,
                    "--conf-threshold",
                    args.conf_threshold,
                    "--cross-stance-only",
                ],
                log_path=log_dir / f"{name}_stage3.log",
            )

        if args.skip_stage4:
            continue

        if s4.exists() and not args.force_stage4:
            print(f"  [skip stage4] {s4} already exists")
        else:
            s4.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    args.vllm_python,
                    "-u",
                    PROJECT_ROOT / "scripts" / "stage4_judge.py",
                    "--stage2",
                    s2,
                    "--stage3",
                    s3,
                    "--output",
                    s4,
                    "--model",
                    args.model,
                    "--topic-limit",
                    args.topic_limit,
                    "--gpu-mem-util",
                    args.gpu_mem_util,
                    "--max-model-len",
                    args.max_model_len,
                    "--max-tokens",
                    args.max_tokens,
                ],
                env={"CUDA_VISIBLE_DEVICES": str(args.gpu)},
                log_path=log_dir / f"{name}_stage4.log",
            )
            time.sleep(5)

    rows = [summarize_config(args.split, c["name"]) for c in configs]
    out_dir = write_summary(args.split, rows)
    print(f"\nWrote summary:")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
