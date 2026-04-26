#!/usr/bin/env python3
"""
Run the no-internal-attack graph experiment on the selected mixed 10 topics.

The topic set is read from `data/eval/google_form/form_topics.jsonl` by default,
which is the same 5 DDO + 5 logic set used by the web inspector / Google Form.

For each graph-using config, this script:
  1. builds a filtered mixed Stage 2 file containing only the selected topics
  2. rebuilds Stage 3 with `--cross-stance-only`
  3. reruns Stage 4 with the no-internal graph context
  4. writes an old-vs-new summary

Example server run:
    ~/env-vllm/bin/python scripts/run_selected10_no_internal_experiment.py \
        --model ~/models/Qwen2.5-3B-Instruct \
        --gpu 0

Notes:
  - The saved full DDO Stage 3 has graph summaries for the selected DDO topics,
    but the saved full DDO Stage 2 relation file may not contain those same
    topic rows. By default, for `full` DDO rows only, the script falls back to
    `dung_no_agents` Stage 2 relations so the mixed selected-10 run can still
    cover all 10 topics. The summary records this fallback explicitly.
  - Disable that behavior with `--no-ddo-full-fallback`; then `full` will likely
    run on only the selected logic topics unless full DDO Stage 2 rows exist.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


GRAPH_CONFIGS = ["two_agents", "dung_no_agents", "full"]


def find_project_root():
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / "outputs").exists() or (p / "scripts").exists():
            return p
    return cwd


ROOT = find_project_root()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_selected_topics(path):
    topics = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        topic = json.loads(line)
        topic_id = topic["topic_id"]
        if topic_id.startswith("DDO_"):
            split = "ddo_sample"
        elif topic_id.startswith("LOGIC_"):
            split = "logic_test"
        else:
            split = topic.get("dataset") or topic.get("split")
        topics.append({**topic, "split": split})
    return topics


def stage2_path(split, config):
    if config == "full":
        return ROOT / "outputs" / "stage2" / split / "stage2_relations.json"
    return ROOT / "outputs" / "stage2" / f"{split}_{config}" / "stage2_relations.json"


def stage3_original_path(split, config):
    if config == "full":
        return ROOT / "outputs" / "stage3" / split / "stage3_graphs.json"
    return ROOT / "outputs" / "stage3" / f"{split}_{config}" / "stage3_graphs.json"


def stage4_original_path(split, config):
    if config == "full":
        return ROOT / "outputs" / "stage4" / split / "stage4_judgments.json"
    return ROOT / "outputs" / "stage4" / f"{split}_{config}" / "stage4_judgments.json"


def mixed_stage2_path(config):
    return ROOT / "outputs" / "stage2" / f"selected10_{config}_mixed" / "stage2_relations.json"


def mixed_stage3_path(config):
    return ROOT / "outputs" / "stage3" / f"selected10_{config}_no_internal" / "stage3_graphs.json"


def mixed_stage4_path(config):
    return ROOT / "outputs" / "stage4" / f"selected10_{config}_no_internal" / "stage4_judgments.json"


def run(cmd, env=None, log_path=None):
    print(f"\n\033[1;36m[run]\033[0m {' '.join(str(c) for c in cmd)}")
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") if log_path else _NullCtx() as lf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=ROOT,
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


def topic_index(path):
    if not path.exists():
        return {}
    doc = load_json(path)
    return {t.get("topic_id"): t for t in doc.get("topics", [])}


def build_mixed_stage2(config, selected_topics, allow_ddo_full_fallback=True):
    """Create a selected-topic mixed Stage 2 input for one config."""
    out_path = mixed_stage2_path(config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    selected_rows = []
    missing = []
    sources = []

    for selected in selected_topics:
        split = selected["split"]
        topic_id = selected["topic_id"]
        source_config = config

        key = (split, source_config)
        if key not in cache:
            cache[key] = topic_index(stage2_path(split, source_config))

        row = cache[key].get(topic_id)

        if (
            row is None
            and allow_ddo_full_fallback
            and config == "full"
            and split == "ddo_sample"
        ):
            source_config = "dung_no_agents"
            key = (split, source_config)
            if key not in cache:
                cache[key] = topic_index(stage2_path(split, source_config))
            row = cache[key].get(topic_id)

        if row is None:
            missing.append({"topic_id": topic_id, "split": split, "config": config})
            continue

        row = dict(row)
        row["selected10_relation_source"] = {
            "split": split,
            "config": source_config,
            "path": str(stage2_path(split, source_config).relative_to(ROOT)),
        }
        selected_rows.append(row)
        sources.append(row["selected10_relation_source"])

    label_counts = Counter()
    total_relations = total_kept = 0
    for topic in selected_rows:
        rels = topic.get("relations", [])
        total_relations += len(rels)
        for rel in rels:
            label_counts[rel.get("label", "None")] += 1
            total_kept += int(bool(rel.get("kept")))

    payload = {
        "topics": selected_rows,
        "summary": {
            "n_topics": len(selected_rows),
            "requested_topics": len(selected_topics),
            "missing_topics": missing,
            "relation_sources": sources,
            "total_relations": total_relations,
            "total_kept_relations": total_kept,
            "label_counts": dict(label_counts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out_path)
    return out_path, missing


def graph_accuracy(graphs):
    total = agree = 0
    for graph in graphs:
        gold = graph.get("benchmark_label")
        pred = (graph.get("graph_verdict") or {}).get("winner")
        if not gold or gold == "TIE":
            continue
        total += 1
        agree += int(gold == pred)
    return round(100 * agree / total, 2) if total else None, total


def judgment_accuracy(judgments):
    total = agree = 0
    for judgment in judgments:
        gold = judgment.get("benchmark_label")
        pred = judgment.get("verdict")
        if not gold or gold == "TIE":
            continue
        total += 1
        agree += int(gold == pred)
    return round(100 * agree / total, 2) if total else None, total


def original_graphs_for_selected(config, selected_topics):
    graphs = []
    missing = []
    cache = {}
    for selected in selected_topics:
        split = selected["split"]
        topic_id = selected["topic_id"]
        key = (split, config)
        if key not in cache:
            path = stage3_original_path(split, config)
            doc = load_json(path) if path.exists() else {"graphs": []}
            cache[key] = {g.get("topic_id"): g for g in doc.get("graphs", [])}
        graph = cache[key].get(topic_id)
        if graph:
            graphs.append(graph)
        else:
            missing.append(topic_id)
    return graphs, missing


def original_judgments_for_selected(config, selected_topics):
    judgments = []
    missing = []
    cache = {}
    for selected in selected_topics:
        split = selected["split"]
        topic_id = selected["topic_id"]
        key = (split, config)
        if key not in cache:
            path = stage4_original_path(split, config)
            doc = load_json(path) if path.exists() else {"judgments": []}
            cache[key] = {j.get("topic_id"): j for j in doc.get("judgments", [])}
        judgment = cache[key].get(topic_id)
        if judgment:
            judgments.append(judgment)
        else:
            missing.append(topic_id)
    return judgments, missing


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else None


def summarize(config, selected_topics, stage2_missing):
    old_graphs, old_graph_missing = original_graphs_for_selected(config, selected_topics)
    old_judgments, old_judgment_missing = original_judgments_for_selected(config, selected_topics)
    new_graph_doc = load_json(mixed_stage3_path(config))
    new_stage4 = mixed_stage4_path(config)
    new_judgment_doc = load_json(new_stage4) if new_stage4.exists() else {"judgments": []}

    old_graph_acc, old_graph_n = graph_accuracy(old_graphs)
    new_graph_acc, new_graph_n = graph_accuracy(new_graph_doc.get("graphs", []))
    old_stage4_acc, old_stage4_n = judgment_accuracy(old_judgments)
    new_stage4_acc, new_stage4_n = judgment_accuracy(new_judgment_doc.get("judgments", []))

    return {
        "config": config,
        "requested_topics": len(selected_topics),
        "stage2_topics": new_graph_doc.get("summary", {}).get("n_topics"),
        "stage2_missing": stage2_missing,
        "original_graph_missing": old_graph_missing,
        "original_stage4_missing": old_judgment_missing,
        "original_graph_acc_pct": old_graph_acc,
        "original_graph_acc_n": old_graph_n,
        "no_internal_graph_acc_pct": new_graph_acc,
        "no_internal_graph_acc_n": new_graph_n,
        "original_stage4_acc_pct": old_stage4_acc,
        "original_stage4_acc_n": old_stage4_n,
        "no_internal_stage4_acc_pct": new_stage4_acc,
        "no_internal_stage4_acc_n": new_stage4_n,
        "dropped_same_stance_attack_edges": new_graph_doc.get("summary", {}).get(
            "dropped_same_stance_attack_edges"
        ),
        "original_graph_verdict_counts": dict(
            Counter((g.get("graph_verdict") or {}).get("winner") for g in old_graphs)
        ),
        "no_internal_graph_verdict_counts": new_graph_doc.get("summary", {}).get(
            "verdict_counts"
        ),
        "original_stage4_verdict_counts": dict(
            Counter(j.get("verdict") for j in old_judgments)
        ),
        "no_internal_stage4_verdict_counts": dict(
            Counter(j.get("verdict") for j in new_judgment_doc.get("judgments", []))
        ),
        "original_mean_grounded_size": mean(g.get("grounded_size", 0) for g in old_graphs),
        "no_internal_mean_grounded_size": mean(
            g.get("grounded_size", 0) for g in new_graph_doc.get("graphs", [])
        ),
        "paths": {
            "mixed_stage2": str(mixed_stage2_path(config).relative_to(ROOT)),
            "no_internal_stage3": str(mixed_stage3_path(config).relative_to(ROOT)),
            "no_internal_stage4": str(mixed_stage4_path(config).relative_to(ROOT)),
        },
    }


def write_summary(rows, selected_topics):
    out_dir = ROOT / "outputs" / "ablations" / "selected10_no_internal"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "topic_source": "data/eval/google_form/form_topics.jsonl",
        "topic_ids": [t["topic_id"] for t in selected_topics],
        "splits": dict(Counter(t["split"] for t in selected_topics)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Selected 10 No-Internal Graph Experiment",
        "",
        "| Config | Topics | Graph Acc Old | Graph Acc New | Stage4 Acc Old | Stage4 Acc New | Dropped Same-Stance Attacks | Missing Stage2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | "
            f"{row.get('stage2_topics', '--')}/{row.get('requested_topics', '--')} | "
            f"{row.get('original_graph_acc_pct', '--')} | "
            f"{row.get('no_internal_graph_acc_pct', '--')} | "
            f"{row.get('original_stage4_acc_pct', '--')} | "
            f"{row.get('no_internal_stage4_acc_pct', '--')} | "
            f"{row.get('dropped_same_stance_attack_edges', '--')} | "
            f"{len(row.get('stage2_missing') or [])} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--topics",
        default=str(ROOT / "data" / "eval" / "google_form" / "form_topics.jsonl"),
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--vllm-python", default=str(Path.home() / "env-vllm" / "bin" / "python"))
    ap.add_argument("--configs", default="two_agents,dung_no_agents,full")
    ap.add_argument("--conf-threshold", type=float, default=0.65)
    ap.add_argument("--gpu-mem-util", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--force-stage2", action="store_true")
    ap.add_argument("--force-stage3", action="store_true")
    ap.add_argument("--force-stage4", action="store_true")
    ap.add_argument("--skip-stage4", action="store_true")
    ap.add_argument("--no-ddo-full-fallback", action="store_true")
    args = ap.parse_args()

    selected_topics = load_selected_topics(args.topics)
    wanted = {c.strip() for c in args.configs.split(",") if c.strip()}
    configs = [c for c in GRAPH_CONFIGS if c in wanted]
    if not configs:
        raise SystemExit(f"No graph configs selected from {sorted(wanted)}")

    log_dir = ROOT / "outputs" / "ablations" / "selected10_no_internal" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {ROOT}")
    print(f"Topics       : {args.topics}")
    print(f"Topic count  : {len(selected_topics)} ({dict(Counter(t['split'] for t in selected_topics))})")
    print(f"Configs      : {configs}")
    print(f"Model        : {args.model}")
    print(f"GPU          : {args.gpu}")

    missing_by_config = {}
    for config in configs:
        s2 = mixed_stage2_path(config)
        if s2.exists() and not args.force_stage2:
            print(f"  [skip mixed stage2] {s2} already exists")
            existing = load_json(s2)
            missing_by_config[config] = existing.get("summary", {}).get("missing_topics", [])
        else:
            s2, missing = build_mixed_stage2(
                config,
                selected_topics,
                allow_ddo_full_fallback=not args.no_ddo_full_fallback,
            )
            missing_by_config[config] = missing
            print(f"  [mixed stage2] {config}: {s2} ({len(selected_topics) - len(missing)}/{len(selected_topics)} topics)")
            if missing:
                print(f"    missing: {[m['topic_id'] for m in missing]}")

        s3 = mixed_stage3_path(config)
        if s3.exists() and not args.force_stage3:
            print(f"  [skip stage3] {s3} already exists")
        else:
            run(
                [
                    args.vllm_python,
                    "-u",
                    ROOT / "scripts" / "stage3_graph.py",
                    "--input",
                    s2,
                    "--output",
                    s3,
                    "--conf-threshold",
                    args.conf_threshold,
                    "--cross-stance-only",
                ],
                log_path=log_dir / f"{config}_stage3.log",
            )

        if args.skip_stage4:
            continue

        s4 = mixed_stage4_path(config)
        if s4.exists() and not args.force_stage4:
            print(f"  [skip stage4] {s4} already exists")
        else:
            run(
                [
                    args.vllm_python,
                    "-u",
                    ROOT / "scripts" / "stage4_judge.py",
                    "--stage2",
                    s2,
                    "--stage3",
                    s3,
                    "--output",
                    s4,
                    "--model",
                    args.model,
                    "--topic-limit",
                    len(selected_topics),
                    "--gpu-mem-util",
                    args.gpu_mem_util,
                    "--max-model-len",
                    args.max_model_len,
                    "--max-tokens",
                    args.max_tokens,
                ],
                env={"CUDA_VISIBLE_DEVICES": str(args.gpu)},
                log_path=log_dir / f"{config}_stage4.log",
            )
            time.sleep(5)

    rows = [summarize(config, selected_topics, missing_by_config.get(config, [])) for config in configs]
    out_dir = write_summary(rows, selected_topics)
    print("\nWrote:")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
