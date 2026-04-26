from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SHORTLIST = {
    "logic_test": [
        "LOGIC_002",
        "LOGIC_005",
        "LOGIC_030",
        "LOGIC_042",
        "LOGIC_045",
    ],
    "ddo_sample": [
        "DDO_18636",
        "DDO_50560",
        "DDO_61182",
        "DDO_70417",
        "DDO_10643",
    ],
}


CONFIGS = [
    "single_llm",
    "cot",
    "direct_judge",
    "two_agents",
    "six_agents",
    "targeted_attacks",
    "dung_no_agents",
    "full",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def index_topics(path: Path, key: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_json(path)
    items = payload[key]
    return {item["topic_id"]: item for item in items}


def candidate_stage_path(stage: str, split: str, config: str) -> Path | None:
    if stage == "stage1":
        candidates: list[Path]
        if config == "full":
            candidates = [ROOT / "outputs" / "stage1" / split / "stage1_arguments.json"]
        elif config == "two_agents":
            candidates = [
                ROOT / "outputs" / "stage1" / f"{split}_two_agents" / "stage1_arguments.json",
                ROOT / "outputs" / "stage1" / f"{split}__two_agents" / "stage1_arguments.json",
            ]
        elif config == "six_agents":
            candidates = [
                ROOT / "outputs" / "stage1" / f"{split}_six_agents" / "stage1_arguments.json",
                ROOT / "outputs" / "stage1" / f"{split}__six_agents" / "stage1_arguments.json",
            ]
        elif config == "dung_no_agents":
            candidates = [
                ROOT / "outputs" / "stage1" / f"{split}_dung_no_agents" / "stage1_arguments.json",
                ROOT / "outputs" / "stage1" / f"{split}__graph_no_agents" / "stage1_arguments.json",
            ]
        elif config == "targeted_attacks":
            candidates = [
                ROOT / "outputs" / "stage1" / f"{split}_targeted_attacks" / "stage1_arguments.json",
                ROOT / "outputs" / "stage1" / f"{split}__no_targeted_attacks" / "stage1_arguments.json",
            ]
        else:
            candidates = []
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    if stage == "stage2":
        if config in {"single_llm", "cot", "six_agents"}:
            return None
        name = split if config == "full" else f"{split}_{config}"
        path = ROOT / "outputs" / "stage2" / name / "stage2_relations.json"
        return path if path.exists() else None

    if stage == "stage3":
        if config not in {"full", "two_agents", "dung_no_agents"}:
            return None
        name = split if config == "full" else f"{split}_{config}"
        path = ROOT / "outputs" / "stage3" / name / "stage3_graphs.json"
        return path if path.exists() else None

    if stage == "stage4":
        name = split if config == "full" else f"{split}_{config}"
        path = ROOT / "outputs" / "stage4" / name / "stage4_judgments.json"
        return path if path.exists() else None

    return None


def build_indices() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    indices: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "stage1": {},
        "stage2": {},
        "stage3": {},
        "stage4": {},
    }
    stage_keys = {
        "stage1": "topics",
        "stage2": "topics",
        "stage3": "graphs",
        "stage4": "judgments",
    }

    for stage, key in stage_keys.items():
        for split in SHORTLIST:
            indices[stage][split] = {}
            for config in CONFIGS:
                path = candidate_stage_path(stage, split, config)
                if path is None:
                    continue
                indices[stage][split][config] = index_topics(path, key)
    return indices


def excerpt(text: str | None, limit: int = 160) -> str | None:
    if not text:
        return text
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def stage1_summary(topic: dict[str, Any]) -> dict[str, Any]:
    arguments = topic.get("arguments", [])
    by_stance: dict[str, list[dict[str, Any]]] = {"PRO": [], "CON": []}
    for arg in arguments:
        by_stance.setdefault(arg["stance"], []).append(arg)
    return {
        "meta": topic.get("meta", {}),
        "argument_counts": {
            "total": len(arguments),
            "PRO": len(by_stance.get("PRO", [])),
            "CON": len(by_stance.get("CON", [])),
        },
        "arguments": [
            {
                "arg_id": arg["arg_id"],
                "stance": arg["stance"],
                "round": arg["round"],
                "persona": arg.get("persona"),
                "targets_arg": arg.get("targets_arg"),
                "text": arg["text"],
            }
            for arg in arguments
        ],
        "samples": {
            stance: [
                {
                    "arg_id": arg["arg_id"],
                    "round": arg["round"],
                    "persona": arg.get("persona"),
                    "text": arg["text"],
                }
                for arg in by_stance.get(stance, [])[:6]
            ]
            for stance in ["PRO", "CON"]
        },
    }


def stage2_summary(topic: dict[str, Any]) -> dict[str, Any]:
    relations = topic.get("relations", [])
    kept = [rel for rel in relations if rel.get("kept")]
    kept_sorted = sorted(
        kept,
        key=lambda rel: (
            rel.get("confidence", 0.0),
            1 if rel.get("source_stance") != rel.get("target_stance") else 0,
        ),
        reverse=True,
    )
    suspicious = [
        rel
        for rel in kept_sorted
        if (
            rel.get("label") == "Attack"
            and rel.get("source_stance") == rel.get("target_stance")
        )
        or (
            rel.get("label") == "Support"
            and rel.get("source_stance") != rel.get("target_stance")
        )
    ]
    return {
        "summary": topic.get("summary", {}),
        "argument_strength": {
            k: v for k, v in list(topic.get("argument_strength", {}).items())[:20]
        },
        "kept_relations_sample": [
            {
                "source_arg_id": rel["source_arg_id"],
                "target_arg_id": rel["target_arg_id"],
                "source_stance": rel["source_stance"],
                "target_stance": rel["target_stance"],
                "label": rel["label"],
                "confidence": rel.get("confidence"),
                "premise": rel.get("premise") or rel.get("attacked_premise"),
                "justification": excerpt(rel.get("justification"), 180),
            }
            for rel in kept_sorted[:20]
        ],
        "suspicious_relations": [
            {
                "source_arg_id": rel["source_arg_id"],
                "target_arg_id": rel["target_arg_id"],
                "source_stance": rel["source_stance"],
                "target_stance": rel["target_stance"],
                "label": rel["label"],
                "confidence": rel.get("confidence"),
                "premise": rel.get("premise") or rel.get("attacked_premise"),
                "justification": excerpt(rel.get("justification"), 180),
            }
            for rel in suspicious[:12]
        ],
    }


def stage3_summary(topic: dict[str, Any], stage1_args: dict[str, dict[str, Any]]) -> dict[str, Any]:
    grounded_ids = topic.get("grounded_extension", [])
    grounded_args = []
    for arg_id in grounded_ids[:12]:
        arg = stage1_args.get(arg_id, {})
        grounded_args.append(
            {
                "arg_id": arg_id,
                "stance": arg.get("stance"),
                "round": arg.get("round"),
                "persona": arg.get("persona"),
                "text": arg.get("text"),
            }
        )
    return {
        "n_arguments": topic.get("n_arguments"),
        "n_attack_edges": topic.get("n_attack_edges"),
        "grounded_size": topic.get("grounded_size"),
        "grounded_extension": grounded_ids,
        "grounded_arguments": grounded_args,
        "n_preferred": topic.get("n_preferred"),
        "n_stable": topic.get("n_stable"),
        "graph_verdict": topic.get("graph_verdict", {}),
    }


def stage4_summary(topic: dict[str, Any]) -> dict[str, Any]:
    return {
        "verdict": topic.get("verdict"),
        "benchmark_label": topic.get("benchmark_label"),
        "confidence": topic.get("confidence"),
        "used_graph": topic.get("used_graph"),
        "graph_verdict": topic.get("graph_verdict"),
        "rationale": topic.get("rationale"),
        "killing_attacks": topic.get("killing_attacks", [])[:12],
        "raw_output_preview": topic.get("raw_output_preview"),
    }


def build_bundle() -> dict[str, Any]:
    indices = build_indices()
    bundle: dict[str, Any] = {
        "generated_from": rel(ROOT / "outputs" / "analysis_obvious_failure_shortlist.md"),
        "topics": [],
    }

    for split, topic_ids in SHORTLIST.items():
        for topic_id in topic_ids:
            full_stage4 = indices["stage4"][split]["full"][topic_id]
            topic_entry: dict[str, Any] = {
                "topic_id": topic_id,
                "dataset": split,
                "topic_text": full_stage4["topic_text"],
                "domain": full_stage4["domain"],
                "benchmark_label": full_stage4["benchmark_label"],
                "config_outputs": {},
            }

            for config in CONFIGS:
                config_entry: dict[str, Any] = {"paths": {}}

                stage1_topic = indices["stage1"].get(split, {}).get(config, {}).get(topic_id)
                stage2_topic = indices["stage2"].get(split, {}).get(config, {}).get(topic_id)
                stage3_topic = indices["stage3"].get(split, {}).get(config, {}).get(topic_id)
                stage4_topic = indices["stage4"].get(split, {}).get(config, {}).get(topic_id)

                for stage in ["stage1", "stage2", "stage3", "stage4"]:
                    path = candidate_stage_path(stage, split, config)
                    if path and path.exists():
                        config_entry["paths"][stage] = rel(path)

                if stage1_topic:
                    config_entry["stage1"] = stage1_summary(stage1_topic)

                if stage2_topic:
                    config_entry["stage2"] = stage2_summary(stage2_topic)

                if stage3_topic:
                    stage1_args = {}
                    if stage1_topic:
                        stage1_args = {
                            arg["arg_id"]: arg for arg in stage1_topic.get("arguments", [])
                        }
                    config_entry["stage3"] = stage3_summary(stage3_topic, stage1_args)

                if stage4_topic:
                    config_entry["stage4"] = stage4_summary(stage4_topic)

                topic_entry["config_outputs"][config] = config_entry

            bundle["topics"].append(topic_entry)

    return bundle


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_stage4_matrix(topic: dict[str, Any]) -> str:
    rows = []
    for config in CONFIGS:
        stage4 = topic["config_outputs"][config].get("stage4")
        if not stage4:
            continue
        agree = "yes" if stage4["verdict"] == stage4["benchmark_label"] else "no"
        rows.append(
            [
                config,
                stage4["verdict"] or "-",
                stage4["benchmark_label"] or "-",
                f"{stage4['confidence']:.2f}" if stage4.get("confidence") is not None else "-",
                agree,
                str(stage4.get("graph_verdict", "-")),
                excerpt(stage4.get("rationale"), 80) or "-",
            ]
        )
    return table(
        ["config", "verdict", "gold", "conf", "agree", "graph_verdict", "rationale"],
        rows,
    )


def format_stage_summaries(topic: dict[str, Any], stage_name: str) -> str | None:
    rows = []
    for config in CONFIGS:
        stage = topic["config_outputs"][config].get(stage_name)
        if not stage:
            continue
        if stage_name == "stage2":
            summary = stage.get("summary", {})
            label_counts = summary.get("label_counts", {})
            rows.append(
                [
                    config,
                    str(summary.get("kept_relations", "-")),
                    str(summary.get("failed_pairs", "-")),
                    str(label_counts.get("Attack", "-")),
                    str(label_counts.get("Support", "-")),
                    str(label_counts.get("Neutral", "-")),
                    str(label_counts.get("None", "-")),
                    str(summary.get("avg_strength", "-")),
                ]
            )
        elif stage_name == "stage3":
            verdict = stage.get("graph_verdict", {})
            rows.append(
                [
                    config,
                    verdict.get("winner", "-"),
                    verdict.get("basis", "-"),
                    str(verdict.get("pro_score", "-")),
                    str(verdict.get("con_score", "-")),
                    str(stage.get("grounded_size", "-")),
                    str(stage.get("n_attack_edges", "-")),
                ]
            )
    if not rows:
        return None

    if stage_name == "stage2":
        return table(
            [
                "config",
                "kept_rel",
                "failed_pairs",
                "Attack",
                "Support",
                "Neutral",
                "None",
                "avg_strength",
            ],
            rows,
        )
    return table(
        ["config", "winner", "basis", "pro_score", "con_score", "grounded_size", "attack_edges"],
        rows,
    )


def format_stage1_samples(topic: dict[str, Any]) -> str:
    blocks: list[str] = []
    for config in CONFIGS:
        stage1 = topic["config_outputs"][config].get("stage1")
        if not stage1:
            continue
        blocks.append(f"### {config}")
        counts = stage1["argument_counts"]
        blocks.append(
            f"`{counts['total']}` args total (`PRO={counts['PRO']}`, `CON={counts['CON']}`)"
        )
        blocks.append("PRO samples:")
        for arg in stage1["samples"]["PRO"][:4]:
            blocks.append(
                f"- `{arg['arg_id']}` r{arg['round']} {arg['persona']}: {arg['text']}"
            )
        blocks.append("CON samples:")
        for arg in stage1["samples"]["CON"][:4]:
            blocks.append(
                f"- `{arg['arg_id']}` r{arg['round']} {arg['persona']}: {arg['text']}"
            )
    return "\n".join(blocks)


def format_graph_focus(topic: dict[str, Any]) -> str:
    blocks: list[str] = []
    for config in CONFIGS:
        stage3 = topic["config_outputs"][config].get("stage3")
        if not stage3:
            continue
        blocks.append(f"### {config}")
        verdict = stage3["graph_verdict"]
        blocks.append(
            f"Graph verdict: `{verdict.get('winner')}` via `{verdict.get('basis')}` "
            f"(pro={verdict.get('pro_score')}, con={verdict.get('con_score')}, grounded_size={stage3.get('grounded_size')})"
        )
        if stage3["grounded_arguments"]:
            blocks.append("Grounded args:")
            for arg in stage3["grounded_arguments"][:8]:
                blocks.append(
                    f"- `{arg['arg_id']}` {arg.get('stance')}/{arg.get('persona')}: {excerpt(arg.get('text'), 110) or '-'}"
                )
    return "\n".join(blocks)


def format_relation_red_flags(topic: dict[str, Any]) -> str:
    blocks: list[str] = []
    for config in CONFIGS:
        stage2 = topic["config_outputs"][config].get("stage2")
        if not stage2:
            continue
        suspicious = stage2.get("suspicious_relations", [])
        if not suspicious:
            continue
        blocks.append(f"### {config}")
        for reln in suspicious[:6]:
            blocks.append(
                f"- `{reln['source_arg_id']} -> {reln['target_arg_id']}` "
                f"`{reln['source_stance']}->{reln['target_stance']}` `{reln['label']}` "
                f"`conf={reln['confidence']}` premise: {excerpt(reln.get('premise'), 90) or '-'}"
            )
    return "\n".join(blocks)


def build_markdown(bundle: dict[str, Any]) -> str:
    lines = [
        "# Failure Stage Bundle",
        "",
        "Generated from the obvious-failure shortlist to inspect where errors enter the pipeline.",
        "",
        "Artifacts:",
        "",
        "- Human-readable summary: `outputs/analysis_obvious_failure_bundle.md`",
        "- Structured data: `outputs/analysis_obvious_failure_bundle.json`",
        "",
    ]

    for topic in bundle["topics"]:
        lines.append(f"## {topic['topic_id']}")
        lines.append("")
        lines.append(f"Topic: `{topic['topic_text']}`")
        lines.append("")
        lines.append(
            f"Dataset: `{topic['dataset']}` | Domain: `{topic['domain']}` | Gold: `{topic['benchmark_label']}`"
        )
        lines.append("")
        lines.append("### Stage 4 Verdict Matrix")
        lines.append("")
        lines.append(format_stage4_matrix(topic))
        lines.append("")

        stage2_table = format_stage_summaries(topic, "stage2")
        if stage2_table:
            lines.append("### Stage 2 Summary")
            lines.append("")
            lines.append(stage2_table)
            lines.append("")

        stage3_table = format_stage_summaries(topic, "stage3")
        if stage3_table:
            lines.append("### Stage 3 Summary")
            lines.append("")
            lines.append(stage3_table)
            lines.append("")

        stage1_block = format_stage1_samples(topic)
        if stage1_block:
            lines.append("### Stage 1 Samples")
            lines.append("")
            lines.append(stage1_block)
            lines.append("")

        graph_block = format_graph_focus(topic)
        if graph_block:
            lines.append("### Graph Focus")
            lines.append("")
            lines.append(graph_block)
            lines.append("")

        rel_block = format_relation_red_flags(topic)
        if rel_block:
            lines.append("### Relation Red Flags")
            lines.append("")
            lines.append(rel_block)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    bundle = build_bundle()
    json_path = ROOT / "outputs" / "analysis_obvious_failure_bundle.json"
    md_path = ROOT / "outputs" / "analysis_obvious_failure_bundle.md"
    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(bundle), encoding="utf-8")
    print(rel(json_path))
    print(rel(md_path))


if __name__ == "__main__":
    main()
