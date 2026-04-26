#!/usr/bin/env python3
"""Build a static failure-inspection web bundle and matching Google Form files."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTED_BUNDLE = ROOT / "outputs" / "analysis_obvious_failure_bundle.json"
WEB_DATA = ROOT / "web_inspector" / "data" / "inspector_bundle.json"
FORM_DIR = ROOT / "data" / "eval" / "google_form"
SELECTED10_NO_INTERNAL_SUMMARY = (
    ROOT / "outputs" / "ablations" / "selected10_no_internal" / "summary.json"
)

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


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_path(stage, split, config="full"):
    suffix = split if config == "full" else f"{split}_{config}"
    if stage == 1 and config in {"full", "six_agents", "direct_judge", "targeted_attacks"}:
        suffix = split
    return ROOT / "outputs" / f"stage{stage}" / suffix


def index_by_topic(items):
    return {item.get("topic_id"): item for item in items if item.get("topic_id")}


def top_args(arguments, stance, n=3):
    stance_args = [a for a in arguments if a.get("stance") == stance]
    r1 = [a for a in stance_args if a.get("round") == 1]
    r2 = [a for a in stance_args if a.get("round") == 2]
    return (r1 + r2 + stance_args)[:n]


def compact_arg(arg):
    return {
        "arg_id": arg.get("arg_id"),
        "stance": arg.get("stance"),
        "round": arg.get("round"),
        "persona": arg.get("persona"),
        "targets_arg": arg.get("targets_arg"),
        "text": arg.get("text", ""),
    }


def compact_relation(rel):
    return {
        "source_arg_id": rel.get("source_arg_id"),
        "target_arg_id": rel.get("target_arg_id"),
        "source_stance": rel.get("source_stance"),
        "target_stance": rel.get("target_stance"),
        "source_round": rel.get("source_round"),
        "target_round": rel.get("target_round"),
        "label": rel.get("label"),
        "confidence": rel.get("confidence"),
        "kept": rel.get("kept"),
        "premise": rel.get("premise") or rel.get("attacked_premise"),
        "justification": rel.get("justification"),
    }


def load_split_indexes(split):
    stage1 = load_json(stage_path(1, split) / "stage1_arguments.json")
    stage2 = load_json(stage_path(2, split) / "stage2_relations.json")
    stage3 = load_json(stage_path(3, split) / "stage3_graphs.json")
    stage4_by_config = {}
    for config in CONFIGS:
        judgments = load_json(stage_path(4, split, config) / "stage4_judgments.json")
        stage4_by_config[config] = index_by_topic(judgments.get("judgments", []))
    return {
        "stage1": index_by_topic(stage1.get("topics", [])),
        "stage2": index_by_topic(stage2.get("topics", [])),
        "stage3": index_by_topic(stage3.get("graphs", [])),
        "stage4": stage4_by_config,
    }


def load_selected10_no_internal():
    if not SELECTED10_NO_INTERNAL_SUMMARY.exists():
        return {"available": False, "rows": [], "by_config": {}, "per_topic": {}}

    summary = load_json(SELECTED10_NO_INTERNAL_SUMMARY)
    by_config = {row.get("config"): row for row in summary.get("rows", [])}
    per_topic = defaultdict(dict)

    for config, row in by_config.items():
        paths = row.get("paths") or {}
        s2_path = ROOT / paths.get("mixed_stage2", "")
        s3_path = ROOT / paths.get("no_internal_stage3", "")
        s4_path = ROOT / paths.get("no_internal_stage4", "")
        stage2 = load_json(s2_path) if s2_path.exists() else {}
        stage3 = load_json(s3_path) if s3_path.exists() else {}
        stage4 = load_json(s4_path) if s4_path.exists() else {}

        relations = {t.get("topic_id"): t for t in stage2.get("topics", [])}
        graphs = {g.get("topic_id"): g for g in stage3.get("graphs", [])}
        judgments = {j.get("topic_id"): j for j in stage4.get("judgments", [])}
        for topic_id in set(relations) | set(graphs) | set(judgments):
            relation_topic = relations.get(topic_id) or {}
            graph = graphs.get(topic_id) or {}
            judgment = judgments.get(topic_id) or {}
            per_topic[topic_id][config] = {
                "relations": [
                    compact_relation(rel)
                    for rel in relation_topic.get("relations", [])
                    if rel.get("kept")
                    and rel.get("label") == "Attack"
                    and rel.get("source_stance") != rel.get("target_stance")
                ],
                "relation_source": (
                    (relation_topic.get("selected10_relation_source") or {}).get("path")
                    or paths.get("mixed_stage2")
                ),
                "graph_verdict": (graph.get("graph_verdict") or {}).get("winner"),
                "graph_basis": (graph.get("graph_verdict") or {}).get("basis"),
                "graph_pro_score": (graph.get("graph_verdict") or {}).get("pro_score"),
                "graph_con_score": (graph.get("graph_verdict") or {}).get("con_score"),
                "dropped_same_stance_attack_edges": graph.get(
                    "dropped_same_stance_attack_edges"
                ),
                "grounded_size": graph.get("grounded_size"),
                "n_attack_edges": graph.get("n_attack_edges"),
                "stage4_verdict": judgment.get("verdict"),
                "stage4_confidence": judgment.get("confidence"),
                "stage4_rationale": judgment.get("rationale"),
            }

    return {
        "available": True,
        "topic_source": summary.get("topic_source"),
        "topic_ids": summary.get("topic_ids", []),
        "splits": summary.get("splits", {}),
        "generated_at": summary.get("generated_at"),
        "rows": summary.get("rows", []),
        "by_config": by_config,
        "per_topic": dict(per_topic),
    }


def load_stage2_topic(split, config, topic_id):
    stage2 = load_json(stage_path(2, split, config) / "stage2_relations.json")
    return index_by_topic(stage2.get("topics", [])).get(topic_id, {})


def build_web_bundle():
    selected = load_json(SELECTED_BUNDLE)
    if not selected.get("topics"):
        raise SystemExit(f"No selected topics found in {SELECTED_BUNDLE}")

    split_indexes = {}
    no_internal = load_selected10_no_internal()
    topics = []
    summary_counts = Counter()
    domain_counts = Counter()

    for source_topic in selected["topics"]:
        split = source_topic["dataset"]
        if split not in split_indexes:
            split_indexes[split] = load_split_indexes(split)
        indexes = split_indexes[split]

        topic_id = source_topic["topic_id"]
        stage1 = indexes["stage1"].get(topic_id, {})
        stage2 = indexes["stage2"].get(topic_id, {})
        stage3 = indexes["stage3"].get(topic_id, {})
        arguments = [compact_arg(a) for a in stage1.get("arguments", [])]
        kept_relations = [
            compact_relation(r)
            for r in stage2.get("relations", [])
            if r.get("kept") and r.get("label") in {"Attack", "Support"}
        ]
        relation_source = "full"
        if not kept_relations:
            for relation_config in ("dung_no_agents", "two_agents"):
                alternate_stage2 = load_stage2_topic(split, relation_config, topic_id)
                alternate_relations = [
                    compact_relation(r)
                    for r in alternate_stage2.get("relations", [])
                    if r.get("kept") and r.get("label") in {"Attack", "Support"}
                ]
                if alternate_relations:
                    kept_relations = alternate_relations
                    relation_source = relation_config
                    break
        if not kept_relations:
            bundled_stage2 = (
                source_topic.get("config_outputs", {})
                .get("full", {})
                .get("stage2", {})
            )
            kept_relations = [
                compact_relation(r)
                for r in bundled_stage2.get("kept_relations_sample", [])
                if r.get("label") in {"Attack", "Support"}
            ]
            if kept_relations:
                relation_source = "failure_bundle_sample"
        relation_counts = Counter(r.get("label", "Unknown") for r in kept_relations)

        verdicts = {}
        for config in CONFIGS:
            judgment = indexes["stage4"].get(config, {}).get(topic_id)
            if not judgment:
                continue
            verdicts[config] = {
                "verdict": judgment.get("verdict"),
                "confidence": judgment.get("confidence"),
                "graph_verdict": judgment.get("graph_verdict"),
                "used_graph": judgment.get("used_graph"),
                "rationale": judgment.get("rationale"),
                "killing_attacks": judgment.get("killing_attacks", []),
            }

        full_verdict = verdicts.get("full", {}).get("verdict")
        single_verdict = verdicts.get("single_llm", {}).get("verdict")
        gold = source_topic.get("benchmark_label")
        graph_winner = (stage3.get("graph_verdict") or {}).get("winner")

        flags = []
        if full_verdict and gold and full_verdict != gold:
            flags.append("full_wrong")
        if single_verdict and full_verdict and single_verdict != full_verdict:
            flags.append("baseline_disagrees")
        if graph_winner and full_verdict and graph_winner != full_verdict:
            flags.append("judge_overrode_graph")
        if full_verdict == "PRO" and gold == "CON":
            flags.append("pro_skew")

        domain_counts[source_topic.get("domain", "unknown")] += 1
        for flag in flags:
            summary_counts[flag] += 1

        topics.append(
            {
                "topic_id": topic_id,
                "dataset": split,
                "topic_text": source_topic.get("topic_text"),
                "domain": source_topic.get("domain"),
                "benchmark_label": gold,
                "flags": flags,
                "arguments": arguments,
                "relations": kept_relations,
                "relation_source": relation_source,
                "relation_counts": dict(relation_counts),
                "argument_strength": stage2.get("argument_strength", {}),
                "stage2_summary": stage2.get("summary", {}),
                "stage3": {
                    "n_arguments": stage3.get("n_arguments"),
                    "n_attack_edges": stage3.get("n_attack_edges"),
                    "grounded_extension": stage3.get("grounded_extension", []),
                    "grounded_size": stage3.get("grounded_size"),
                    "n_preferred": stage3.get("n_preferred"),
                    "n_stable": stage3.get("n_stable"),
                    "graph_verdict": stage3.get("graph_verdict"),
                    "acceptance": stage3.get("acceptance", {}),
                },
                "verdicts": verdicts,
                "no_internal": no_internal.get("per_topic", {}).get(topic_id, {}),
                "failure_notes": source_topic.get("config_outputs", {}).get("full", {}),
            }
        )

    bundle = {
        "generated_from": str(SELECTED_BUNDLE.relative_to(ROOT)),
        "topic_count": len(topics),
        "experiments": {
            "selected10_no_internal": {
                "available": no_internal.get("available", False),
                "generated_at": no_internal.get("generated_at"),
                "splits": no_internal.get("splits", {}),
                "rows": no_internal.get("rows", []),
            }
        },
        "summary": {
            "flag_counts": dict(summary_counts),
            "domain_counts": dict(domain_counts),
            "datasets": dict(Counter(t["dataset"] for t in topics)),
        },
        "topics": topics,
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


def build_form_content(bundle):
    lines = [
        "# MAJ-Debate Selected Failure Review - Google Form Content",
        "",
        f"{bundle['topic_count']} selected topics from `outputs/analysis_obvious_failure_bundle.json`.",
        "These are the same topics shown in `web_inspector/`.",
        "",
        "## Form Setup",
        "",
        "1. Create a new Google Form titled `MAJ-Debate Selected Failure Review`.",
        "2. Paste `form_instructions.md` into the form description.",
        "3. For each topic below, add one section, one verdict multiple-choice question, and one confidence linear-scale question.",
        "4. Keep each `[topic_id: ...]` marker in the question titles so `scripts/score_form_responses.py` can score the CSV.",
        "",
        "---",
        "",
    ]

    for idx, topic in enumerate(bundle["topics"], 1):
        pros = top_args(topic["arguments"], "PRO", 3)
        cons = top_args(topic["arguments"], "CON", 3)
        lines.extend(
            [
                f"## Topic {idx}/{bundle['topic_count']} - `{topic['topic_id']}`",
                "",
                "### Title and Description Block",
                "",
                "**Title field:**",
                "```",
                f"Topic {idx} of {bundle['topic_count']}",
                "```",
                "",
                "**Description field:**",
                "```",
                f"Resolution: {topic['topic_text']}",
                f"Dataset: {topic['dataset']} | Domain: {topic.get('domain')}",
                "",
                "ARGUMENTS FOR (PRO):",
            ]
        )
        for i, arg in enumerate(pros, 1):
            lines.append(f"  {i}. {arg.get('text', '')[:450]}")
        lines.append("")
        lines.append("ARGUMENTS AGAINST (CON):")
        for i, arg in enumerate(cons, 1):
            lines.append(f"  {i}. {arg.get('text', '')[:450]}")
        lines.extend(
            [
                "```",
                "",
                "### Multiple-choice Question",
                "```",
                f"Who wins this debate? [topic_id: {topic['topic_id']}]",
                "```",
                "",
                "Options:",
                "```",
                "PRO (the resolution is correct)",
                "CON (the resolution is wrong)",
                "TIE (arguments genuinely balanced)",
                "```",
                "",
                "### Linear-scale Question",
                "```",
                f"How confident are you? [topic_id: {topic['topic_id']}]",
                "```",
                "",
                "Scale: 1 to 5",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def sync_google_form(bundle):
    FORM_DIR.mkdir(parents=True, exist_ok=True)
    (FORM_DIR / "form_topics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "topic_id": topic["topic_id"],
                    "topic_text": topic["topic_text"],
                    "domain": topic.get("domain"),
                    "benchmark_label": topic.get("benchmark_label"),
                    "selected_reason": ",".join(topic.get("flags", [])),
                }
            )
            for topic in bundle["topics"]
        )
        + "\n",
        encoding="utf-8",
    )
    (FORM_DIR / "form_instructions.md").write_text(
        "Review the 10 selected MAJ-Debate failure topics. For each topic, read the resolution and the sampled PRO/CON arguments, choose the side that made the stronger case, and rate your confidence from 1 to 5. Judge the arguments as presented, not your personal agreement with the resolution.\n",
        encoding="utf-8",
    )
    (FORM_DIR / "google_form_content.md").write_text(
        build_form_content(bundle),
        encoding="utf-8",
    )


def main():
    bundle = build_web_bundle()
    sync_google_form(bundle)
    print(f"Wrote {WEB_DATA.relative_to(ROOT)} with {bundle['topic_count']} topics")
    print(f"Synced selected 10 into {FORM_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
