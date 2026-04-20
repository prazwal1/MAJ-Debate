#!/usr/bin/env python
"""Create Stage 1 argument-configuration variants from an existing Stage 1 run.

The goal is to reuse already-generated Stage 1 arguments for ablations instead of
re-running expensive model calls. The sampler keeps Pro/Con balance, preserves
persona diversity where possible, and emits Stage 1-compatible JSON files that
Stage 2 can consume directly.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_CONFIGS = {
    "minimal_1v1": {
        "n_pro_personas": 1,
        "n_con_personas": 1,
        "r1_per_persona": 2,
        "r2_per_persona": 1,
        "description": "Small baseline-style debate with one persona per side.",
    },
    "balanced_2v2": {
        "n_pro_personas": 2,
        "n_con_personas": 2,
        "r1_per_persona": 2,
        "r2_per_persona": 1,
        "description": "Balanced medium setting for efficient ablations.",
    },
    "academic_3v3": {
        "n_pro_personas": 3,
        "n_con_personas": 3,
        "r1_per_persona": 2,
        "r2_per_persona": 1,
        "description": "Recommended academic default: full persona diversity with lighter argument count.",
    },
    "full_3v3": {
        "n_pro_personas": 3,
        "n_con_personas": 3,
        "r1_per_persona": 3,
        "r2_per_persona": 2,
        "description": "Keep the original full Stage 1 setting.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/stage1/ddo_sample/stage1_arguments.json",
        help="Path to an existing Stage 1 JSON output.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage1_variants/ddo_sample",
        help="Directory where configuration variants will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible argument sampling.",
    )
    parser.add_argument(
        "--configs-json",
        help="Optional JSON file overriding DEFAULT_CONFIGS.",
    )
    parser.add_argument(
        "--topic-limit",
        type=int,
        default=0,
        help="Optional limit on number of topics to include.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def group_arguments(topic: dict) -> Dict[str, Dict[str, List[dict]]]:
    grouped: Dict[str, Dict[str, List[dict]]] = {"PRO": {}, "CON": {}}
    for arg in topic.get("arguments", []):
        stance = arg["stance"]
        persona_id = arg["persona_id"]
        grouped.setdefault(stance, {}).setdefault(persona_id, []).append(arg)
    for stance_groups in grouped.values():
        for persona_args in stance_groups.values():
            persona_args.sort(key=lambda item: (item.get("round", 0), item["arg_id"]))
    return grouped


def select_personas(
    persona_map: Dict[str, List[dict]],
    count: int,
    rng: random.Random,
) -> List[str]:
    persona_ids = sorted(persona_map.keys())
    if count >= len(persona_ids):
        return persona_ids
    return sorted(rng.sample(persona_ids, count))


def sample_persona_arguments(
    persona_args: List[dict],
    r1_count: int,
    r2_count: int,
    rng: random.Random,
) -> List[dict]:
    r1_args = [deepcopy(arg) for arg in persona_args if arg.get("round") == 1]
    r2_args = [deepcopy(arg) for arg in persona_args if arg.get("round") == 2]

    selected: List[dict] = []

    if r1_args:
        keep_r1 = min(r1_count, len(r1_args))
        selected.extend(sorted(rng.sample(r1_args, keep_r1), key=lambda item: item["arg_id"]))

    if r2_args and r2_count > 0:
        keep_r2 = min(r2_count, len(r2_args))
        selected.extend(sorted(rng.sample(r2_args, keep_r2), key=lambda item: item["arg_id"]))

    return selected


def sample_topic(topic: dict, config_name: str, config: dict, base_seed: int) -> dict:
    rng = random.Random(f"{base_seed}:{config_name}:{topic['topic_id']}")
    grouped = group_arguments(topic)

    selected_args: List[dict] = []
    selected_personas = {"PRO": [], "CON": []}

    for stance, persona_count_key in (("PRO", "n_pro_personas"), ("CON", "n_con_personas")):
        persona_map = grouped.get(stance, {})
        chosen_personas = select_personas(persona_map, config[persona_count_key], rng)
        selected_personas[stance] = chosen_personas
        for persona_id in chosen_personas:
            selected_args.extend(
                sample_persona_arguments(
                    persona_map[persona_id],
                    r1_count=config["r1_per_persona"],
                    r2_count=config["r2_per_persona"],
                    rng=rng,
                )
            )

    selected_args.sort(key=lambda item: item["arg_id"])

    r1_count = sum(1 for arg in selected_args if arg.get("round") == 1)
    r2_count = sum(1 for arg in selected_args if arg.get("round") == 2)

    topic_copy = deepcopy(topic)
    topic_copy["arguments"] = selected_args
    topic_copy["run_name"] = f"{topic.get('run_name', 'stage1')}-{config_name}"
    topic_copy["sampling_config"] = {
        "name": config_name,
        "description": config.get("description", ""),
        "selected_personas": selected_personas,
        "seed": base_seed,
    }
    topic_copy["meta"] = {
        **topic_copy.get("meta", {}),
        "n_pro": len(selected_personas["PRO"]),
        "n_con": len(selected_personas["CON"]),
        "r1_per_agent": config["r1_per_persona"],
        "r2_per_agent": config["r2_per_persona"],
        "total_args": len(selected_args),
        "r1_args": r1_count,
        "r2_args": r2_count,
        "derived_from_stage1_run": topic.get("run_name"),
        "sampling_config": config_name,
    }
    return topic_copy


def compute_summary(topics: Iterable[dict]) -> dict:
    topics = list(topics)
    total_r1 = sum(topic.get("meta", {}).get("r1_args", 0) for topic in topics)
    total_r2 = sum(topic.get("meta", {}).get("r2_args", 0) for topic in topics)
    total_args = total_r1 + total_r2
    total_topics = len(topics)
    return {
        "total_topics": total_topics,
        "total_r1_args": total_r1,
        "total_r2_args": total_r2,
        "total_args": total_args,
        "avg_args_per_topic": round(total_args / total_topics, 2) if total_topics else 0.0,
        "r2_coverage_pct": round((total_r2 / total_args) * 100, 2) if total_args else 0.0,
    }


def build_variant_doc(stage1_doc: dict, config_name: str, config: dict, seed: int, topic_limit: int) -> dict:
    topics = stage1_doc.get("topics", [])
    if topic_limit > 0:
        topics = topics[:topic_limit]

    sampled_topics = [sample_topic(topic, config_name, config, seed) for topic in topics]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return {
        "stage": 1,
        "run_name": f"stage1-variant-{config_name}-{timestamp}",
        "timestamp": timestamp,
        "config": {
            **stage1_doc.get("config", {}),
            "provider": "derived_from_existing_stage1",
            "sampling_config": config_name,
            "sampling_description": config.get("description", ""),
            "sampling_seed": seed,
            "source_stage1_run": stage1_doc.get("run_name"),
        },
        "personas": stage1_doc.get("personas", {}),
        "topics": sampled_topics,
        "summary": compute_summary(sampled_topics),
    }


def load_configs(path: Path | None) -> dict:
    if not path:
        return DEFAULT_CONFIGS
    payload = load_json(path)
    return payload.get("configs", payload)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    configs = load_configs(Path(args.configs_json).resolve()) if args.configs_json else DEFAULT_CONFIGS
    stage1_doc = load_json(input_path)

    manifest = {
        "source_stage1_file": str(input_path),
        "source_stage1_run": stage1_doc.get("run_name"),
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(),
        "variants": {},
    }

    for config_name, config in configs.items():
        variant_doc = build_variant_doc(stage1_doc, config_name, config, args.seed, args.topic_limit)
        variant_dir = output_dir / config_name
        variant_path = variant_dir / "stage1_arguments.json"
        write_json(variant_path, variant_doc)
        manifest["variants"][config_name] = {
            "path": str(variant_path),
            "summary": variant_doc["summary"],
            "description": config.get("description", ""),
        }
        print(
            f"[{config_name}] wrote {variant_path} | "
            f"{variant_doc['summary']['total_topics']} topics | "
            f"{variant_doc['summary']['avg_args_per_topic']} avg args/topic"
        )

    write_json(output_dir / "manifest.json", manifest)
    print(f"Manifest written to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
