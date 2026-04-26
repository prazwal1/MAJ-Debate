#!/usr/bin/env python3
"""
Helpers for running the selected-10 ablation suite from a Jupyter notebook
using the Groq chat-completions API.

Design goals:
  - keep Groq hits low via aggressive on-disk caching and artifact reuse
  - stay schema-compatible with the repo's stage1/stage2/stage3/stage4 outputs
  - be notebook-friendly: importable, resumable, and easy to inspect

Default topic source:
    data/eval/google_form/form_topics.jsonl

Main entrypoint:
    run_selected10_suite(...)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import requests

load_dotenv()  # load GROQ_API_KEY from .env if present

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPIC_FILE = ROOT / "data" / "eval" / "google_form" / "form_topics.jsonl"

if str(ROOT / "scripts") not in sys.path:
    sys.path.append(str(ROOT / "scripts"))

from stage3_graph import process_topic as stage3_process_topic  # noqa: E402


PRO_PERSONAS = [
    {
        "id": "pro_rationalist",
        "name": "Rationalist Pro",
        "stance": "PRO",
        "reasoning_style": "logical-empirical",
        "rhetorical_mode": "cite quantitative evidence and causal mechanisms",
        "description": "Argues from data, statistics, and formal logic. Prioritises measurable outcomes.",
    },
    {
        "id": "pro_ethicist",
        "name": "Ethics Advocate Pro",
        "stance": "PRO",
        "reasoning_style": "ethical-normative",
        "rhetorical_mode": "appeal to moral principles and rights-based frameworks",
        "description": "Argues from fairness and justice. References established ethical frameworks.",
    },
    {
        "id": "pro_futurist",
        "name": "Futurist Pro",
        "stance": "PRO",
        "reasoning_style": "economic-consequentialist",
        "rhetorical_mode": "project long-term societal and economic benefits",
        "description": "Argues from systemic benefits and long-horizon impact. Accepts short-term trade-offs.",
    },
]

CON_PERSONAS = [
    {
        "id": "con_skeptic",
        "name": "Skeptic Con",
        "stance": "CON",
        "reasoning_style": "logical-empirical",
        "rhetorical_mode": "challenge evidence quality and burden of proof",
        "description": "Contests factual claims, demands rigorous evidence, identifies logical fallacies.",
    },
    {
        "id": "con_rights",
        "name": "Rights Defender Con",
        "stance": "CON",
        "reasoning_style": "ethical-normative",
        "rhetorical_mode": "appeal to human rights, procedural justice, and democratic accountability",
        "description": "Argues the proposal violates fundamental rights regardless of practical merits.",
    },
    {
        "id": "con_pragmatist",
        "name": "Pragmatist Con",
        "stance": "CON",
        "reasoning_style": "economic-consequentialist",
        "rhetorical_mode": "highlight implementation barriers and unintended consequences",
        "description": "Argues from practical constraints: cost, feasibility, second-order effects.",
    },
]


BASELINE_SYSTEM = (
    "You are an impartial debate judge. Respond ONLY with valid JSON "
    "(no markdown fences, no prose preamble)."
)

STAGE1_SYSTEM = (
    "You are a careful debate agent. Respond ONLY with valid JSON. "
    "No preamble, no prose, no markdown fences."
)

STAGE2_SYSTEM = (
    "You are a careful debate analyst. Respond ONLY with the requested JSON. "
    "No reasoning prose, no markdown fences, just the JSON."
)

STAGE4_SYSTEM = (
    "You are an impartial debate judge. You evaluate arguments on the basis "
    "of logical structure and which side has more undefeated claims. "
    "Respond ONLY with valid JSON. Keep rationale under 30 words and "
    "list at most 5 killing_attacks."
)


MODEL_ALIASES = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

NVIDIA_MODEL_ALIASES = {
    # shorthand → NVIDIA catalog ID
    "llama-405b": "meta/llama-3.1-405b-instruct",
    "nemotron-ultra": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nemotron-70b": "nvidia/llama-3.1-nemotron-70b-instruct",
    "llama-70b": "meta/llama-3.3-70b-instruct",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct-v0.1",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_existing_items(path: Path, key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return load_json(path).get(key, [])
    except Exception:
        return []


def upsert_by_topic_id(items: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    topic_id = item.get("topic_id")
    replaced = False
    updated = []
    for existing in items:
        if existing.get("topic_id") == topic_id:
            updated.append(item)
            replaced = True
        else:
            updated.append(existing)
    if not replaced:
        updated.append(item)
    return updated


def save_topics_checkpoint(
    path: Path,
    topics: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    save_json(path, {"topics": topics, "summary": summary})


def save_judgments_checkpoint(
    path: Path,
    judgments: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    save_json(path, {"judgments": judgments, "summary": summary})


def normalize_topic(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    topic_id = raw.get("topic_id") or raw.get("id") or f"SELECTED10_{idx:04d}"
    topic_text = raw.get("topic_text") or raw.get("text")
    if not topic_text:
        raise ValueError(f"missing topic text for {topic_id}")
    if topic_id.startswith("DDO_"):
        split = "ddo_sample"
    elif topic_id.startswith("LOGIC_"):
        split = "logic_test"
    else:
        split = raw.get("split") or raw.get("dataset") or "selected10"
    return {
        "topic_id": topic_id,
        "topic_text": topic_text,
        "domain": raw.get("domain", "unknown"),
        "benchmark_label": raw.get("benchmark_label"),
        "source_dataset": raw.get("source_dataset", split),
        "split": split,
        "selected_reason": raw.get("selected_reason"),
    }


def load_selected_topics(topic_file: Path | str = DEFAULT_TOPIC_FILE) -> list[dict[str, Any]]:
    topic_path = Path(topic_file)
    topics: list[dict[str, Any]] = []
    for idx, line in enumerate(topic_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        topics.append(normalize_topic(json.loads(line), idx))
    return topics


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}
    return {}


def parse_json_array(text: str) -> list[Any]:
    if not text:
        return []
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    start = cleaned.find("[")
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "[":
            depth += 1
        elif cleaned[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return obj if isinstance(obj, list) else []
                except Exception:
                    return []
    return []


def coerce_label(raw: Any) -> str:
    value = str(raw or "").strip().title()
    return value if value in {"Attack", "Support", "Neutral", "None"} else "None"


def coerce_float01(raw: Any, default: float = 0.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))


class GroqClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        base_url: str = "https://api.groq.com/openai/v1/chat/completions",
        timeout: int = 120,
        max_retries: int = 5,
    ) -> None:
        self.model = MODEL_ALIASES.get(model, model)
        self.api_keys = self._load_api_keys(api_key)
        if not self.api_keys:
            raise ValueError(
                "At least one Groq API key is required. Set GROQ_API_KEY, "
                "GROQ_API_KEYS, or GROQ_API_KEY_2 / GROQ_API_KEY_3 ..."
            )
        self._active_key_index = 0
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir or (ROOT / "outputs" / "groq_selected10" / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_api_keys(self, direct_key: str | None) -> list[str]:
        keys = []
        if direct_key:
            keys.append(direct_key.strip())
        primary = os.environ.get("GROQ_API_KEY", "").strip()
        if primary:
            keys.append(primary)
        multi = os.environ.get("GROQ_API_KEYS", "").strip()
        if multi:
            keys.extend([part.strip() for part in re.split(r"[,\n;]", multi) if part.strip()])
        for name, value in os.environ.items():
            if re.fullmatch(r"GROQ_API_KEY_\d+", name) and value.strip():
                keys.append(value.strip())
        deduped = []
        seen = set()
        for key in keys:
            if key and key not in seen:
                deduped.append(key)
                seen.add(key)
        return deduped

    @property
    def active_key_index(self) -> int:
        return self._active_key_index

    @property
    def active_key_count(self) -> int:
        return len(self.api_keys)

    def _current_api_key(self) -> str:
        return self.api_keys[self._active_key_index]

    def _rotate_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        previous = self._active_key_index
        self._active_key_index = (self._active_key_index + 1) % len(self.api_keys)
        return self._active_key_index != previous

    def _parse_retry_after(self, response: Any) -> int | None:
        """Return seconds to wait from Retry-After header or error body, or None."""
        # Standard HTTP header
        header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if header:
            try:
                return max(int(float(header)), 1)
            except (TypeError, ValueError):
                pass
        # Body often says "Please try again in 9m25.056s" or "retry in 60s"
        body = response.text or ""
        m = re.search(r"try again in\s+(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", body, re.IGNORECASE)
        if m:
            minutes = int(m.group(1) or 0)
            seconds = math.ceil(float(m.group(2) or 0))
            total = minutes * 60 + seconds
            return max(total + 5, 1)  # +5s buffer
        return None

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "system": system_prompt,
                "user": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 700,
        namespace: str = "default",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_key = self._cache_key(system_prompt, user_prompt, temperature, max_tokens)
        cache_path = self.cache_dir / namespace / f"{cache_key}.json"
        if use_cache and cache_path.exists():
            return load_json(cache_path)

        payload = {
            "model": self.model,
            "temperature": max(temperature, 1e-8),
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        error_message = "Groq request failed"
        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self._current_api_key()}",
                    "Content-Type": "application/json",
                }
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    error_message = f"{response.status_code}: {response.text[:300]}"
                    if response.status_code == 429:
                        self._rotate_key()
                        wait = self._parse_retry_after(response) or 60
                    else:
                        wait = min(2 ** attempt, 15)
                    print(f"[retry {attempt}/{self.max_retries}] HTTP {response.status_code} — waiting {wait}s …")
                    time.sleep(wait)
                    continue
                if not response.ok:
                    error_message = f"{response.status_code}: {response.text[:500]}"
                    response.raise_for_status()
                data = response.json()
                message = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                result = {
                    "content": message,
                    "usage": data.get("usage", {}),
                    "model": data.get("model", self.model),
                    "created": data.get("created"),
                    "id": data.get("id"),
                    "raw": data,
                }
                save_json(cache_path, result)
                return result
            except Exception as exc:
                error_message = str(exc)
                if "429" in error_message or "rate limit" in error_message.lower():
                    self._rotate_key()
                    wait = 60
                else:
                    wait = min(2 ** attempt, 15)
                if attempt == self.max_retries:
                    break
                print(f"[retry {attempt}/{self.max_retries}] {error_message[:120]} — waiting {wait}s …")
                time.sleep(wait)
        raise RuntimeError(error_message)


class NvidiaClient(GroqClient):
    """Drop-in replacement for GroqClient that targets the NVIDIA NIM API.

    NVIDIA's chat-completions endpoint is OpenAI-compatible so the entire
    request/response logic is inherited unchanged.  Only three things differ:
      - base_url points at NVIDIA instead of Groq
      - API keys are read from NVIDIA_API_KEY / NVIDIA_API_KEYS / NVIDIA_API_KEY_N
      - model aliases resolve via NVIDIA_MODEL_ALIASES
      - cache lives under outputs/nvidia_selected10 by default
    """

    def __init__(
        self,
        model: str = "meta/llama-3.1-405b-instruct",
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1/chat/completions",
        timeout: int = 180,
        max_retries: int = 5,
    ) -> None:
        resolved_model = NVIDIA_MODEL_ALIASES.get(model, model)
        object.__init__(self)  # skip GroqClient.__init__; set all attrs manually below
        self.model = resolved_model
        self.api_keys = self._nvidia_load_api_keys(api_key)
        if not self.api_keys:
            raise ValueError(
                "At least one NVIDIA API key is required. "
                "Set NVIDIA_API_KEY, NVIDIA_API_KEYS, or NVIDIA_API_KEY_2 / NVIDIA_API_KEY_3 ..."
            )
        self._active_key_index = 0
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir or (ROOT / "outputs" / "nvidia_selected10" / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _nvidia_load_api_keys(self, direct_key: str | None) -> list[str]:
        keys = []
        if direct_key:
            keys.append(direct_key.strip())
        for env_name in ("NVIDIA_API_KEY", "NIM_API_KEY"):
            val = os.environ.get(env_name, "").strip()
            if val:
                keys.append(val)
        multi = os.environ.get("NVIDIA_API_KEYS", "").strip()
        if multi:
            keys.extend([p.strip() for p in re.split(r"[,\n;]", multi) if p.strip()])
        for name, value in os.environ.items():
            if re.fullmatch(r"NVIDIA_API_KEY_\d+", name) and value.strip():
                keys.append(value.strip())
        deduped: list[str] = []
        seen: set[str] = set()
        for key in keys:
            if key and key not in seen:
                deduped.append(key)
                seen.add(key)
        return deduped

    # _load_api_keys is called nowhere after __init__, but override for safety
    def _load_api_keys(self, direct_key: str | None) -> list[str]:  # type: ignore[override]
        return self._nvidia_load_api_keys(direct_key)


def _persona_block(personas: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"- {p['id']} | {p['name']} | stance={p['stance']} | "
            f"style={p['reasoning_style']} | mode={p['rhetorical_mode']} | "
            f"profile={p['description']}"
            for p in personas
        ]
    )


def build_stage1_r1_prompt(topic_text: str, personas: list[dict[str, Any]], n_args: int) -> str:
    ids = ", ".join(p["id"] for p in personas)
    return (
        f'Debate topic: "{topic_text}"\n\n'
        f"Generate arguments for these personas:\n{_persona_block(personas)}\n\n"
        f"For EACH persona, generate exactly {n_args} distinct single-sentence "
        f"arguments, max 40 words each.\n\n"
        f"Output ONLY JSON as an object keyed by persona_id.\n"
        f"Use exactly these keys: {ids}\n"
        f'Example: {{"{personas[0]["id"]}": ["arg 1", "arg 2"]}}'
    )


def build_stage1_r2_prompt(
    topic_text: str,
    personas: list[dict[str, Any]],
    opposing_args: list[str],
    n_args: int,
) -> str:
    numbered = "\n".join(f"[{i + 1}] {arg}" for i, arg in enumerate(opposing_args))
    ids = ", ".join(p["id"] for p in personas)
    return (
        f'Debate topic: "{topic_text}"\n\n'
        f"Opposing arguments:\n{numbered}\n\n"
        f"Generate counter-arguments for these personas:\n{_persona_block(personas)}\n\n"
        f"For EACH persona, generate exactly {n_args} targeted counter-arguments. "
        f"Each must be one sentence, max 30 words, and include targets_arg as the "
        f"1-based index of the opposing argument it attacks.\n\n"
        f"Output ONLY JSON as an object keyed by persona_id.\n"
        f"Each value must be a list of objects like "
        f'{{"targets_arg": 1, "argument": "..."}}.\n'
        f"Use exactly these keys: {ids}"
    )


def build_stage2_strength_prompt(topic_text: str, args: list[dict[str, Any]]) -> str:
    lines = []
    for idx, arg in enumerate(args):
        text = (arg.get("text") or "").replace("\n", " ")
        lines.append(f'ARG {idx} ({arg["arg_id"]}): {text}')
    return (
        f'Topic: "{topic_text}"\n\n'
        f"Rate each argument's strength from 0.0 to 1.0.\n\n"
        f"{chr(10).join(lines)}\n\n"
        f'Output ONLY a JSON array: [{{"arg": 0, "strength": 0.7}}, ...]'
    )


def build_stage2_pair_prompt(
    topic_text: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    targeted_attacks: bool,
) -> str:
    lines = []
    for idx, (src, tgt) in enumerate(pairs):
        src_text = (src.get("text") or "").replace("\n", " ")
        tgt_text = (tgt.get("text") or "").replace("\n", " ")
        if targeted_attacks:
            lines.append(
                f"PAIR {idx}:\n"
                f'  SOURCE ({src["arg_id"]}, stance={src.get("stance")}): {src_text}\n'
                f'  TARGET ({tgt["arg_id"]}, stance={tgt.get("stance")}): {tgt_text}'
            )
        else:
            lines.append(
                f"PAIR {idx}:\n"
                f'  SOURCE ({src["arg_id"]}): {src_text}\n'
                f'  TARGET ({tgt["arg_id"]}): {tgt_text}'
            )
    if targeted_attacks:
        return (
            f'Topic: "{topic_text}"\n\n'
            f"For each pair, identify the implicit premise of TARGET that SOURCE "
            f"is challenging or reinforcing, then choose one relation:\n"
            f"Attack, Support, Neutral, or None.\n\n"
            f"{chr(10).join(lines)}\n\n"
            f'Output ONLY JSON array: [{{"pair": 0, "label": "Attack|Support|Neutral|None", '
            f'"confidence": 0.0-1.0, "premise": "short phrase"}}, ...]'
        )
    return (
        f'Topic: "{topic_text}"\n\n'
        f"For each pair, decide whether SOURCE attacks, supports, is neutral to, "
        f"or has no relation with TARGET.\n\n"
        f"{chr(10).join(lines)}\n\n"
        f'Output ONLY JSON array: [{{"pair": 0, "label": "Attack|Support|Neutral|None", '
        f'"confidence": 0.0-1.0}}, ...]'
    )


def build_baseline_prompt(topic_text: str, mode: str) -> str:
    if mode == "cot":
        return (
            f'Debate topic: "{topic_text}"\n\n'
            f"Think through the strongest PRO and CON arguments, compare them, "
            f"then decide the winner.\n\n"
            f'Output ONLY JSON: {{"verdict": "PRO|CON|TIE", "confidence": 0.0-1.0, '
            f'"rationale": "<= 40 words", "top_pro": "one sentence", '
            f'"top_con": "one sentence"}}'
        )
    return (
        f'Debate topic: "{topic_text}"\n\n'
        f"Decide which side would win a debate on this topic.\n\n"
        f'Output ONLY JSON: {{"verdict": "PRO|CON|TIE", "confidence": 0.0-1.0, '
        f'"rationale": "<= 40 words"}}'
    )


def build_stage4_prompt(
    topic: dict[str, Any],
    stage3_graph: dict[str, Any] | None = None,
    *,
    conf_threshold: float = 0.65,
    max_args: int = 30,
    max_rels: int = 60,
) -> str:
    args = topic.get("arguments", [])
    rels = topic.get("relations", [])
    strength = topic.get("argument_strength") or {}

    arg_lines = []
    for arg in args[:max_args]:
        text = (arg.get("text") or "")[:160].replace("\n", " ")
        info = strength.get(arg["arg_id"], {})
        if isinstance(info, dict) and "strength" in info:
            arg_lines.append(
                f'  {arg["arg_id"]} [{arg.get("stance")}, s={info["strength"]:.2f}]: {text}'
            )
        else:
            arg_lines.append(f'  {arg["arg_id"]} [{arg.get("stance")}]: {text}')

    kept = [r for r in rels if r.get("kept") and r.get("label") in {"Attack", "Support"}]
    kept.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)
    rel_lines = [
        f'  {rel["source_arg_id"]} --{rel["label"]}--> {rel["target_arg_id"]} '
        f'(c={rel.get("confidence", 0.0):.2f})'
        for rel in kept[:max_rels]
    ]
    graph_block = ""
    if stage3_graph:
        graph_verdict = stage3_graph.get("graph_verdict", {})
        skeptical = [
            arg_id
            for arg_id, accepted in (stage3_graph.get("acceptance") or {}).items()
            if accepted.get("skeptical")
        ]
        graph_block = (
            "\nDUNG-SEMANTICS RESULTS:\n"
            f'  grounded extension ({len(stage3_graph.get("grounded_extension", []))}): '
            f'{stage3_graph.get("grounded_extension", [])[:15]}\n'
            f'  preferred extensions: {stage3_graph.get("n_preferred")}\n'
            f"  skeptically accepted: {skeptical[:15]}\n"
            f'  formal graph verdict: {graph_verdict.get("winner")} '
            f'(pro={graph_verdict.get("pro_score")}, con={graph_verdict.get("con_score")}, '
            f'basis={graph_verdict.get("basis")})\n'
        )
    return (
        f'Topic: "{topic["topic_text"]}"\n'
        f'Topic ID: {topic["topic_id"]}\n\n'
        f"ARGUMENTS:\n{chr(10).join(arg_lines)}\n\n"
        f"HIGH-CONFIDENCE RELATIONS (confidence >= {conf_threshold}):\n"
        f'{chr(10).join(rel_lines) if rel_lines else "  (none)"}\n'
        f"{graph_block}\n"
        f'Decide the winner and output ONLY JSON: {{"verdict": "PRO|CON|TIE", '
        f'"confidence": 0.0-1.0, "rationale": "30 words max", '
        f'"killing_attacks": ["SRC -> TGT", "..."]}}'
    )


def parse_verdict_payload(raw_text: str) -> dict[str, Any]:
    obj = parse_json_object(raw_text)
    if obj and "verdict" in obj:
        return obj
    match = re.search(r'"verdict"\s*:\s*"(PRO|CON|TIE)"', raw_text or "", re.IGNORECASE)
    if match:
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text or "")
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]{0,400})', raw_text or "")
        return {
            "verdict": match.group(1).upper(),
            "confidence": float(conf_match.group(1)) if conf_match else 0.5,
            "rationale": (rationale_match.group(1) if rationale_match else "") + " [recovered]",
            "killing_attacks": [],
        }
    return {}


_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    except Exception:
        _EMBEDDER = False
    return _EMBEDDER


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def compute_similarity_matrix(args: list[dict[str, Any]]) -> list[list[float]] | Any:
    embedder = get_embedder()
    if embedder:
        vecs = embedder.encode(
            [arg.get("text") or "" for arg in args],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return (vecs @ vecs.T).tolist()
    token_sets = [_token_set(arg.get("text") or "") for arg in args]
    matrix: list[list[float]] = []
    for left in token_sets:
        row = []
        for right in token_sets:
            union = left | right
            row.append(len(left & right) / max(len(union), 1))
        matrix.append(row)
    return matrix


def _none_relation(src: dict[str, Any], tgt: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source_arg_id": src["arg_id"],
        "target_arg_id": tgt["arg_id"],
        "source_stance": src.get("stance"),
        "target_stance": tgt.get("stance"),
        "source_round": src.get("round"),
        "target_round": tgt.get("round"),
        "label": "None",
        "confidence": 0.0,
        "kept": False,
        "justification": f"prefiltered: {reason}",
        "prefiltered": True,
    }


def prefilter_pairs(
    args: list[dict[str, Any]],
    sim: Any,
    *,
    max_round_gap: int = 4,
    min_similarity: float = 0.15,
    same_stance_min_similarity: float = 0.35,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    keep: list[tuple[dict[str, Any], dict[str, Any]]] = []
    auto_none: list[dict[str, Any]] = []
    for i, j in permutations(range(len(args)), 2):
        src = args[i]
        tgt = args[j]
        score = float(sim[i][j]) if sim is not None else 1.0
        src_round = src.get("round", 0)
        tgt_round = tgt.get("round", 0)
        if src_round is not None and tgt_round is not None and abs(src_round - tgt_round) > max_round_gap:
            auto_none.append(_none_relation(src, tgt, "round_gap"))
            continue
        if src.get("stance") == tgt.get("stance") and score < same_stance_min_similarity:
            auto_none.append(_none_relation(src, tgt, "same_stance_low_sim"))
            continue
        if score < min_similarity:
            auto_none.append(_none_relation(src, tgt, "low_sim"))
            continue
        keep.append((src, tgt))
    return keep, auto_none


def _chunks(seq: list[Any], size: int):
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


@dataclass
class OutputPaths:
    root: Path

    @property
    def stage1_root(self) -> Path:
        return self.root / "stage1"

    @property
    def stage2_root(self) -> Path:
        return self.root / "stage2"

    @property
    def stage3_root(self) -> Path:
        return self.root / "stage3"

    @property
    def stage4_root(self) -> Path:
        return self.root / "stage4"

    @property
    def reports_root(self) -> Path:
        return self.root / "reports"

    def stage1(self, name: str) -> Path:
        return self.stage1_root / name / "stage1_arguments.json"

    def stage2(self, name: str) -> Path:
        return self.stage2_root / name / "stage2_relations.json"

    def stage3(self, name: str) -> Path:
        return self.stage3_root / name / "stage3_graphs.json"

    def stage4(self, name: str) -> Path:
        return self.stage4_root / name / "stage4_judgments.json"


def run_stage1(
    client: GroqClient,
    topics: list[dict[str, Any]],
    out_path: Path,
    *,
    n_pro: int,
    n_con: int,
    r1_args: int = 3,
    r2_args: int = 2,
    force: bool = False,
) -> Path:
    if force and out_path.exists():
        out_path.unlink()

    active_pro = PRO_PERSONAS[:n_pro]
    active_con = CON_PERSONAS[:n_con]
    existing_topics = [] if force else load_existing_items(out_path, "topics")
    done_ids = {topic.get("topic_id") for topic in existing_topics}
    output_topics = list(existing_topics)

    for topic in topics:
        if topic["topic_id"] in done_ids:
            continue
        pro_r1 = parse_json_object(
            client.complete(
                STAGE1_SYSTEM,
                build_stage1_r1_prompt(topic["topic_text"], active_pro, r1_args),
                temperature=0.3,
                max_tokens=900,
                namespace="stage1_r1",
            )["content"]
        )
        con_r1 = parse_json_object(
            client.complete(
                STAGE1_SYSTEM,
                build_stage1_r1_prompt(topic["topic_text"], active_con, r1_args),
                temperature=0.3,
                max_tokens=900,
                namespace="stage1_r1",
            )["content"]
        )

        pro_round1_args = {
            pid: [str(x).strip() for x in (pro_r1.get(pid) or []) if str(x).strip()][:r1_args]
            for pid in [p["id"] for p in active_pro]
        }
        con_round1_args = {
            pid: [str(x).strip() for x in (con_r1.get(pid) or []) if str(x).strip()][:r1_args]
            for pid in [p["id"] for p in active_con]
        }

        all_pro = [arg for pid in pro_round1_args for arg in pro_round1_args[pid]]
        all_con = [arg for pid in con_round1_args for arg in con_round1_args[pid]]

        pro_r2 = parse_json_object(
            client.complete(
                STAGE1_SYSTEM,
                build_stage1_r2_prompt(topic["topic_text"], active_pro, all_con, r2_args),
                temperature=0.3,
                max_tokens=1100,
                namespace="stage1_r2",
            )["content"]
        )
        con_r2 = parse_json_object(
            client.complete(
                STAGE1_SYSTEM,
                build_stage1_r2_prompt(topic["topic_text"], active_con, all_pro, r2_args),
                temperature=0.3,
                max_tokens=1100,
                namespace="stage1_r2",
            )["content"]
        )

        flat_args = []
        idx = 0
        for persona in active_pro:
            for text in pro_round1_args.get(persona["id"], []):
                flat_args.append(
                    {
                        "arg_id": f'{topic["topic_id"]}_A{idx:03d}',
                        "persona_id": persona["id"],
                        "persona": persona["name"],
                        "stance": "PRO",
                        "round": 1,
                        "targets_arg": None,
                        "text": text,
                    }
                )
                idx += 1
        for persona in active_con:
            for text in con_round1_args.get(persona["id"], []):
                flat_args.append(
                    {
                        "arg_id": f'{topic["topic_id"]}_A{idx:03d}',
                        "persona_id": persona["id"],
                        "persona": persona["name"],
                        "stance": "CON",
                        "round": 1,
                        "targets_arg": None,
                        "text": text,
                    }
                )
                idx += 1
        for persona in active_pro:
            for item in (pro_r2.get(persona["id"]) or [])[:r2_args]:
                if isinstance(item, dict):
                    text = str(item.get("argument", "")).strip()
                    target = item.get("targets_arg")
                else:
                    text = str(item).strip()
                    target = None
                if not text:
                    continue
                flat_args.append(
                    {
                        "arg_id": f'{topic["topic_id"]}_A{idx:03d}',
                        "persona_id": persona["id"],
                        "persona": persona["name"],
                        "stance": "PRO",
                        "round": 2,
                        "targets_arg": target,
                        "text": text,
                    }
                )
                idx += 1
        for persona in active_con:
            for item in (con_r2.get(persona["id"]) or [])[:r2_args]:
                if isinstance(item, dict):
                    text = str(item.get("argument", "")).strip()
                    target = item.get("targets_arg")
                else:
                    text = str(item).strip()
                    target = None
                if not text:
                    continue
                flat_args.append(
                    {
                        "arg_id": f'{topic["topic_id"]}_A{idx:03d}',
                        "persona_id": persona["id"],
                        "persona": persona["name"],
                        "stance": "CON",
                        "round": 2,
                        "targets_arg": target,
                        "text": text,
                    }
                )
                idx += 1

        output_topics.append(
            {
                "topic_id": topic["topic_id"],
                "topic_text": topic["topic_text"],
                "domain": topic["domain"],
                "benchmark_label": topic["benchmark_label"],
                "source_dataset": topic["source_dataset"],
                "source_ref": topic.get("selected_reason"),
                "evaluation_split": "selected10",
                "run_name": f"groq-stage1-{n_pro}p{n_con}c",
                "arguments": flat_args,
                "meta": {
                    "n_pro": n_pro,
                    "n_con": n_con,
                    "r1_per_agent": r1_args,
                    "r2_per_agent": r2_args,
                    "total_args": len(flat_args),
                    "provider": "groq",
                    "model": client.model,
                },
            }
        )
        save_topics_checkpoint(
            out_path,
            output_topics,
            {
                "total_topics": len(output_topics),
                "total_args": sum(len(t["arguments"]) for t in output_topics),
                "provider": "groq",
                "model": client.model,
                "active_api_key_index": client.active_key_index,
                "available_api_keys": client.active_key_count,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        done_ids.add(topic["topic_id"])

    save_topics_checkpoint(
        out_path,
        output_topics,
        {
            "total_topics": len(output_topics),
            "total_args": sum(len(t["arguments"]) for t in output_topics),
            "provider": "groq",
            "model": client.model,
            "active_api_key_index": client.active_key_index,
            "available_api_keys": client.active_key_count,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return out_path


def run_stage2(
    client: GroqClient,
    stage1_path: Path,
    out_path: Path,
    *,
    targeted_attacks: bool,
    pair_batch_size: int = 40,
    confidence_threshold: float = 0.65,
    force: bool = False,
) -> Path:
    if force and out_path.exists():
        out_path.unlink()

    stage1 = load_json(stage1_path)
    existing_topics = [] if force else load_existing_items(out_path, "topics")
    done_ids = {topic.get("topic_id") for topic in existing_topics}
    result_topics = list(existing_topics)

    for topic in stage1.get("topics", []):
        if topic["topic_id"] in done_ids:
            continue
        args = topic.get("arguments", [])
        sim = compute_similarity_matrix(args)
        keep_pairs, auto_none = prefilter_pairs(args, sim)

        strength_raw = client.complete(
            STAGE2_SYSTEM,
            build_stage2_strength_prompt(topic["topic_text"], args),
            temperature=0.1,
            max_tokens=900,
            namespace="stage2_strength",
        )["content"]
        strength_arr = parse_json_array(strength_raw)
        strength_by_idx = {}
        for item in strength_arr:
            if isinstance(item, dict) and "arg" in item:
                try:
                    strength_by_idx[int(item["arg"])] = item
                except (TypeError, ValueError):
                    pass
        argument_strength = {}
        for idx, arg in enumerate(args):
            item = strength_by_idx.get(idx, {})
            argument_strength[arg["arg_id"]] = {
                "strength": round(coerce_float01(item.get("strength", 0.5), 0.5), 3),
                "rationale": str(item.get("rationale", "No rationale."))[:200],
            }

        relations = list(auto_none)
        failed_pairs = 0
        for batch in _chunks(keep_pairs, pair_batch_size):
            raw = client.complete(
                STAGE2_SYSTEM,
                build_stage2_pair_prompt(
                    topic["topic_text"],
                    batch,
                    targeted_attacks=targeted_attacks,
                ),
                temperature=0.1,
                max_tokens=1200,
                namespace="stage2_pairs_targeted" if targeted_attacks else "stage2_pairs_zeroshot",
            )["content"]
            arr = parse_json_array(raw)
            by_idx = {}
            for item in arr:
                if isinstance(item, dict) and "pair" in item:
                    try:
                        by_idx[int(item["pair"])] = item
                    except (TypeError, ValueError):
                        pass
            for idx, (src, tgt) in enumerate(batch):
                item = by_idx.get(idx)
                if item is None:
                    failed_pairs += 1
                    relations.append(
                        {
                            "source_arg_id": src["arg_id"],
                            "target_arg_id": tgt["arg_id"],
                            "source_stance": src.get("stance"),
                            "target_stance": tgt.get("stance"),
                            "source_round": src.get("round"),
                            "target_round": tgt.get("round"),
                            "label": "None",
                            "confidence": 0.0,
                            "kept": False,
                            "justification": "parse_failed",
                            "failed": True,
                        }
                    )
                    continue
                rel = {
                    "source_arg_id": src["arg_id"],
                    "target_arg_id": tgt["arg_id"],
                    "source_stance": src.get("stance"),
                    "target_stance": tgt.get("stance"),
                    "source_round": src.get("round"),
                    "target_round": tgt.get("round"),
                    "label": coerce_label(item.get("label")),
                    "confidence": round(coerce_float01(item.get("confidence", 0.0)), 3),
                }
                rel["kept"] = rel["confidence"] >= confidence_threshold and rel["label"] != "None"
                if "premise" in item:
                    rel["premise"] = str(item.get("premise", ""))[:200]
                relations.append(rel)

        label_counts = Counter(rel.get("label", "None") for rel in relations)
        kept_relations = sum(int(rel.get("kept", False)) for rel in relations)
        avg_strength = round(
            statistics.mean(v["strength"] for v in argument_strength.values()),
            3,
        ) if argument_strength else 0.0

        result_topics.append(
            {
                "topic_id": topic["topic_id"],
                "topic_text": topic["topic_text"],
                "domain": topic.get("domain"),
                "benchmark_label": topic.get("benchmark_label"),
                "source_dataset": topic.get("source_dataset"),
                "source_ref": topic.get("source_ref"),
                "evaluation_split": topic.get("evaluation_split"),
                "run_name": topic.get("run_name"),
                "arguments": args,
                "argument_strength": argument_strength,
                "relations": relations,
                "summary": {
                    "n_arguments": len(args),
                    "n_ordered_pairs": len(args) * max(len(args) - 1, 0),
                    "n_llm_classified": len(keep_pairs),
                    "n_prefiltered": len(auto_none),
                    "kept_relations": kept_relations,
                    "failed_pairs": failed_pairs,
                    "label_counts": dict(label_counts),
                    "confidence_threshold": confidence_threshold,
                    "avg_strength": avg_strength,
                    "targeted_attacks": targeted_attacks,
                },
            }
        )
        save_topics_checkpoint(
            out_path,
            result_topics,
            {
                "n_topics": len(result_topics),
                "targeted_attacks": targeted_attacks,
                "provider": "groq",
                "model": client.model,
                "active_api_key_index": client.active_key_index,
                "available_api_keys": client.active_key_count,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        done_ids.add(topic["topic_id"])

    save_topics_checkpoint(
        out_path,
        result_topics,
        {
            "n_topics": len(result_topics),
            "targeted_attacks": targeted_attacks,
            "provider": "groq",
            "model": client.model,
            "active_api_key_index": client.active_key_index,
            "available_api_keys": client.active_key_count,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return out_path


def run_stage3(
    stage2_path: Path,
    out_path: Path,
    *,
    conf_threshold: float = 0.65,
    cross_stance_only: bool = False,
    force: bool = False,
) -> Path:
    if out_path.exists() and not force:
        return out_path
    stage2 = load_json(stage2_path)
    graphs = [
        stage3_process_topic(topic, conf_threshold, cross_stance_only=cross_stance_only)
        for topic in stage2.get("topics", [])
    ]
    save_json(
        out_path,
        {
            "graphs": graphs,
            "summary": {
                "n_topics": len(graphs),
                "conf_threshold": conf_threshold,
                "cross_stance_only": cross_stance_only,
                "verdict_counts": dict(
                    Counter(graph["graph_verdict"]["winner"] for graph in graphs)
                ),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": str(stage2_path),
            },
        },
    )
    return out_path


def run_stage4(
    client: GroqClient,
    stage2_path: Path,
    out_path: Path,
    *,
    stage3_path: Path | None = None,
    force: bool = False,
) -> Path:
    if force and out_path.exists():
        out_path.unlink()

    stage2 = load_json(stage2_path)
    graphs_by_id = {}
    if stage3_path and stage3_path.exists():
        graphs_by_id = {g["topic_id"]: g for g in load_json(stage3_path).get("graphs", [])}

    existing_judgments = [] if force else load_existing_items(out_path, "judgments")
    done_ids = {judgment.get("topic_id") for judgment in existing_judgments}
    judgments = list(existing_judgments)
    for topic in stage2.get("topics", []):
        if topic["topic_id"] in done_ids:
            continue
        raw = client.complete(
            STAGE4_SYSTEM,
            build_stage4_prompt(topic, graphs_by_id.get(topic["topic_id"])),
            temperature=0.1,
            max_tokens=700,
            namespace="stage4_graph" if stage3_path else "stage4_nograph",
        )["content"]
        parsed = parse_verdict_payload(raw)
        verdict = str(parsed.get("verdict", "TIE")).strip().upper()
        if verdict not in {"PRO", "CON", "TIE"}:
            verdict = "TIE"
        judgments.append(
            {
                "topic_id": topic["topic_id"],
                "topic_text": topic.get("topic_text"),
                "domain": topic.get("domain"),
                "benchmark_label": topic.get("benchmark_label"),
                "source_dataset": topic.get("source_dataset"),
                "verdict": verdict,
                "confidence": round(coerce_float01(parsed.get("confidence", 0.0)), 3),
                "rationale": str(parsed.get("rationale", ""))[:400],
                "killing_attacks": [str(x) for x in (parsed.get("killing_attacks") or [])][:5],
                "used_graph": bool(stage3_path),
                "graph_verdict": (
                    graphs_by_id.get(topic["topic_id"], {})
                    .get("graph_verdict", {})
                    .get("winner")
                ),
                "raw_output_preview": raw[:200],
            }
        )
        save_judgments_checkpoint(
            out_path,
            judgments,
            {
                "n_topics": len(judgments),
                "verdict_counts": dict(Counter(j["verdict"] for j in judgments)),
                "agreement_with_benchmark_pct": agreement_pct(judgments),
                "used_graph": bool(stage3_path),
                "active_api_key_index": client.active_key_index,
                "available_api_keys": client.active_key_count,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        done_ids.add(topic["topic_id"])

    save_judgments_checkpoint(
        out_path,
        judgments,
        {
            "n_topics": len(judgments),
            "verdict_counts": dict(Counter(j["verdict"] for j in judgments)),
            "agreement_with_benchmark_pct": agreement_pct(judgments),
            "used_graph": bool(stage3_path),
            "active_api_key_index": client.active_key_index,
            "available_api_keys": client.active_key_count,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return out_path


def run_baseline(
    client: GroqClient,
    topics: list[dict[str, Any]],
    out_path: Path,
    *,
    mode: str,
    force: bool = False,
) -> Path:
    if force and out_path.exists():
        out_path.unlink()
    existing_judgments = [] if force else load_existing_items(out_path, "judgments")
    done_ids = {judgment.get("topic_id") for judgment in existing_judgments}
    judgments = list(existing_judgments)
    namespace = f"baseline_{mode}"
    for topic in topics:
        if topic["topic_id"] in done_ids:
            continue
        raw = client.complete(
            BASELINE_SYSTEM,
            build_baseline_prompt(topic["topic_text"], mode),
            temperature=0.1 if mode == "single" else 0.2,
            max_tokens=500,
            namespace=namespace,
        )["content"]
        parsed = parse_verdict_payload(raw)
        verdict = str(parsed.get("verdict", "TIE")).strip().upper()
        if verdict not in {"PRO", "CON", "TIE"}:
            verdict = "TIE"
        judgments.append(
            {
                "topic_id": topic["topic_id"],
                "topic_text": topic["topic_text"],
                "domain": topic["domain"],
                "benchmark_label": topic.get("benchmark_label"),
                "source_dataset": topic.get("source_dataset"),
                "verdict": verdict,
                "confidence": round(coerce_float01(parsed.get("confidence", 0.0)), 3),
                "rationale": str(parsed.get("rationale", ""))[:400],
                "killing_attacks": [],
                "used_graph": False,
                "raw_output_preview": raw[:200],
                "baseline_mode": mode,
            }
        )
        save_judgments_checkpoint(
            out_path,
            judgments,
            {
                "n_topics": len(judgments),
                "verdict_counts": dict(Counter(j["verdict"] for j in judgments)),
                "agreement_with_benchmark_pct": agreement_pct(judgments),
                "baseline_mode": mode,
                "active_api_key_index": client.active_key_index,
                "available_api_keys": client.active_key_count,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        done_ids.add(topic["topic_id"])
    save_judgments_checkpoint(
        out_path,
        judgments,
        {
            "n_topics": len(judgments),
            "verdict_counts": dict(Counter(j["verdict"] for j in judgments)),
            "agreement_with_benchmark_pct": agreement_pct(judgments),
            "baseline_mode": mode,
            "active_api_key_index": client.active_key_index,
            "available_api_keys": client.active_key_count,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return out_path


def agreement_pct(judgments: list[dict[str, Any]]) -> float | None:
    total = 0
    agree = 0
    for judgment in judgments:
        gold = judgment.get("benchmark_label")
        pred = judgment.get("verdict")
        if not gold or gold == "TIE":
            continue
        total += 1
        agree += int(gold == pred)
    return round(100.0 * agree / total, 2) if total else None


def summarize_suite(paths: OutputPaths) -> list[dict[str, Any]]:
    rows = []
    configs = {
        "single_llm": paths.stage4("single_llm"),
        "cot": paths.stage4("cot"),
        "direct_judge": paths.stage4("direct_judge"),
        "two_agents": paths.stage4("two_agents"),
        "six_agents": paths.stage4("six_agents"),
        "targeted_attacks": paths.stage4("targeted_attacks"),
        "dung_no_agents": paths.stage4("dung_no_agents"),
        "full": paths.stage4("full"),
    }
    for config, path in configs.items():
        if not path.exists():
            rows.append({"config": config, "status": "missing"})
            continue
        doc = load_json(path)
        judgments = doc.get("judgments", [])
        rows.append(
            {
                "config": config,
                "status": "ok",
                "n_topics": len(judgments),
                "accuracy_pct": agreement_pct(judgments),
                "mean_confidence": round(
                    statistics.mean(j.get("confidence", 0.0) for j in judgments),
                    3,
                ) if judgments else None,
                "verdict_counts": doc.get("summary", {}).get("verdict_counts", {}),
            }
        )
    return rows


def write_suite_summary(paths: OutputPaths) -> Path:
    rows = summarize_suite(paths)
    out_path = paths.reports_root / "selected10_ablation_summary.json"
    save_json(out_path, {"rows": rows, "generated_at": datetime.now().isoformat(timespec="seconds")})

    md_path = paths.reports_root / "selected10_ablation_summary.md"
    lines = [
        "# Selected 10 Groq Ablation Summary",
        "",
        "| Config | Status | Topics | Accuracy % | Mean confidence |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["config"]} | {row.get("status")} | '
            f'{row.get("n_topics", "--")} | {row.get("accuracy_pct", "--")} | '
            f'{row.get("mean_confidence", "--")} |'
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def estimate_request_plan(topics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    topic_count = len(topics or load_selected_topics())
    return {
        "topic_count": topic_count,
        "minimum_new_request_groups": {
            "baselines": topic_count * 2,
            "stage1_2agent": topic_count * 4,
            "stage1_6agent": topic_count * 4,
            "stage4_unique_configs": topic_count * 6,
        },
        "notes": [
            "Stage1 is compressed to 4 requests per topic per agent-count setting.",
            "Stage2 is the dominant cost; pair batches are cached and reused.",
            "six_agents and targeted_attacks intentionally share no-graph judgments to save hits.",
            "full reuses six-agent targeted Stage1 and Stage2, then only adds Stage3 and graph-grounded Stage4.",
        ],
    }


def run_selected10_suite(
    *,
    model: str = "llama-3.3-70b-versatile",
    topic_file: Path | str = DEFAULT_TOPIC_FILE,
    output_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force: bool = False,
    pair_batch_size: int = 40,
) -> dict[str, Any]:
    topics = load_selected_topics(topic_file)
    root = Path(output_root or (ROOT / "outputs" / "groq_selected10"))
    paths = OutputPaths(root)
    client = GroqClient(model=model, cache_dir=cache_dir or (root / "cache"))

    run_baseline(client, topics, paths.stage4("single_llm"), mode="single", force=force)
    run_baseline(client, topics, paths.stage4("cot"), mode="cot", force=force)

    stage1_two = run_stage1(client, topics, paths.stage1("two_agents_shared"), n_pro=1, n_con=1, force=force)
    stage1_six = run_stage1(client, topics, paths.stage1("six_agents_shared"), n_pro=3, n_con=3, force=force)

    stage2_two_targeted = run_stage2(
        client,
        stage1_two,
        paths.stage2("two_agents"),
        targeted_attacks=True,
        pair_batch_size=pair_batch_size,
        force=force,
    )
    stage2_two_zeroshot = run_stage2(
        client,
        stage1_two,
        paths.stage2("dung_no_agents"),
        targeted_attacks=False,
        pair_batch_size=pair_batch_size,
        force=force,
    )
    stage2_six_targeted = run_stage2(
        client,
        stage1_six,
        paths.stage2("six_agents_shared"),
        targeted_attacks=True,
        pair_batch_size=pair_batch_size,
        force=force,
    )
    stage2_six_zeroshot = run_stage2(
        client,
        stage1_six,
        paths.stage2("direct_judge"),
        targeted_attacks=False,
        pair_batch_size=pair_batch_size,
        force=force,
    )

    stage3_two_targeted = run_stage3(stage2_two_targeted, paths.stage3("two_agents"), force=force)
    stage3_two_zeroshot = run_stage3(stage2_two_zeroshot, paths.stage3("dung_no_agents"), force=force)
    stage3_six_targeted = run_stage3(stage2_six_targeted, paths.stage3("full"), force=force)

    run_stage4(client, stage2_two_targeted, paths.stage4("two_agents"), stage3_path=stage3_two_targeted, force=force)
    run_stage4(client, stage2_two_zeroshot, paths.stage4("dung_no_agents"), stage3_path=stage3_two_zeroshot, force=force)
    run_stage4(client, stage2_six_zeroshot, paths.stage4("direct_judge"), stage3_path=None, force=force)

    six_no_graph = run_stage4(
        client,
        stage2_six_targeted,
        paths.stage4("six_agents"),
        stage3_path=None,
        force=force,
    )
    if force or not paths.stage4("targeted_attacks").exists():
        doc = load_json(six_no_graph)
        doc.setdefault("summary", {})["aliased_from"] = "six_agents"
        save_json(paths.stage4("targeted_attacks"), doc)

    run_stage4(client, stage2_six_targeted, paths.stage4("full"), stage3_path=stage3_six_targeted, force=force)

    report_path = write_suite_summary(paths)
    return {
        "output_root": str(root),
        "summary_json": str(report_path),
        "summary_rows": summarize_suite(paths),
        "plan_estimate": estimate_request_plan(topics),
    }


def run_nvidia_selected10_suite(
    *,
    model: str = "meta/llama-3.1-405b-instruct",
    topic_file: Path | str = DEFAULT_TOPIC_FILE,
    output_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    force: bool = False,
    pair_batch_size: int = 40,
) -> dict[str, Any]:
    """Run the full ablation suite using NVIDIA NIM API instead of Groq.

    Recommended models (largest → fastest):
      - "meta/llama-3.1-405b-instruct"          (405B, best reasoning)
      - "nvidia/llama-3.1-nemotron-ultra-253b-v1" (253B, NVIDIA-tuned)
      - "nvidia/llama-3.1-nemotron-70b-instruct"  (70B, good balance)
      - "meta/llama-3.3-70b-instruct"             (70B, fast)

    Short aliases also work: "llama-405b", "nemotron-ultra", "nemotron-70b".

    Outputs are written to outputs/nvidia_selected10/ by default so they
    never collide with the Groq run under outputs/groq_selected10/.
    """
    topics = load_selected_topics(topic_file)
    root = Path(output_root or (ROOT / "outputs" / "nvidia_selected10"))
    paths = OutputPaths(root)
    client = NvidiaClient(model=model, cache_dir=cache_dir or (root / "cache"))

    run_baseline(client, topics, paths.stage4("single_llm"), mode="single", force=force)
    run_baseline(client, topics, paths.stage4("cot"), mode="cot", force=force)

    stage1_two = run_stage1(client, topics, paths.stage1("two_agents_shared"), n_pro=1, n_con=1, force=force)
    stage1_six = run_stage1(client, topics, paths.stage1("six_agents_shared"), n_pro=3, n_con=3, force=force)

    stage2_two_targeted = run_stage2(
        client, stage1_two, paths.stage2("two_agents"),
        targeted_attacks=True, pair_batch_size=pair_batch_size, force=force,
    )
    stage2_two_zeroshot = run_stage2(
        client, stage1_two, paths.stage2("dung_no_agents"),
        targeted_attacks=False, pair_batch_size=pair_batch_size, force=force,
    )
    stage2_six_targeted = run_stage2(
        client, stage1_six, paths.stage2("six_agents_shared"),
        targeted_attacks=True, pair_batch_size=pair_batch_size, force=force,
    )
    stage2_six_zeroshot = run_stage2(
        client, stage1_six, paths.stage2("direct_judge"),
        targeted_attacks=False, pair_batch_size=pair_batch_size, force=force,
    )

    stage3_two_targeted = run_stage3(stage2_two_targeted, paths.stage3("two_agents"), force=force)
    stage3_two_zeroshot = run_stage3(stage2_two_zeroshot, paths.stage3("dung_no_agents"), force=force)
    stage3_six_targeted = run_stage3(stage2_six_targeted, paths.stage3("full"), force=force)

    run_stage4(client, stage2_two_targeted, paths.stage4("two_agents"), stage3_path=stage3_two_targeted, force=force)
    run_stage4(client, stage2_two_zeroshot, paths.stage4("dung_no_agents"), stage3_path=stage3_two_zeroshot, force=force)
    run_stage4(client, stage2_six_zeroshot, paths.stage4("direct_judge"), stage3_path=None, force=force)

    six_no_graph = run_stage4(
        client, stage2_six_targeted, paths.stage4("six_agents"), stage3_path=None, force=force,
    )
    if force or not paths.stage4("targeted_attacks").exists():
        doc = load_json(six_no_graph)
        doc.setdefault("summary", {})["aliased_from"] = "six_agents"
        save_json(paths.stage4("targeted_attacks"), doc)

    run_stage4(client, stage2_six_targeted, paths.stage4("full"), stage3_path=stage3_six_targeted, force=force)

    report_path = write_suite_summary(paths)
    return {
        "output_root": str(root),
        "summary_json": str(report_path),
        "summary_rows": summarize_suite(paths),
        "plan_estimate": estimate_request_plan(topics),
        "model": client.model,
        "provider": "nvidia",
    }
