"""
Single-topic debate pipeline for the MAJ Debate web app.

Adapted from groq_selected10_notebook.py:
  - Processes one topic at a time (no batch checkpointing)
  - Accepts an on_progress(event_dict) callback for SSE streaming
  - NvidiaClient is self-contained; no Groq imports needed
  - stage3_graph is imported from the same directory (copied in by Docker)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from stage3_graph import process_topic as _stage3_process_topic
    STAGE3_AVAILABLE = True
except ImportError:
    _stage3_process_topic = None  # type: ignore[assignment]
    STAGE3_AVAILABLE = False


# ── Available models ──────────────────────────────────────────────────────────

NVIDIA_MODELS: dict[str, str] = {
    "meta/llama-3.1-405b-instruct":            "Llama 3.1 405B — Best reasoning",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": "Nemotron Ultra 253B — NVIDIA-tuned",
    "nvidia/llama-3.1-nemotron-70b-instruct":  "Nemotron 70B — Fast & capable",
    "meta/llama-3.3-70b-instruct":             "Llama 3.3 70B — Efficient",
    "mistralai/mixtral-8x22b-instruct-v0.1":   "Mixtral 8×22B — Mixture-of-Experts",
}


# ── Personas ──────────────────────────────────────────────────────────────────

PRO_PERSONAS: list[dict] = [
    {
        "id": "pro_rationalist", "name": "Rationalist Pro", "stance": "PRO",
        "reasoning_style": "logical-empirical",
        "rhetorical_mode": "cite quantitative evidence and causal mechanisms",
        "description": "Argues from data, statistics, and formal logic. Prioritises measurable outcomes.",
    },
    {
        "id": "pro_ethicist", "name": "Ethics Advocate Pro", "stance": "PRO",
        "reasoning_style": "ethical-normative",
        "rhetorical_mode": "appeal to moral principles and rights-based frameworks",
        "description": "Argues from fairness and justice. References established ethical frameworks.",
    },
    {
        "id": "pro_futurist", "name": "Futurist Pro", "stance": "PRO",
        "reasoning_style": "economic-consequentialist",
        "rhetorical_mode": "project long-term societal and economic benefits",
        "description": "Argues from systemic benefits and long-horizon impact.",
    },
]

CON_PERSONAS: list[dict] = [
    {
        "id": "con_skeptic", "name": "Skeptic Con", "stance": "CON",
        "reasoning_style": "logical-empirical",
        "rhetorical_mode": "challenge evidence quality and burden of proof",
        "description": "Contests factual claims, demands rigorous evidence, identifies fallacies.",
    },
    {
        "id": "con_rights", "name": "Rights Defender Con", "stance": "CON",
        "reasoning_style": "ethical-normative",
        "rhetorical_mode": "appeal to human rights, procedural justice, and democratic accountability",
        "description": "Argues the proposal violates fundamental rights regardless of practical merits.",
    },
    {
        "id": "con_pragmatist", "name": "Pragmatist Con", "stance": "CON",
        "reasoning_style": "economic-consequentialist",
        "rhetorical_mode": "highlight implementation barriers and unintended consequences",
        "description": "Argues from practical constraints: cost, feasibility, second-order effects.",
    },
]


# ── System prompts ────────────────────────────────────────────────────────────

STAGE1_SYSTEM = (
    "You are a careful debate agent. Respond ONLY with valid JSON. "
    "No preamble, no prose, no markdown fences."
)
STAGE2_SYSTEM = (
    "You are a careful debate analyst. Respond ONLY with the requested JSON. "
    "No reasoning prose, no markdown fences, just the JSON."
)
STAGE4_SYSTEM = (
    "You are an impartial debate judge. Evaluate arguments on the basis of "
    "logical structure and which side has more undefeated claims. "
    "Respond ONLY with valid JSON. Keep rationale under 40 words and "
    "list at most 5 killing_attacks."
)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def parse_json_object(text: str) -> dict:
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
                    return json.loads(cleaned[start: i + 1])
                except Exception:
                    return {}
    return {}


def parse_json_array(text: str) -> list:
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
                    return json.loads(cleaned[start: i + 1])
                except Exception:
                    return []
    return []


def coerce_label(raw: Any) -> str:
    value = str(raw or "").strip().title()
    return value if value in {"Attack", "Support", "Neutral", "None"} else "None"


def coerce_float01(raw: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return default


def generate_topic_id(topic_text: str) -> str:
    return "CUSTOM_" + hashlib.md5(topic_text.encode()).hexdigest()[:8].upper()


# ── NVIDIA API client ─────────────────────────────────────────────────────────

class NvidiaClient:
    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        model: str = "meta/llama-3.1-405b-instruct",
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        timeout: int = 180,
        max_retries: int = 5,
    ) -> None:
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("NVIDIA_API_KEY", "")
            or os.environ.get("NIM_API_KEY", "")
        ).strip()
        if not self.api_key:
            raise ValueError(
                "No NVIDIA API key found. Set NVIDIA_API_KEY env var or pass api_key=."
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir or (Path(__file__).parent / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": self.model, "system": system, "user": user,
             "temperature": temperature, "max_tokens": max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _parse_retry_after(self, response: Any) -> int | None:
        header = (
            response.headers.get("Retry-After")
            or response.headers.get("retry-after")
        )
        if header:
            try:
                return max(int(float(header)), 1)
            except (TypeError, ValueError):
                pass
        m = re.search(
            r"try again in\s+(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?",
            response.text or "",
            re.IGNORECASE,
        )
        if m:
            minutes = int(m.group(1) or 0)
            seconds = math.ceil(float(m.group(2) or 0))
            return max(minutes * 60 + seconds + 5, 1)
        return None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 700,
        namespace: str = "default",
        use_cache: bool = True,
    ) -> dict:
        key = self._cache_key(system_prompt, user_prompt, temperature, max_tokens)
        cache_path = self.cache_dir / namespace / f"{key}.json"
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        payload = {
            "model": self.model,
            "temperature": max(temperature, 1e-8),
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }

        error_msg = "NVIDIA API request failed"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    error_msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    wait = (
                        self._parse_retry_after(resp) or 60
                        if resp.status_code == 429
                        else min(2 ** attempt, 15)
                    )
                    print(f"[retry {attempt}/{self.max_retries}] {error_msg[:80]} — waiting {wait}s")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                result = {
                    "content": content,
                    "usage": data.get("usage", {}),
                    "model": data.get("model", self.model),
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(result), encoding="utf-8")
                return result
            except Exception as exc:
                error_msg = str(exc)
                wait = (
                    60 if "429" in error_msg or "rate limit" in error_msg.lower()
                    else min(2 ** attempt, 15)
                )
                if attempt == self.max_retries:
                    break
                print(f"[retry {attempt}/{self.max_retries}] {error_msg[:80]} — waiting {wait}s")
                time.sleep(wait)
        raise RuntimeError(error_msg)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _persona_block(personas: list) -> str:
    return "\n".join(
        f"- {p['id']} | {p['name']} | stance={p['stance']} | "
        f"style={p['reasoning_style']} | mode={p['rhetorical_mode']} | "
        f"profile={p['description']}"
        for p in personas
    )


def build_r1_prompt(topic: str, personas: list, n_args: int) -> str:
    ids = ", ".join(p["id"] for p in personas)
    return (
        f'Debate topic: "{topic}"\n\n'
        f"Generate arguments for these personas:\n{_persona_block(personas)}\n\n"
        f"For EACH persona, generate exactly {n_args} distinct single-sentence "
        f"arguments, max 40 words each.\n\n"
        f"Output ONLY JSON as an object keyed by persona_id. "
        f"Use exactly these keys: {ids}\n"
        f'Example: {{"{personas[0]["id"]}": ["arg 1", "arg 2"]}}'
    )


def build_r2_prompt(topic: str, personas: list, opposing: list, n_args: int) -> str:
    numbered = "\n".join(f"[{i+1}] {a}" for i, a in enumerate(opposing))
    ids = ", ".join(p["id"] for p in personas)
    return (
        f'Debate topic: "{topic}"\n\n'
        f"Opposing arguments:\n{numbered}\n\n"
        f"Generate counter-arguments for:\n{_persona_block(personas)}\n\n"
        f"For EACH persona, generate exactly {n_args} targeted counter-arguments. "
        f"Each must be one sentence, max 30 words, and include targets_arg as the "
        f"1-based index of the opposing argument it attacks.\n\n"
        f"Output ONLY JSON keyed by persona_id. "
        f'Each value is a list of {{"targets_arg": 1, "argument": "..."}}. '
        f"Use exactly these keys: {ids}"
    )


def build_strength_prompt(topic: str, args: list) -> str:
    lines = "\n".join(
        f'ARG {i} ({a["arg_id"]}): {(a.get("text") or "").replace(chr(10), " ")}'
        for i, a in enumerate(args)
    )
    return (
        f'Topic: "{topic}"\n\nRate each argument\'s strength from 0.0 to 1.0.\n\n'
        f"{lines}\n\n"
        f'Output ONLY a JSON array: [{{"arg": 0, "strength": 0.7}}, ...]'
    )


def build_pair_prompt(topic: str, pairs: list, *, targeted: bool) -> str:
    lines = []
    for i, (src, tgt) in enumerate(pairs):
        st = (src.get("text") or "").replace("\n", " ")
        tt = (tgt.get("text") or "").replace("\n", " ")
        if targeted:
            lines.append(
                f"PAIR {i}:\n"
                f'  SOURCE ({src["arg_id"]}, stance={src.get("stance")}): {st}\n'
                f'  TARGET ({tgt["arg_id"]}, stance={tgt.get("stance")}): {tt}'
            )
        else:
            lines.append(
                f"PAIR {i}:\n"
                f'  SOURCE ({src["arg_id"]}): {st}\n'
                f'  TARGET ({tgt["arg_id"]}): {tt}'
            )
    body = "\n".join(lines)
    if targeted:
        return (
            f'Topic: "{topic}"\n\n'
            f"For each pair, identify the implicit premise of TARGET that SOURCE is "
            f"challenging or reinforcing, then choose: Attack, Support, Neutral, or None.\n\n"
            f"{body}\n\n"
            f'Output ONLY JSON array: [{{"pair": 0, "label": "Attack|Support|Neutral|None", '
            f'"confidence": 0.0-1.0, "premise": "short phrase"}}, ...]'
        )
    return (
        f'Topic: "{topic}"\n\n'
        f"For each pair decide whether SOURCE attacks, supports, is neutral to, "
        f"or has no relation with TARGET.\n\n"
        f"{body}\n\n"
        f'Output ONLY JSON array: [{{"pair": 0, "label": "Attack|Support|Neutral|None", '
        f'"confidence": 0.0-1.0}}, ...]'
    )


def build_stage4_prompt(
    topic: str,
    args: list,
    rels: list,
    arg_strength: dict,
    stage3_graph: dict | None,
    conf_threshold: float,
) -> str:
    arg_lines = []
    for arg in args[:30]:
        text = (arg.get("text") or "")[:160].replace("\n", " ")
        info = arg_strength.get(arg["arg_id"], {})
        if isinstance(info, dict) and "strength" in info:
            arg_lines.append(
                f'  {arg["arg_id"]} [{arg.get("stance")}, s={info["strength"]:.2f}]: {text}'
            )
        else:
            arg_lines.append(f'  {arg["arg_id"]} [{arg.get("stance")}]: {text}')

    kept = sorted(
        [r for r in rels if r.get("kept") and r.get("label") in {"Attack", "Support"}],
        key=lambda r: r.get("confidence", 0.0),
        reverse=True,
    )
    rel_lines = [
        f'  {r["source_arg_id"]} --{r["label"]}--> {r["target_arg_id"]} '
        f'(c={r.get("confidence", 0.0):.2f})'
        for r in kept[:60]
    ]

    graph_block = ""
    if stage3_graph:
        gv = stage3_graph.get("graph_verdict", {})
        skeptical = [
            aid for aid, acc in (stage3_graph.get("acceptance") or {}).items()
            if acc.get("skeptical")
        ]
        graph_block = (
            "\nDUNG-SEMANTICS RESULTS:\n"
            f'  grounded extension ({len(stage3_graph.get("grounded_extension", []))}): '
            f'{stage3_graph.get("grounded_extension", [])[:15]}\n'
            f'  preferred extensions: {stage3_graph.get("n_preferred")}\n'
            f"  skeptically accepted: {skeptical[:15]}\n"
            f'  formal verdict: {gv.get("winner")} '
            f'(pro={gv.get("pro_score")}, con={gv.get("con_score")}, '
            f'basis={gv.get("basis")})\n'
        )

    return (
        f'Topic: "{topic}"\n\n'
        f"ARGUMENTS:\n{chr(10).join(arg_lines)}\n\n"
        f"HIGH-CONFIDENCE RELATIONS (>= {conf_threshold}):\n"
        f'{chr(10).join(rel_lines) if rel_lines else "  (none)"}\n'
        f"{graph_block}\n"
        f'Decide the winner and output ONLY JSON: {{"verdict": "PRO|CON|TIE", '
        f'"confidence": 0.0-1.0, "rationale": "40 words max", '
        f'"killing_attacks": ["SRC -> TGT", "..."]}}'
    )


# ── Pair prefiltering (token-similarity fallback) ─────────────────────────────

def _token_set(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def compute_similarity(args: list) -> list[list[float]]:
    token_sets = [_token_set(a.get("text") or "") for a in args]
    matrix: list[list[float]] = []
    for left in token_sets:
        row: list[float] = []
        for right in token_sets:
            union = left | right
            row.append(len(left & right) / max(len(union), 1))
        matrix.append(row)
    return matrix


def _none_rel(src: dict, tgt: dict, reason: str) -> dict:
    return {
        "source_arg_id": src["arg_id"], "target_arg_id": tgt["arg_id"],
        "source_stance": src.get("stance"), "target_stance": tgt.get("stance"),
        "source_round": src.get("round"), "target_round": tgt.get("round"),
        "label": "None", "confidence": 0.0, "kept": False,
        "justification": f"prefiltered:{reason}", "prefiltered": True,
    }


def prefilter_pairs(
    args: list,
    sim: list,
    *,
    max_round_gap: int = 4,
    min_sim: float = 0.12,
    same_stance_min: float = 0.30,
) -> tuple[list, list]:
    keep: list = []
    auto_none: list = []
    for i, j in permutations(range(len(args)), 2):
        src, tgt = args[i], args[j]
        score = float(sim[i][j])
        r_gap = abs((src.get("round") or 0) - (tgt.get("round") or 0))
        if r_gap > max_round_gap:
            auto_none.append(_none_rel(src, tgt, "round_gap"))
            continue
        if src.get("stance") == tgt.get("stance") and score < same_stance_min:
            auto_none.append(_none_rel(src, tgt, "same_stance_low_sim"))
            continue
        if score < min_sim:
            auto_none.append(_none_rel(src, tgt, "low_sim"))
            continue
        keep.append((src, tgt))
    return keep, auto_none


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


# ── Stage runners (single-topic) ──────────────────────────────────────────────

def run_stage1(
    client: NvidiaClient,
    topic_id: str,
    topic_text: str,
    config: dict,
    on_progress: Callable,
) -> dict:
    n_pro = min(config.get("n_pro", 3), 3)
    n_con = min(config.get("n_con", 3), 3)
    r1_args = config.get("r1_args", 3)
    r2_args = config.get("r2_args", 2)
    active_pro = PRO_PERSONAS[:n_pro]
    active_con = CON_PERSONAS[:n_con]

    on_progress({"type": "progress", "stage": 1, "step": "r1_pro",
                 "message": "Generating PRO Round 1 arguments…"})
    pro_r1 = parse_json_object(
        client.complete(STAGE1_SYSTEM, build_r1_prompt(topic_text, active_pro, r1_args),
                        temperature=0.3, max_tokens=900, namespace="s1_r1")["content"]
    )

    on_progress({"type": "progress", "stage": 1, "step": "r1_con",
                 "message": "Generating CON Round 1 arguments…"})
    con_r1 = parse_json_object(
        client.complete(STAGE1_SYSTEM, build_r1_prompt(topic_text, active_con, r1_args),
                        temperature=0.3, max_tokens=900, namespace="s1_r1")["content"]
    )

    pro_by_pid = {
        p["id"]: [str(x).strip() for x in (pro_r1.get(p["id"]) or []) if str(x).strip()][:r1_args]
        for p in active_pro
    }
    con_by_pid = {
        p["id"]: [str(x).strip() for x in (con_r1.get(p["id"]) or []) if str(x).strip()][:r1_args]
        for p in active_con
    }
    all_pro_texts = [t for pid in pro_by_pid for t in pro_by_pid[pid]]
    all_con_texts = [t for pid in con_by_pid for t in con_by_pid[pid]]

    on_progress({"type": "progress", "stage": 1, "step": "r2_pro",
                 "message": "Generating PRO Round 2 counter-arguments…"})
    pro_r2 = parse_json_object(
        client.complete(STAGE1_SYSTEM, build_r2_prompt(topic_text, active_pro, all_con_texts, r2_args),
                        temperature=0.3, max_tokens=1100, namespace="s1_r2")["content"]
    )

    on_progress({"type": "progress", "stage": 1, "step": "r2_con",
                 "message": "Generating CON Round 2 counter-arguments…"})
    con_r2 = parse_json_object(
        client.complete(STAGE1_SYSTEM, build_r2_prompt(topic_text, active_con, all_pro_texts, r2_args),
                        temperature=0.3, max_tokens=1100, namespace="s1_r2")["content"]
    )

    flat: list[dict] = []
    idx = 0
    for persona in active_pro:
        for text in pro_by_pid.get(persona["id"], []):
            flat.append({"arg_id": f"{topic_id}_A{idx:03d}", "persona_id": persona["id"],
                         "persona": persona["name"], "stance": "PRO", "round": 1,
                         "targets_arg": None, "text": text})
            idx += 1
    for persona in active_con:
        for text in con_by_pid.get(persona["id"], []):
            flat.append({"arg_id": f"{topic_id}_A{idx:03d}", "persona_id": persona["id"],
                         "persona": persona["name"], "stance": "CON", "round": 1,
                         "targets_arg": None, "text": text})
            idx += 1
    for persona in active_pro:
        for item in (pro_r2.get(persona["id"]) or [])[:r2_args]:
            text = str(item.get("argument", item) if isinstance(item, dict) else item).strip()
            target = item.get("targets_arg") if isinstance(item, dict) else None
            if text:
                flat.append({"arg_id": f"{topic_id}_A{idx:03d}", "persona_id": persona["id"],
                             "persona": persona["name"], "stance": "PRO", "round": 2,
                             "targets_arg": target, "text": text})
                idx += 1
    for persona in active_con:
        for item in (con_r2.get(persona["id"]) or [])[:r2_args]:
            text = str(item.get("argument", item) if isinstance(item, dict) else item).strip()
            target = item.get("targets_arg") if isinstance(item, dict) else None
            if text:
                flat.append({"arg_id": f"{topic_id}_A{idx:03d}", "persona_id": persona["id"],
                             "persona": persona["name"], "stance": "CON", "round": 2,
                             "targets_arg": target, "text": text})
                idx += 1

    on_progress({"type": "stage_done", "stage": 1,
                 "data": {"arguments": flat, "n_arguments": len(flat)}})
    return {"topic_id": topic_id, "topic_text": topic_text, "arguments": flat}


def run_stage2(
    client: NvidiaClient,
    stage1: dict,
    config: dict,
    on_progress: Callable,
) -> dict:
    topic_id = stage1["topic_id"]
    topic_text = stage1["topic_text"]
    args = stage1["arguments"]
    targeted = config.get("targeted_attacks", True)
    conf_threshold = float(config.get("confidence_threshold", 0.65))
    batch_size = int(config.get("pair_batch_size", 40))

    on_progress({"type": "progress", "stage": 2, "step": "similarity",
                 "message": "Computing argument similarity matrix…"})
    sim = compute_similarity(args)

    on_progress({"type": "progress", "stage": 2, "step": "prefilter",
                 "message": "Prefiltering argument pairs…"})
    keep_pairs, auto_none = prefilter_pairs(args, sim)

    on_progress({"type": "progress", "stage": 2, "step": "strength",
                 "message": "Scoring argument strength…"})
    raw_strength = client.complete(
        STAGE2_SYSTEM, build_strength_prompt(topic_text, args),
        temperature=0.1, max_tokens=900, namespace="s2_strength",
    )["content"]
    strength_arr = parse_json_array(raw_strength)
    strength_by_idx: dict[int, dict] = {}
    for item in strength_arr:
        if isinstance(item, dict) and "arg" in item:
            try:
                strength_by_idx[int(item["arg"])] = item
            except (TypeError, ValueError):
                pass
    arg_strength: dict[str, dict] = {}
    for i, arg in enumerate(args):
        item = strength_by_idx.get(i, {})
        arg_strength[arg["arg_id"]] = {
            "strength": round(coerce_float01(item.get("strength", 0.5), 0.5), 3),
            "rationale": str(item.get("rationale", ""))[:200],
        }

    relations: list[dict] = list(auto_none)
    n_batches = max(math.ceil(len(keep_pairs) / batch_size), 1)
    for b_idx, batch in enumerate(_chunks(keep_pairs, batch_size)):
        on_progress({"type": "progress", "stage": 2, "step": f"pairs_{b_idx}",
                     "message": f"Classifying pairs (batch {b_idx + 1}/{n_batches})…"})
        raw = client.complete(
            STAGE2_SYSTEM, build_pair_prompt(topic_text, batch, targeted=targeted),
            temperature=0.1, max_tokens=1200,
            namespace="s2_pairs_targeted" if targeted else "s2_pairs_zeroshot",
        )["content"]
        arr = parse_json_array(raw)
        by_idx: dict[int, dict] = {}
        for item in arr:
            if isinstance(item, dict) and "pair" in item:
                try:
                    by_idx[int(item["pair"])] = item
                except (TypeError, ValueError):
                    pass
        for i, (src, tgt) in enumerate(batch):
            item = by_idx.get(i)
            if item is None:
                relations.append({
                    "source_arg_id": src["arg_id"], "target_arg_id": tgt["arg_id"],
                    "source_stance": src.get("stance"), "target_stance": tgt.get("stance"),
                    "source_round": src.get("round"), "target_round": tgt.get("round"),
                    "label": "None", "confidence": 0.0, "kept": False,
                    "justification": "parse_failed",
                })
                continue
            rel: dict = {
                "source_arg_id": src["arg_id"], "target_arg_id": tgt["arg_id"],
                "source_stance": src.get("stance"), "target_stance": tgt.get("stance"),
                "source_round": src.get("round"), "target_round": tgt.get("round"),
                "label": coerce_label(item.get("label")),
                "confidence": round(coerce_float01(item.get("confidence", 0.0)), 3),
            }
            rel["kept"] = rel["confidence"] >= conf_threshold and rel["label"] != "None"
            if "premise" in item:
                rel["premise"] = str(item.get("premise", ""))[:200]
            relations.append(rel)

    label_counts = dict(Counter(r.get("label", "None") for r in relations))
    kept_count = sum(int(r.get("kept", False)) for r in relations)
    result = {
        "topic_id": topic_id, "topic_text": topic_text,
        "arguments": args, "argument_strength": arg_strength, "relations": relations,
        "summary": {
            "n_arguments": len(args), "n_relations": len(relations),
            "kept_relations": kept_count, "label_counts": label_counts,
            "targeted_attacks": targeted, "confidence_threshold": conf_threshold,
        },
    }
    on_progress({"type": "stage_done", "stage": 2, "data": {
        "argument_strength": arg_strength, "relations": relations,
        "arguments": args, "summary": result["summary"],
    }})
    return result


def run_stage3(stage2: dict, config: dict, on_progress: Callable) -> dict | None:
    if not STAGE3_AVAILABLE:
        on_progress({"type": "progress", "stage": 3, "step": "skip",
                     "message": "Stage 3 unavailable (stage3_graph module not found)."})
        return None
    conf = float(config.get("confidence_threshold", 0.65))
    # Semantic sets: PRO arguments form one set, CON another.
    # Attacks are only valid across stances (PRO↔CON), never within the same set.
    on_progress({"type": "progress", "stage": 3, "step": "dung",
                 "message": "Computing Dung semantics (cross-stance attacks only)…"})
    result = _stage3_process_topic(stage2, conf, cross_stance_only=True)
    on_progress({"type": "stage_done", "stage": 3, "data": result})
    return result


def run_stage4(
    client: NvidiaClient,
    stage2: dict,
    stage3: dict | None,
    config: dict,
    on_progress: Callable,
) -> dict:
    topic_text = stage2["topic_text"]
    args = stage2["arguments"]
    rels = stage2["relations"]
    arg_strength = stage2.get("argument_strength", {})
    conf = float(config.get("confidence_threshold", 0.65))

    on_progress({"type": "progress", "stage": 4, "step": "judging",
                 "message": "LLM judge evaluating the debate…"})
    prompt = build_stage4_prompt(topic_text, args, rels, arg_strength, stage3, conf)
    raw = client.complete(
        STAGE4_SYSTEM, prompt,
        temperature=0.1, max_tokens=700,
        namespace="s4_graph" if stage3 else "s4_nograph",
    )["content"]

    obj = parse_json_object(raw)
    if not obj or "verdict" not in obj:
        m = re.search(r'"verdict"\s*:\s*"(PRO|CON|TIE)"', raw or "", re.IGNORECASE)
        obj = {
            "verdict": m.group(1).upper() if m else "TIE",
            "confidence": 0.5 if m else 0.0,
            "rationale": "[recovered from partial response]",
            "killing_attacks": [],
        }

    verdict = str(obj.get("verdict", "TIE")).strip().upper()
    if verdict not in {"PRO", "CON", "TIE"}:
        verdict = "TIE"

    result: dict = {
        "verdict": verdict,
        "confidence": round(coerce_float01(obj.get("confidence", 0.0)), 3),
        "rationale": str(obj.get("rationale", ""))[:400],
        "killing_attacks": [str(x) for x in (obj.get("killing_attacks") or [])][:5],
        "used_graph": stage3 is not None,
        "graph_verdict": (stage3 or {}).get("graph_verdict"),
    }
    on_progress({"type": "stage_done", "stage": 4, "data": result})
    return result


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run_debate(topic_text: str, config: dict, on_progress: Callable) -> dict:
    """
    Run the full debate pipeline for a single topic.

    config keys
    -----------
    model               str   NVIDIA model id
    api_key             str   optional override (falls back to env var)
    n_pro               int   number of PRO personas  (1-3)
    n_con               int   number of CON personas  (1-3)
    r1_args             int   Round-1 args per persona
    r2_args             int   Round-2 counter-args per persona
    targeted_attacks    bool  use targeted-attack prompt in Stage 2
    confidence_threshold float relation confidence cutoff
    pair_batch_size     int   pairs per Stage-2 API call
    cross_stance_only   bool  drop same-stance attacks before Dung
    run_stage1/2/3/4    bool  which stages to execute

    on_progress(event_dict) is called at every significant step.
    Returns the full result dict.
    """
    topic_id = generate_topic_id(topic_text)
    client = NvidiaClient(
        model=config.get("model", "meta/llama-3.1-405b-instruct"),
        api_key=config.get("api_key") or None,
        cache_dir=Path(__file__).parent / "cache",
    )

    on_progress({
        "type": "start",
        "topic_id": topic_id,
        "topic_text": topic_text,
        "model": client.model,
        "config": {k: v for k, v in config.items() if k != "api_key"},
    })

    s1 = s2 = s3 = s4 = None

    if config.get("run_stage1", True):
        s1 = run_stage1(client, topic_id, topic_text, config, on_progress)

    if config.get("run_stage2", True) and s1:
        s2 = run_stage2(client, s1, config, on_progress)

    if config.get("run_stage3", True) and s2:
        s3 = run_stage3(s2, config, on_progress)

    if config.get("run_stage4", True) and s2:
        s4 = run_stage4(client, s2, s3, config, on_progress)

    full = {
        "topic_id": topic_id, "topic_text": topic_text, "model": client.model,
        "stage1": s1, "stage2": s2, "stage3": s3, "stage4": s4,
    }
    on_progress({"type": "complete", "data": full})
    return full
