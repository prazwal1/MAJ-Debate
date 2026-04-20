#!/usr/bin/env python3
"""
Stage 1 vLLM — persona-driven argument generation (replaces OpenRouter).

Produces outputs/stage1/<split>/stage1_arguments.json in the exact schema
that stage2_vllm_shard.py expects:

    {
      "topics": [
        {
          "topic_id": "...", "topic_text": "...", "domain": "...",
          "benchmark_label": "...", "source_dataset": "...",
          "source_ref": "...", "evaluation_split": "...", "run_name": "...",
          "arguments": [
             {"arg_id": "<tid>_A000", "persona_id": "pro_rationalist",
              "persona": "Rationalist Pro", "stance": "PRO",
              "round": 1, "targets_arg": null, "text": "..."},
             ...
          ],
          "meta": {...}
        },
        ...
      ],
      "summary": {...}
    }

Runs on ONE GPU (CUDA_VISIBLE_DEVICES is honored). Uses batched vLLM
generation: all (topic x persona x round) prompts are flattened into one
engine.generate() call per round, which is ~30x faster than per-prompt.

Arguments:
    --split <name>          evaluation split name (used for run_name + output subdir)
    --topic-file <path>     .jsonl or .json with topic records
    --output <path>         where to write stage1_arguments.json
    --model <path>          local HF model dir (e.g. ~/models/Qwen2.5-3B-Instruct)
    --n-pro <int>           number of PRO agents (1..3)
    --n-con <int>           number of CON agents (1..3)
    --r1-args <int>         round-1 arguments per agent
    --r2-args <int>         round-2 counter-arguments per agent
    --topic-limit <int>     0 = all
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Personas (mirrors the notebook exactly so results are comparable)
# ---------------------------------------------------------------------------

PRO_PERSONAS = [
    {'id': 'pro_rationalist', 'name': 'Rationalist Pro', 'stance': 'PRO',
     'reasoning_style': 'logical-empirical',
     'rhetorical_mode': 'cite quantitative evidence and causal mechanisms',
     'description': 'Argues from data, statistics, and formal logic. Prioritises measurable outcomes.'},
    {'id': 'pro_ethicist', 'name': 'Ethics Advocate Pro', 'stance': 'PRO',
     'reasoning_style': 'ethical-normative',
     'rhetorical_mode': 'appeal to moral principles and rights-based frameworks',
     'description': 'Argues from fairness and justice. References established ethical frameworks.'},
    {'id': 'pro_futurist', 'name': 'Futurist Pro', 'stance': 'PRO',
     'reasoning_style': 'economic-consequentialist',
     'rhetorical_mode': 'project long-term societal and economic benefits',
     'description': 'Argues from systemic benefits and long-horizon impact. Accepts short-term trade-offs.'},
]
CON_PERSONAS = [
    {'id': 'con_skeptic', 'name': 'Skeptic Con', 'stance': 'CON',
     'reasoning_style': 'logical-empirical',
     'rhetorical_mode': 'challenge evidence quality and burden of proof',
     'description': 'Contests factual claims, demands rigorous evidence, identifies logical fallacies.'},
    {'id': 'con_rights', 'name': 'Rights Defender Con', 'stance': 'CON',
     'reasoning_style': 'ethical-normative',
     'rhetorical_mode': 'appeal to human rights, procedural justice, and democratic accountability',
     'description': 'Argues the proposal violates fundamental rights regardless of practical merits.'},
    {'id': 'con_pragmatist', 'name': 'Pragmatist Con', 'stance': 'CON',
     'reasoning_style': 'economic-consequentialist',
     'rhetorical_mode': 'highlight implementation barriers and unintended consequences',
     'description': 'Argues from practical constraints: cost, feasibility, second-order effects.'},
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    'You are a careful debate agent. Respond ONLY with a valid JSON array. '
    'No preamble, no prose, no markdown fences, no chain of thought. Just the JSON.'
)


def build_r1_prompt(topic, persona, n_args):
    return (
        f'You are a debate agent with the following profile:\n'
        f'- Name            : {persona["name"]}\n'
        f'- Stance          : {persona["stance"]} (argue FOR this position only)\n'
        f'- Reasoning style : {persona["reasoning_style"]}\n'
        f'- Rhetorical mode : {persona["rhetorical_mode"]}\n'
        f'- Profile         : {persona["description"]}\n\n'
        f'Debate topic: "{topic}"\n\n'
        f'Generate exactly {n_args} distinct high-quality arguments for the '
        f'{persona["stance"]} position.\n'
        f'Each argument must be a single clear sentence, max 40 words, '
        f'substantively different from the others.\n\n'
        f'Output ONLY a valid JSON array of strings:\n'
        f'["argument 1", "argument 2", "argument 3"]'
    )


def build_r2_prompt(topic, persona, opposing_args, n_args):
    numbered = '\n'.join(f'  [{i + 1}] {a}' for i, a in enumerate(opposing_args))
    return (
        f'You are a debate agent with the following profile:\n'
        f'- Name            : {persona["name"]}\n'
        f'- Stance          : {persona["stance"]}\n'
        f'- Reasoning style : {persona["reasoning_style"]}\n'
        f'- Rhetorical mode : {persona["rhetorical_mode"]}\n\n'
        f'Debate topic: "{topic}"\n\n'
        f'The opposing side has made these arguments (read-only context):\n'
        f'{numbered}\n\n'
        f'Generate exactly {n_args} targeted counter-arguments from your '
        f'{persona["stance"]} position.\n'
        f'Each counter-argument must directly attack one specific opposing '
        f'argument, be a single clear sentence, and be at most 30 words.\n\n'
        f'Output ONLY a valid JSON array of objects:\n'
        f'[{{"targets_arg": 1, "argument": "..."}}]'
    )


# ---------------------------------------------------------------------------
# JSON parsing (tolerant — Qwen sometimes wraps in ```json ... ```)
# ---------------------------------------------------------------------------

def parse_json_list(text):
    if not text:
        return []
    t = re.sub(r'```(?:json)?', '', text).strip().strip('`').strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, list) else []
    except Exception:
        pass
    # bracket-balanced fallback
    start = t.find('[')
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(t)):
        if t[i] == '[':
            depth += 1
        elif t[i] == ']':
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[start:i + 1])
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
    return []


# ---------------------------------------------------------------------------
# Topic loading (mirrors notebook's normalize_topic)
# ---------------------------------------------------------------------------

def normalize_topic(raw, idx, split):
    tid = raw.get('topic_id') or raw.get('id') or f'{split.upper()}_{idx:04d}'
    text = raw.get('topic_text') or raw.get('text')
    if not text:
        raise ValueError(f'missing topic_text/text for record {tid}')
    return {
        'topic_id': tid,
        'topic_text': text,
        'domain': raw.get('domain', 'unknown'),
        'benchmark_label': raw.get('benchmark_label'),
        'source_dataset': raw.get('source_dataset', split),
        'source_ref': raw.get('source_ref'),
    }


def load_topics(path, split):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'topic file not found: {p}')
    topics = []
    if p.suffix.lower() == '.jsonl':
        with open(p) as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    topics.append(normalize_topic(json.loads(line), idx, split))
    else:
        with open(p) as f:
            data = json.load(f)
        rows = data.get('topics') or data.get('data') or data
        for idx, r in enumerate(rows, 1):
            topics.append(normalize_topic(r, idx, split))
    return topics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s stage1 %(levelname)-7s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for lib in ['transformers', 'torch', 'vllm', 'urllib3', 'httpx']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger('stage1')


def format_chat(prompt, tokenizer):
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True)
    ap.add_argument('--topic-file', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-pro', type=int, default=3)
    ap.add_argument('--n-con', type=int, default=3)
    ap.add_argument('--r1-args', type=int, default=3)
    ap.add_argument('--r2-args', type=int, default=2)
    ap.add_argument('--topic-limit', type=int, default=0)
    ap.add_argument('--gpu-mem-util', type=float, default=0.80)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=512)
    args = ap.parse_args()

    log = setup_logging()
    log.info('=' * 70)
    log.info('Stage 1 vLLM — n_pro=%d n_con=%d r1=%d r2=%d',
             args.n_pro, args.n_con, args.r1_args, args.r2_args)
    log.info('model=%s', args.model)
    log.info('=' * 70)

    active_pro = PRO_PERSONAS[:args.n_pro]
    active_con = CON_PERSONAS[:args.n_con]
    all_personas = active_pro + active_con
    log.info('Personas: %s', [p['id'] for p in all_personas])

    topics = load_topics(args.topic_file, args.split)
    log.info('Loaded %d topics from %s', len(topics), args.topic_file)
    if args.topic_limit > 0:
        topics = topics[:args.topic_limit]
        log.info('Topic limit -> %d topics', len(topics))

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    existing = []
    if out_path.exists():
        try:
            doc = json.loads(out_path.read_text())
            existing = doc.get('topics', [])
            done_ids = {t['topic_id'] for t in existing}
            log.info('Resuming: %d topics already present', len(done_ids))
        except Exception:
            pass

    remaining = [t for t in topics if t['topic_id'] not in done_ids]
    if not remaining:
        log.info('All topics done. Nothing to generate.')
        return

    log.info('Initialising vLLM (%s)...', args.model)
    t0 = time.monotonic()
    from vllm import LLM, SamplingParams
    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype='float16',
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_custom_all_reduce=True,
    )
    tok = engine.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
    )
    log.info('Engine ready in %.1fs', time.monotonic() - t0)

    run_name = f'stage1-{args.split}-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    results = list(existing)

    # ---- Round 1 across ALL topics ----
    log.info('Round 1: building %d prompts (%d topics × %d personas)',
             len(remaining) * len(all_personas), len(remaining), len(all_personas))
    r1_prompts, r1_keys = [], []
    for topic in remaining:
        for persona in all_personas:
            r1_prompts.append(format_chat(
                build_r1_prompt(topic['topic_text'], persona, args.r1_args), tok))
            r1_keys.append((topic['topic_id'], persona['id']))

    t = time.monotonic()
    r1_raw = engine.generate(r1_prompts, sampling)
    r1_texts = [o.outputs[0].text.strip() for o in r1_raw]
    log.info('Round 1: %d prompts done in %.1fs', len(r1_prompts), time.monotonic() - t)

    # Collect R1 outputs
    r1_by_topic = {t['topic_id']: {} for t in remaining}
    persona_by_id = {p['id']: p for p in all_personas}
    for (tid, pid), raw in zip(r1_keys, r1_texts):
        parsed = parse_json_list(raw)
        items = [str(x).strip() for x in parsed if str(x).strip()][:args.r1_args]
        r1_by_topic[tid][pid] = {
            'persona': persona_by_id[pid],
            'arguments': items,
            'raw': raw[:300],
        }

    # ---- Round 2 across ALL topics (needs R1 opposing args) ----
    log.info('Round 2: building prompts with opposing context')
    r2_prompts, r2_keys = [], []
    for topic in remaining:
        tid = topic['topic_id']
        pro_args_all = [a for p in active_pro
                        for a in r1_by_topic[tid][p['id']]['arguments']]
        con_args_all = [a for p in active_con
                        for a in r1_by_topic[tid][p['id']]['arguments']]
        for persona in active_pro:
            r2_prompts.append(format_chat(
                build_r2_prompt(topic['topic_text'], persona, con_args_all,
                                args.r2_args), tok))
            r2_keys.append((tid, persona['id'], 'PRO'))
        for persona in active_con:
            r2_prompts.append(format_chat(
                build_r2_prompt(topic['topic_text'], persona, pro_args_all,
                                args.r2_args), tok))
            r2_keys.append((tid, persona['id'], 'CON'))

    t = time.monotonic()
    if r2_prompts:
        r2_raw = engine.generate(r2_prompts, sampling)
        r2_texts = [o.outputs[0].text.strip() for o in r2_raw]
    else:
        r2_texts = []
    log.info('Round 2: %d prompts done in %.1fs', len(r2_prompts), time.monotonic() - t)

    r2_by_topic = {t['topic_id']: {} for t in remaining}
    for (tid, pid, stance), raw in zip(r2_keys, r2_texts):
        parsed = parse_json_list(raw)
        counters = []
        for x in parsed:
            if isinstance(x, dict) and 'argument' in x:
                counters.append({
                    'targets_arg': x.get('targets_arg'),
                    'argument': str(x['argument']).strip(),
                })
            elif isinstance(x, str):
                counters.append({'targets_arg': None, 'argument': x.strip()})
            if len(counters) >= args.r2_args:
                break
        r2_by_topic[tid][pid] = {
            'persona': persona_by_id[pid],
            'counter_args': counters,
            'raw': raw[:300],
        }

    # ---- Assemble into the schema stage2 consumes ----
    for topic in remaining:
        tid = topic['topic_id']
        flat, idx = [], 0
        for persona in active_pro:
            for arg in r1_by_topic[tid][persona['id']]['arguments']:
                flat.append({
                    'arg_id': f'{tid}_A{idx:03d}',
                    'persona_id': persona['id'],
                    'persona': persona['name'],
                    'stance': 'PRO', 'round': 1, 'targets_arg': None,
                    'text': arg,
                })
                idx += 1
        for persona in active_con:
            for arg in r1_by_topic[tid][persona['id']]['arguments']:
                flat.append({
                    'arg_id': f'{tid}_A{idx:03d}',
                    'persona_id': persona['id'],
                    'persona': persona['name'],
                    'stance': 'CON', 'round': 1, 'targets_arg': None,
                    'text': arg,
                })
                idx += 1
        for persona in active_pro:
            for ca in r2_by_topic[tid][persona['id']]['counter_args']:
                flat.append({
                    'arg_id': f'{tid}_A{idx:03d}',
                    'persona_id': persona['id'],
                    'persona': persona['name'],
                    'stance': 'PRO', 'round': 2,
                    'targets_arg': ca.get('targets_arg'),
                    'text': ca['argument'],
                })
                idx += 1
        for persona in active_con:
            for ca in r2_by_topic[tid][persona['id']]['counter_args']:
                flat.append({
                    'arg_id': f'{tid}_A{idx:03d}',
                    'persona_id': persona['id'],
                    'persona': persona['name'],
                    'stance': 'CON', 'round': 2,
                    'targets_arg': ca.get('targets_arg'),
                    'text': ca['argument'],
                })
                idx += 1

        r1_count = sum(len(r1_by_topic[tid][p['id']]['arguments']) for p in all_personas)
        r2_count = sum(len(r2_by_topic[tid][p['id']]['counter_args']) for p in all_personas)
        results.append({
            'topic_id': tid,
            'topic_text': topic['topic_text'],
            'domain': topic['domain'],
            'benchmark_label': topic['benchmark_label'],
            'source_dataset': topic['source_dataset'],
            'source_ref': topic['source_ref'],
            'evaluation_split': args.split,
            'run_name': run_name,
            'arguments': flat,
            'meta': {
                'n_pro': args.n_pro, 'n_con': args.n_con,
                'r1_per_agent': args.r1_args, 'r2_per_agent': args.r2_args,
                'total_args': len(flat),
                'r1_args': r1_count, 'r2_args': r2_count,
                'model': str(args.model), 'provider': 'vllm-local',
            },
        })

    summary = {
        'total_topics': len(results),
        'total_args': sum(t['meta']['total_args'] for t in results),
        'model': str(args.model),
        'provider': 'vllm-local',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
    }
    final = {'topics': results, 'summary': summary}

    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(out_path)
    log.info('Wrote %s (%d topics, %d total args)',
             out_path, len(results), summary['total_args'])


if __name__ == '__main__':
    main()
