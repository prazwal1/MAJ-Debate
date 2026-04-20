#!/usr/bin/env python3
"""
Baseline judges — Single-LLM and CoT — using vLLM only.

These skip the entire MAJ-Debate pipeline and ask a single Qwen model to
judge the debate from the topic alone, or with chain-of-thought. They
provide the "no graph, no agents" floor in the ablation table.

Output schema matches stage4_judge.py so evaluate_ablations.py can mix
them uniformly.

Modes:
    --mode single   — direct topic-only verdict, no reasoning scaffolding
    --mode cot      — same but with "think step by step" reasoning prefix

Arguments:
    --topic-file <path>     topic .jsonl (same file Stage 1 consumes)
    --split <n>             used only for run metadata
    --output <path>         stage4-style judgments JSON
    --model <path>
    --topic-limit <int>
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path


log = logging.getLogger('baseline')


# ---------------------------------------------------------------------------
# Topic loading (same as stage1_vllm)
# ---------------------------------------------------------------------------

def normalize_topic(raw, idx, split):
    tid = raw.get('topic_id') or raw.get('id') or f'{split.upper()}_{idx:04d}'
    text = raw.get('topic_text') or raw.get('text')
    if not text:
        raise ValueError(f'missing text for record {tid}')
    return {
        'topic_id': tid,
        'topic_text': text,
        'domain': raw.get('domain', 'unknown'),
        'benchmark_label': raw.get('benchmark_label'),
        'source_dataset': raw.get('source_dataset', split),
    }


def load_topics(path, split):
    p = Path(path)
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
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    'You are an impartial debate judge. Respond ONLY with valid JSON '
    '(no markdown fences, no prose preamble).'
)

SINGLE_TEMPLATE = (
    'Debate topic: "{topic}"\n\n'
    'Decide which side would win a debate on this topic based on the merits '
    'of typical arguments each side would make.\n\n'
    'Output JSON:\n'
    '{{"verdict": "PRO" or "CON" or "TIE", '
    '"confidence": 0.0-1.0, '
    '"rationale": "<= 40 words"}}'
)

COT_TEMPLATE = (
    'Debate topic: "{topic}"\n\n'
    'Think through the main arguments on each side step by step, then decide '
    'which side would win. First list the top PRO argument and top CON '
    'argument, compare them, then produce the verdict.\n\n'
    'Output JSON (NO chain-of-thought in the output, just the final verdict):\n'
    '{{"verdict": "PRO" or "CON" or "TIE", '
    '"confidence": 0.0-1.0, '
    '"rationale": "<= 40 words summarising the decisive factor", '
    '"top_pro": "one sentence", "top_con": "one sentence"}}'
)


def build_prompt(topic_text, mode):
    if mode == 'cot':
        return COT_TEMPLATE.format(topic=topic_text)
    return SINGLE_TEMPLATE.format(topic=topic_text)


def format_chat(prompt, tok):
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def parse_verdict(raw):
    """Parse verdict JSON with partial-recovery fallback for truncated output."""
    if not raw:
        return None
    t = re.sub(r'```(?:json)?', '', raw).strip()
    # Full parse
    try:
        v = json.loads(t)
        if isinstance(v, dict) and 'verdict' in v:
            return v
    except Exception:
        pass
    # Brace-balanced
    start = t.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == '{':
                depth += 1
            elif t[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict) and 'verdict' in v:
                            return v
                    except Exception:
                        break
    # Partial recovery from truncated output
    m = re.search(r'"verdict"\s*:\s*"(PRO|CON|TIE)"', t, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper()
        cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', t)
        confidence = float(cm.group(1)) if cm else 0.5
        rm = re.search(r'"rationale"\s*:\s*"([^"]{0,400})', t)
        rationale = (rm.group(1) if rm else '') + ' [recovered]'
        return {
            'verdict': verdict,
            'confidence': confidence,
            'rationale': rationale,
            '_partial_recovery': True,
        }
    return None


def coerce(j):
    v = str(j.get('verdict', 'TIE')).strip().upper()
    if v not in ('PRO', 'CON', 'TIE'):
        v = 'TIE'
    try:
        c = float(j.get('confidence', 0.0))
    except (TypeError, ValueError):
        c = 0.0
    return {
        'verdict': v,
        'confidence': round(max(0.0, min(1.0, c)), 3),
        'rationale': str(j.get('rationale', ''))[:400],
    }


def compute_agreement(judgments):
    total = agree = 0
    for j in judgments:
        gold = j.get('benchmark_label')
        pred = j.get('verdict')
        if gold and pred and gold != 'TIE':
            total += 1
            if gold == pred:
                agree += 1
    return (round(100.0 * agree / total, 2) if total else None), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic-file', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--mode', choices=['single', 'cot'], required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--topic-limit', type=int, default=0)
    ap.add_argument('--gpu-mem-util', type=float, default=0.80)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-tokens', type=int, default=400)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s baseline %(levelname)-7s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    for lib in ['transformers', 'torch', 'vllm', 'urllib3', 'httpx']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    topics = load_topics(args.topic_file, args.split)
    log.info('Loaded %d topics (mode=%s)', len(topics), args.mode)
    if args.topic_limit > 0:
        topics = topics[:args.topic_limit]
        log.info('Topic limit -> %d', len(topics))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing, done = [], set()
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text())
            existing = d.get('judgments', [])
            done = {j['topic_id'] for j in existing}
        except Exception:
            pass
    remaining = [t for t in topics if t['topic_id'] not in done]
    log.info('Resume: %d already done, %d to go', len(done), len(remaining))

    if remaining:
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
            temperature=0.0 if args.mode == 'single' else 0.3,
            max_tokens=args.max_tokens,
            repetition_penalty=1.05,
        )
        log.info('Engine ready in %.1fs', time.monotonic() - t0)

        prompts = [format_chat(build_prompt(t['topic_text'], args.mode), tok)
                   for t in remaining]
        t = time.monotonic()
        outs = engine.generate(prompts, sampling)
        raws = [o.outputs[0].text.strip() for o in outs]
        log.info('Generated in %.1fs', time.monotonic() - t)

        parse_fail = 0
        for topic, raw in zip(remaining, raws):
            parsed = parse_verdict(raw)
            if parsed is None:
                parse_fail += 1
                parsed = {'verdict': 'TIE', 'confidence': 0.0,
                          'rationale': 'parse_failed'}
            j = coerce(parsed)
            existing.append({
                'topic_id': topic['topic_id'],
                'topic_text': topic['topic_text'],
                'domain': topic['domain'],
                'benchmark_label': topic['benchmark_label'],
                'source_dataset': topic['source_dataset'],
                'verdict': j['verdict'],
                'confidence': j['confidence'],
                'rationale': j['rationale'],
                'killing_attacks': [],
                'used_graph': False,
                'raw_output_preview': raw[:200],
                'baseline_mode': args.mode,
            })
        log.info('Parse failures: %d/%d', parse_fail, len(remaining))

    from collections import Counter
    vc = Counter(j['verdict'] for j in existing)
    pct, n = compute_agreement(existing)
    final = {
        'judgments': existing,
        'summary': {
            'n_topics': len(existing),
            'verdict_counts': dict(vc),
            'agreement_with_benchmark_pct': pct,
            'agreement_with_benchmark_n': n,
            'baseline_mode': args.mode,
            'model': str(args.model),
            'generated_at': datetime.now().isoformat(timespec='seconds'),
        },
    }
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(out_path)
    log.info('Wrote %s (n=%d, agreement=%s%%)', out_path, len(existing), pct)


if __name__ == '__main__':
    main()